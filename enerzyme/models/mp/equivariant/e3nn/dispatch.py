"""Use e3nn's scripted kernel eagerly and trace the same generated FX operations."""
import pickle

import torch
from e3nn import o3
from e3nn.o3._spherical_harmonics import _spherical_harmonics
from e3nn.util import jit

_SCRIPTED_SH = torch.jit.script(_spherical_harmonics)


def _dispatch_spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor,
                                  z: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting() or torch.compiler.is_compiling():
        return _spherical_harmonics(lmax, x, y, z)
    return _SCRIPTED_SH(lmax, x, y, z)


class _GeneratedKernel(torch.nn.Module):
    """One registered set of constants for the two representations of an e3nn kernel."""

    def __init__(self, graph):
        super().__init__()
        self._bind(graph)

    def _bind(self, graph):
        # Both representations stay outside the module hierarchy. The wrapper
        # owns their constants under the exact names used in old checkpoints.
        self._buffers = graph._buffers
        self._non_persistent_buffers_set = graph._non_persistent_buffers_set
        object.__setattr__(self, "_fx", graph)
        native = jit.compile(graph, recurse=False)
        object.__setattr__(self, "_script", native)
        object.__setattr__(self, "_script_forward", native.forward)
        self._sync_script_buffers()

    def _sync_script_buffers(self):
        for name, value in self._buffers.items():
            setattr(self._script, name, value)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self._sync_script_buffers()
        return result

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        native = self.__dict__.get("_script")
        if native is not None and name in self._buffers:
            # Also covers load_state_dict(assign=True). Ordinary loads copy in place.
            setattr(native, name, value)

    def train(self, mode=True):
        # These graphs have no children. Repeated model.train()/eval() calls
        # need no state update when this kernel is already in the requested mode.
        if mode is self.training:
            return self
        super().train(mode)
        self._script.train(mode)
        self._fx.train(mode)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        graph = state.pop("_fx")
        state.pop("_script")
        state.pop("_script_forward")
        # Let torch.load(map_location=...) remap the registered buffers.
        # Store just the generated code and the shape/type of its attributes.
        shapes = {name: buf.to("meta") for name, buf in self._buffers.items()}
        state["_generated_code"] = pickle.dumps(torch.fx.GraphModule(shapes, graph.graph))
        return state

    def __setstate__(self, state):
        graph = pickle.loads(state.pop("_generated_code"))
        super().__setstate__(state)
        graph._buffers = self._buffers
        graph._non_persistent_buffers_set = self._non_persistent_buffers_set
        self._bind(graph)
        self._script.train(self.training)
        self._fx.train(self.training)

    def __prepare_scriptable__(self):
        return self._script

    def forward(self, *args):
        if torch.compiler.is_compiling():
            return self._fx(*args)
        return self._script_forward(*args)


def install_generated_kernels(model):
    """Install dispatch only on generated, parameter-free tensor operations."""
    scripted = False
    for owner in list(model.modules()):
        names = getattr(owner, "__codegen__", ())
        if names:
            managed_by_e3nn = []
            for name in names:
                graph = getattr(owner, name)
                if (isinstance(graph, torch.fx.GraphModule) and not graph._parameters
                        and not graph._modules
                        and all(n.target in graph._buffers for n in graph.graph.nodes if n.op == "get_attr")):
                    setattr(owner, name, _GeneratedKernel(graph))
                else:
                    if isinstance(graph, torch.fx.GraphModule):
                        setattr(owner, name, jit.compile(graph, recurse=False))
                    managed_by_e3nn.append(name)
                    scripted = True
            # e3nn's serializer handles stock ScriptModules and FX GraphModules.
            # Our wrapper serializes its own graph and keeps the original storage keys.
            owner.__codegen__ = managed_by_e3nn
        if type(owner) is o3.SphericalHarmonics and owner.sph_func is _spherical_harmonics:
            owner.sph_func = _dispatch_spherical_harmonics
    model._scripted_force_kernels = scripted
    return model
