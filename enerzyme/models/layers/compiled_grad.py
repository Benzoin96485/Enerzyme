"""Compile force-field inference partitions or joint force-training derivatives."""

from functools import partial
from typing import Any, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data

from .geometry import DistanceLayer, RangeSeparationLayer
from .gradient import ForceLayer


@torch.compiler.assume_constant_result
def compiling_fullgraph() -> bool:
    """Keep strict traces in the joint energy/force transform.

    Dynamo supplies fullgraph mode on its translator, with no public
    torch.compiler predicate. Read it at trace time only. Other tracing
    contexts continue to use the self-contained functional VJP.
    """
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        return InstructionTranslator.current_tx().one_graph
    except AttributeError:
        return True


@torch.compiler.assume_constant_result
def _cpu_backend_may_use_inductor() -> bool:
    """Custom backends may delegate to Inductor. Eager backends do no kernel lowering."""
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        wrapper = InstructionTranslator.current_tx().output.compiler_fn
    except AttributeError:
        return True
    backend = getattr(wrapper, "_torchdynamo_orig_backend", wrapper)
    compiler = getattr(backend, "compiler_fn", backend)
    return all(compiler is not lookup_backend(name) for name in ("eager", "aot_eager"))


def uses_cpu_inductor_forces(core, inputs, *, include_eval=False, dtype=None):
    """Detect an active force layer on CPU coordinates, optionally of the given dtype."""
    for stages in (core.pre_sequence, core.post_sequence):
        for layer in stages:
            if type(layer) is not ForceLayer:
                continue
            if (layer.training and layer.eval_only) or (not layer.training and (not include_eval or layer.train_only)):
                continue
            if (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode):
                continue
            coordinates = inputs.get(layer._name_mapping["Ra"])
            if (coordinates is not None and coordinates.device.type == "cpu"
                    and (dtype is None or coordinates.dtype == dtype)):
                return _cpu_backend_may_use_inductor()
    return False


@torch.compiler.disable
def native_forces(core, inputs):
    """Run the original force layers with native coordinate and weight derivatives."""
    return core._run_layers(inputs)


@torch.compiler.disable
def has_module_hooks(root):
    """Read live callbacks before using a compiled path that moves an output head."""
    return any(module._forward_hooks or module._forward_pre_hooks or
               module._backward_hooks or module._backward_pre_hooks for module in root.modules())


def has_force_hooks(core):
    """The joint VJP replaces ForceLayer.forward, including its hook boundary."""
    from torch.nn.modules import module

    global_hooks = (module._global_forward_hooks or module._global_forward_pre_hooks
                    or module._global_backward_hooks or module._global_backward_pre_hooks)
    for stages in (core.pre_sequence, core.post_sequence):
        for layer in stages:
            if (type(layer) is not ForceLayer or (layer.train_only and not layer.training)
                    or (layer.eval_only and layer.training)):
                continue
            if (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode):
                continue
            if (global_hooks or layer._forward_hooks or layer._forward_pre_hooks
                    or layer._backward_hooks or layer._backward_pre_hooks):
                return True
    return False


@torch.compiler.assume_constant_result
def cpu_inductor_options(options):
    """Read explicit Inductor backend overrides before entering its lowering."""
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        wrapper = InstructionTranslator.current_tx().output.compiler_fn
    except AttributeError:
        return (None,) * len(options)
    backend = getattr(wrapper, "_torchdynamo_orig_backend", wrapper)
    overrides = getattr(backend, "config", None)
    if overrides is None:
        # torch.compile forwards options to custom backends as keyword arguments.
        overrides = getattr(backend, "kwargs", {}).get("options", {})
    if not isinstance(overrides, dict):
        overrides = {}
    return tuple(overrides.get(key) for key in options)


@torch.compiler.disable(recursive=False)
def compiled_cpu_forces_with_options(core, inputs, options):
    """Lower the forward and backward under the preset's required CPU kernel options."""
    from torch._functorch import config as autograd_config
    from torch._inductor import config

    if not (all(hasattr(config.cpp, key.removeprefix("cpp.")) for key in options)
            and hasattr(autograd_config, "force_non_lazy_backward_lowering")):
        return native_forces(core, inputs)
    # Training's weight backward happens after this scope. During inference
    # the Force layer runs its retained backward here. Let AOT lower that
    # backward on its first use so it records that the caller can reuse it.
    with config.patch(dict.fromkeys(options, 1)), autograd_config.patch(
        force_non_lazy_backward_lowering=core.training
    ):
        if core.training:
            return compiled_coordinate_vjp(core, inputs)
        return retained_inference(core, inputs)


@torch.compiler.disable
def _prepare_radius_graph(core, inputs):
    """Choose discrete neighbors once, before tracing the differentiable geometry."""
    if len(core.pre_sequence) < 2:
        return None, None
    distance, ranges = core.pre_sequence[0], core.pre_sequence[1]
    if type(distance) is not DistanceLayer or type(ranges) is not RangeSeparationLayer:
        return None, None
    for layer in (distance, ranges):
        if (layer.train_only or layer.eval_only or layer.test_only or layer.test_exclude
                or layer._forward_hooks or layer._forward_pre_hooks):
            return None, None
    if distance._name_mapping["Dij"] != ranges._name_mapping["Dij_lr"]:
        return None, None
    with torch.no_grad():
        fields = {k: inputs.get(distance._name_mapping[k]) for k in distance._input_fields}
        geometry = distance.get_output(**fields)
        edges = torch.where(geometry["Dij"] < ranges.cutoff_sr)[0]
        embed = getattr(core, "embed", None)
        block_index = getattr(core, "_force_topology_block", None)
        provider = embed if block_index is None else core.blocks[block_index]
        if getattr(provider, "prepare_force_topology", None) is not None:
            names = [core._name_mapping.get(k, k) for k in provider.topology_fields]
            if isinstance(inputs, Data):
                names = ["batch" if k == "batch_seg" else name for k, name in zip(provider.topology_fields, names)]
            changed = any(layer._forward_hooks or layer._forward_pre_hooks or
                          any(layer._name_mapping[k] in names for k in layer._output_fields)
                          for layer in core.pre_sequence)
            if block_index is not None:
                changed |= any(stage._forward_hooks or stage._forward_pre_hooks
                               for stage in (embed, *core.blocks[:block_index + 1]))
                changed |= any(p.is_floating_point() and p.dtype != geometry["Dij"].dtype
                               for p in core.parameters())
            if not changed and not embed._forward_pre_hooks:
                return edges, provider.prepare_force_topology(*(inputs.get(name) for name in names))
        return edges, _topology_indices(core, ranges, fields, geometry, edges)


def _topology_indices(core, ranges, fields, geometry, edges):
    """Build the declared integer topology for an unmodified short-range graph."""
    from ..mp.graph import GraphContext, PreparedTopology, QuadrupletRequest, ReverseEdgeRequest, TripletRequest

    requires = getattr(core, "requires", None)
    if requires is None or requires.edge_set != "sr" or not requires.topology:
        return None
    if any(type(r) not in (TripletRequest, QuadrupletRequest, ReverseEdgeRequest)
           and not isinstance(getattr(r, "force_cache_key", None), str) for r in requires.topology):
        return None
    # A user-defined preprocessing layer can replace the original geometry.
    # Only reuse indices when the Core consumes these exact distance-layer edges.
    names = ("idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr")
    if any(core._name_mapping.get(k, k) != ranges._name_mapping[k] for k in names):
        return None
    geometry_names = {ranges._name_mapping[n] for n in names}
    if any(layer._name_mapping[k] in geometry_names
           for layer in list(core.pre_sequence)[2:] for k in layer._output_fields):
        return None
    distance = core.pre_sequence[0]
    if any(ranges._name_mapping[dst] != distance._name_mapping[src] for src, dst in (
        ("idx_i", "idx_i_lr"), ("idx_j", "idx_j_lr"), ("vij", "vij_lr")
    )):
        return None
    if any(p.is_floating_point() and p.dtype != geometry["Dij"].dtype for p in core.parameters()):
        return None
    dist = geometry["Dij"][edges]
    vec = geometry["vij"][edges] if "vij" in geometry else None
    receiver, sender = fields["idx_i"][edges].long(), fields["idx_j"][edges].long()
    if requires.edge_convention == "swap":
        receiver, sender = sender, receiver
        vec = -vec if vec is not None else None
    ctx = GraphContext(receiver, sender, fields["Ra"].shape[0], dist=dist, vec=vec)
    return PreparedTopology.build(ctx, requires.topology)


@torch.compiler.disable
def _post_pair_indices(core, inputs):
    """Prepare discrete all-pairs membership before differentiating the distances."""
    prepared = {}
    if not any("_force_pair_fields" in type(layer).__dict__ for layer in core.post_sequence):
        return prepared
    preceding = [*core.pre_sequence, core]
    for i, layer in enumerate(core.post_sequence):
        fields = type(layer).__dict__.get("_force_pair_fields")
        active = not ((layer.eval_only and layer.training) or (layer.train_only and not layer.training)
                      or (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode))
        if fields and active and not layer._forward_hooks and not layer._forward_pre_hooks:
            names = {key: ("batch" if key == "batch_seg" and isinstance(inputs, Data) else layer._name_mapping[key])
                     for key in fields}
            # Hooks and earlier layers can change the coordinates, membership or boundary flags.
            changed = any(step._forward_hooks or step._forward_pre_hooks
                          or any(step._name_mapping[k] in names.values() for k in step._output_fields)
                          for step in preceding)
            if not changed and inputs.get(names["Ra"]) is not None:
                pairs = layer._prepare_force_pairs(**{key: inputs.get(name) for key, name in names.items()})
                if pairs is not None:
                    prepared[i] = pairs
        preceding.append(layer)
    return prepared


@torch.compiler.disable(recursive=False)
def compiled_coordinate_vjp(core, inputs):
    """Keep differentiable distances inside the VJP and discrete sizes outside it."""
    from torch._dynamo import config

    if getattr(core, "_scripted_force_kernels", False) or has_force_hooks(core):
        # Scripted kernels close over weights, and force hooks can replace
        # either the inputs or outputs of the coordinate differentiation.
        return native_forces(core, inputs)
    separate_vectors = core.training and getattr(core, "architecture", None) in ("visnet", "quinnet", "freecg", "visnet_lsrm")
    if separate_vectors:
        from ..presets.compiled_vector import has_training_hooks, train_with_vector_outputs
        if has_training_hooks(core):
            return native_forces(core, inputs)
    edges, topology = _prepare_radius_graph(core, inputs)
    if getattr(core, "_native_empty_short_range_forces", False) and edges is not None and edges.numel() == 0:
        # With no MACE messages, an energy/force loss can be independent of
        # the coordinates. Its joint VJP would materialize a zero derivative.
        return native_forces(core, inputs)
    post_pairs = _post_pair_indices(core, inputs)
    options = {"capture_dynamic_output_shape_ops": True, "capture_scalar_outputs": True}
    if hasattr(config, "skip_nnmodule_hook_guards"):
        # A cached joint VJP must be retraced when callbacks change the
        # energy or its inputs, including after the first compiled call.
        options["skip_nnmodule_hook_guards"] = False
    with config.patch(options):
        if separate_vectors:
            return train_with_vector_outputs(core, inputs, edges, topology, post_pairs)
        if core.training and getattr(core, "architecture", None) == "chgnet":
            from ..chgnet.stages import train_with_magnetic_readout
            return train_with_magnetic_readout(core, inputs, edges, topology, post_pairs)
        return forward_with_coordinate_gradients(core, inputs, edges, topology, post_pairs)


@torch.compiler.disable(recursive=False)
def retained_inference(core, inputs, *, joint=False):
    """Compile inference with reusable backward activations.

    Force evaluation keeps the energy graph for other response heads and for
    subsequent gradients requested by the caller. An eagerly lowered dynamic
    backward can otherwise overwrite its saved tensors on the first use.
    Only this configuration frame is skipped, so the layers still compile.
    """
    from contextlib import nullcontext
    from torch._functorch import config

    run = partial(compiled_coordinate_vjp, core) if joint else core._run_layers
    if not hasattr(config, "donated_buffer"):
        return run(inputs)

    # The Inductor FX cache key on PyTorch 2.13 omits AOTAutograd's donated
    # input indices. A cached single-use backward can overwrite buffers even
    # when this invocation requested reusable ones. Keep the caller's tag.
    compiler_config = getattr(torch.compiler, "config", None)
    if hasattr(compiler_config, "cache_key_tag"):
        cache_scope = compiler_config.patch(
            cache_key_tag=compiler_config.cache_key_tag + ":enerzyme-retained-backward"
        )
    else:
        cache_scope = nullcontext()
    with config.patch(donated_buffer=False), cache_scope:
        return run(inputs)


def forward_with_coordinate_gradients(core, inputs: Dict[str, Any], edges=None, topology=None,
                                      post_pairs=None, *, core_stage=None, skip_post=()) -> Dict[str, Any]:
    """Functional coordinate VJPs make force fitting a single compiled backward.

    The prefix ending at each Force layer returns energy and tensor-valued
    auxiliary outputs in one pass. This preserves random draws and lets losses
    on forces and other heads differentiate the same weights.
    """
    if torch.compiler.is_compiling():
        if compiling_fullgraph():
            torch._assert(not has_force_hooks(core),
                          "ForceLayer hooks require ordinary torch.compile with graph breaks")
    elif has_force_hooks(core):
        return core._run_layers(inputs)

    def active(layer):
        return not ((layer.test_only and not core.test_mode) or
                    (layer.test_exclude and core.test_mode))

    stages = [layer for layer in core.pre_sequence if active(layer)]
    if edges is not None:
        ranges = stages[1]

        def select_neighbors(state):
            fields = {k: state.get(ranges._name_mapping[k]) for k in ranges._input_fields}
            selected = ranges._select_edges(edges, **fields)
            return {**state, **{ranges._name_mapping[k]: v for k, v in selected.items()}}

        stages[1] = select_neighbors
    if core_stage is None:
        core_stage = core._forward if topology is None else partial(
            core._forward, extra_inputs={"_prepared_topology": topology}
        )
    stages.append(core_stage)
    for i, layer in enumerate(core.post_sequence):
        if i not in skip_post and active(layer):
            if post_pairs and i in post_pairs:
                stages.append(partial(layer._forward, extra_inputs={"_pair_indices": post_pairs[i]}))
            else:
                stages.append(layer)

    def evaluate(net_input, stop):
        # A later Force layer can differentiate quantities produced by an earlier
        # one. Only the innermost prefix runs the actual message-passing stack.
        start = 0
        for i in reversed(range(stop)):
            layer = stages[i]
            if type(layer) is not ForceLayer:
                continue
            if (layer.train_only and not layer.training) or (layer.eval_only and layer.training):
                continue
            position_name, energy_name, force_name = (layer._name_mapping[k] for k in ("Ra", "E", "Fa"))
            if position_name not in net_input:
                continue

            def energy(positions):
                state = evaluate({**net_input, position_name: positions}, i)
                # Non-tensor input metadata (e.g. a list of molecule sizes)
                # passes through the layer stack and does not enter the VJP.
                return state[energy_name], {key: value for key, value in state.items() if isinstance(value, Tensor)}

            energies, pullback, tensor_fields = torch.func.vjp(energy, net_input[position_name], has_aux=True)
            if energies.ndim > 1:
                weights = torch.eye(energies.shape[-1], device=energies.device, dtype=energies.dtype)
                forces = torch.stack([
                    -pullback(row.expand_as(energies))[0] for row in weights
                ], dim=-1)
            else:
                forces = -pullback(torch.ones_like(energies))[0]
            net_input = {**net_input, **tensor_fields, force_name: forces if layer.training else forces.detach()}
            start = i + 1
            break
        for i in range(start, stop):
            net_input = stages[i](net_input)
        return net_input

    return _restore_input_type(inputs, evaluate(_input_fields(inputs), len(stages)))


def _input_fields(inputs):
    if isinstance(inputs, Data):
        fields = {name: value for name, value in inputs}
        fields["batch_seg"] = fields.get("batch")
        return fields
    return inputs


def _restore_input_type(inputs, fields):
    if isinstance(inputs, Data):
        # PyG spells the membership vector "batch" and updates its Data in place.
        for name, value in fields.items():
            if name != "batch_seg":
                inputs[name] = value
        return inputs
    return fields
