"""Build force fields from a preset and its configured pipeline of layers."""

from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Literal, Optional

from torch.nn import Module

from ..utils import logger
from . import layers as Layers
from ._e3nn_codegen import construction_lock, traceable_e3nn_kernels


def build_layer(layer: Callable, layer_params: Dict[str, Any], build_params: Dict[str, Any], built_layers: Optional[Dict[str, Module]]=None) -> Module:
    final_params = dict()
    var_keyword = False
    for name, attr in signature(layer).parameters.items():
        if name in layer_params:
            final_params[name] = layer_params[name]
        elif name in build_params:
            final_params[name] = build_params[name]
        elif name == "built_layers" and built_layers is not None:
            final_params[name] = built_layers
        else:
            if attr.kind == Parameter.VAR_KEYWORD:
                var_keyword = True
            elif attr.default is attr.empty:
                raise TypeError(f"{name} value should be provided")
    if var_keyword:
        for key, value in layer_params.items():
            if key not in final_params:
                final_params[key] = value
    return layer(**final_params)


def build_model(
    architecture: str,
    layer_params: Optional[List[Dict[Literal["name", "params"], Any]]]=None,
    build_params: Optional[Dict]=None,
    verbose: int=1
) -> Module:
    from .ff import get_ff_core

    Core, default_build_params, default_layer_params = get_ff_core(architecture)
    if layer_params is None:
        layer_params = default_layer_params
    if build_params is None:
        build_params = default_build_params
    e3nn_codegen = build_params.get("e3nn_codegen", "default")
    if e3nn_codegen == "default":
        with construction_lock:
            return _build_layers(Core, layer_params, build_params, verbose)
    if e3nn_codegen != "fx":
        raise ValueError("e3nn_codegen must be 'default' or 'fx'")

    with traceable_e3nn_kernels():
        core = _build_layers(Core, layer_params, build_params, verbose)
    if architecture.casefold() == "mace":
        from .mace.compile import prepare_edge_tensor_products

        prepare_edge_tensor_products(core)
    elif getattr(core, "architecture", None) == "nequix":
        core._cpu_force_training_option = "cpp.simdlen"
    elif getattr(core, "architecture", None) in ("equflashv2", "equiformer"):
        # PyTorch 2.13 CPU kernels miscompute Equiformer forces and can abort
        # in the EquFlashV2 force-loss backward.
        core._native_cpu_inductor_forces = True
    return core


def _build_layers(Core, layer_params, build_params, verbose):
    built_layers = []
    core = None
    for layer_param in layer_params:
        layer_name = layer_param["name"]
        params = layer_param.get("params", dict())
        if layer_name == "Core":
            Layer = Core
        elif hasattr(Layers, layer_name + "Layer"):
            Layer = getattr(Layers, layer_name + "Layer")
        elif hasattr(Layers, layer_name):
            Layer = getattr(Layers, layer_name)
        else:
            from .compat.layers import LEGACY_LAYERS
            if layer_name not in LEGACY_LAYERS:
                raise NotImplementedError(f"Layer {layer_name} not implemented")
            Layer = LEGACY_LAYERS[layer_name]

        layer = build_layer(Layer, params, build_params, built_layers)
        if layer_name == "Core":
            core = layer
        if verbose:
            logger.info(f"built {layer_name}")
        built_layers.append(layer)

    if hasattr(core, "build"):
        core.build(built_layers)
    return core
