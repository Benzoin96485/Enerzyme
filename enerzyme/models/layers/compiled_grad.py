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
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        wrapper = InstructionTranslator.current_tx().output.compiler_fn
    except AttributeError:
        return True
    backend = getattr(wrapper, "_torchdynamo_orig_backend", wrapper)
    name = getattr(backend, "compiler_name", getattr(backend, "__name__", None))
    return name not in ("eager", "aot_eager")


def uses_cpu_inductor_force_training(core, inputs):
    for stages in (core.pre_sequence, core.post_sequence):
        for layer in stages:
            if type(layer) is not ForceLayer or not layer.training or layer.eval_only:
                continue
            if (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode):
                continue
            coordinates = inputs.get(layer._name_mapping["Ra"])
            if coordinates is not None and coordinates.device.type == "cpu":
                return _cpu_backend_may_use_inductor()
    return False


@torch.compiler.disable
def native_cpu_force_training(core, inputs):
    # Force-loss double backward in the affected CPU kernels can abort the process.
    return core._run_layers(inputs)


@torch.compiler.assume_constant_result
def cpu_horizontal_fusion_option():
    """Read an explicit Inductor backend override before entering its lowering."""
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        wrapper = InstructionTranslator.current_tx().output.compiler_fn
    except AttributeError:
        return None
    backend = getattr(wrapper, "_torchdynamo_orig_backend", wrapper)
    return getattr(backend, "config", {}).get("cpp.max_horizontal_fusion_size")


@torch.compiler.disable(recursive=False)
def compiled_training_with_separate_reductions(core, inputs):
    """Lower both DPA3 derivative graphs before leaving the CPU fusion scope."""
    from torch._functorch import config as autograd_config
    from torch._inductor import config

    if not (hasattr(config.cpp, "max_horizontal_fusion_size")
            and hasattr(autograd_config, "force_non_lazy_backward_lowering")):
        return native_cpu_force_training(core, inputs)
    # RepFlow reduces contributions from both endpoints of an edge. CPU
    # horizontal fusion can drop one term in the force or its weight gradient.
    # AOT normally defers the weight backward until after this scope exits.
    with config.patch({"cpp.max_horizontal_fusion_size": 1}), autograd_config.patch(
        force_non_lazy_backward_lowering=True
    ):
        return compiled_training(core, inputs)


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
def compiled_training(core, inputs):
    """Keep differentiable distances inside the VJP and discrete sizes outside it."""
    from torch._dynamo import config

    if getattr(core, "_scripted_force_kernels", False):
        return _scripted_force_training(core, inputs)
    edges, topology = _prepare_radius_graph(core, inputs)
    post_pairs = _post_pair_indices(core, inputs)
    with config.patch(capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True):
        return forward_with_coordinate_gradients(core, inputs, edges, topology, post_pairs)


@torch.compiler.disable
def _scripted_force_training(core, inputs):
    """Use native higher derivatives for scripted kernels with closed-over weights."""
    return core._run_layers(inputs)


@torch.compiler.disable(recursive=False)
def retained_inference(core, inputs):
    """Compile each inference partition with reusable backward activations.

    Force evaluation keeps the energy graph for other response heads and for
    subsequent gradients requested by the caller. An eagerly lowered dynamic
    backward can otherwise overwrite its saved tensors on the first use.
    Only this configuration frame is skipped, so the layers still compile.
    """
    from contextlib import nullcontext
    from torch._functorch import config

    if not hasattr(config, "donated_buffer"):
        return core._run_layers(inputs)

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
        return core._run_layers(inputs)


def forward_with_coordinate_gradients(core, inputs: Dict[str, Any], edges=None, topology=None,
                                      post_pairs=None) -> Dict[str, Any]:
    """Functional coordinate VJPs make force fitting a single compiled backward.

    The prefix ending at each Force layer returns energy and tensor-valued
    auxiliary outputs in one pass. This preserves random draws and lets losses
    on forces and other heads differentiate the same weights.
    """
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
    stages.append(core._forward if topology is None else
                  partial(core._forward, extra_inputs={"_prepared_topology": topology}))
    for i, layer in enumerate(core.post_sequence):
        if active(layer):
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

    if isinstance(inputs, Data):
        # PyG spells the membership vector "batch" and updates its Data in place.
        fields = {name: value for name, value in inputs}
        fields["batch_seg"] = fields.get("batch")
        for name, value in evaluate(fields, len(stages)).items():
            if name != "batch_seg":
                inputs[name] = value
        return inputs
    return evaluate(inputs, len(stages))
