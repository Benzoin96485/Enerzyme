"""Compile terminal vector updates separately from the scalar energy and force prefix."""

from functools import partial

import torch

from ..layers.compiled_grad import (
    _input_fields, _restore_input_type, forward_with_coordinate_gradients, has_module_hooks, native_forces,
)
from ..layers.gradient import ForceLayer
from ..layers.readout import SimpleReadout
from ..layers.reduce import EnergyReduceLayer
from ..layers.tensorial import EquivariantDipoleLayer, EquivariantPolarizabilityLayer
from ..mp.block import FnBlock
from ..mp.core import ComposableCore
from ..mp.message.vector import AngularFeatureHead, edge_attention_step
from ..mp.ops import scatter_sum
from ..quinnet.interaction import QuinNetInteraction
from ..visnet.interaction import ViSInteraction


_VECTOR_INPUTS = tuple("_visnet_" + field for field in (
    "scalar_message", "vector_n", "vec3", "out_scale", "idx_i", "idx_j", "direction", "vector"
))


_FRAGMENT_INPUTS = tuple("_lsrm_" + field for field in (
    "message", "group_vector", "aggregated_scalar", "normalized_vector", "atoms",
    "fragments", "direction", "vector", "short_vector",
))


def _active(layer, core):
    return not ((layer.train_only and not layer.training) or (layer.eval_only and layer.training)
                or (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode))


def has_training_hooks(core):
    # A full backward hook inside any interaction installs a custom autograd
    # function that cannot participate in the functional coordinate pullback.
    # Inspect live handles so hooks added after compilation also work.
    return has_module_hooks(core)


def _supports_terminal_update(core):
    last = core.blocks[core.schedule[-1]]
    if (type(last) is not FnBlock or "forward" in vars(last) or "forward" in vars(last.module)
            or "forward" in vars(core.head)):
        return False
    if core.architecture == "visnet_lsrm":
        from ..mp.nonlocal_ import FragmentLR
        from ..visnet.lsr import FragmentFusionHead, fragment_step

        return (last._fn is fragment_step and type(last.module) is FragmentLR
                and type(core.head) is FragmentFusionHead and type(core.head.short) is AngularFeatureHead
                and core.head.short.lmax == 1 and "forward" not in vars(core.head.short))
    eligible = (ViSInteraction, QuinNetInteraction)
    if core.architecture == "freecg":
        # FreeCG loads e3nn's tensor-product kernels only for its own preset.
        from ..freecg.interaction import FreeCGInteraction
        eligible = (FreeCGInteraction,)
    return (last._fn is edge_attention_step and type(last.module) in eligible and last.module.last_layer
            and type(core.head) is AngularFeatureHead)


def _deferred_layers(core, inputs):
    """Select independent built-in vector heads and the suffix already after Force."""
    if type(core) is not ComposableCore or not core.schedule or not _supports_terminal_update(core):
        return None
    private_fields = _FRAGMENT_INPUTS if core.architecture == "visnet_lsrm" else _VECTOR_INPUTS
    if any(key in inputs for key in private_fields):
        return None
    layers = core.post_sequence
    forces = [i for i, layer in enumerate(layers) if type(layer) is ForceLayer and _active(layer, core)]
    if len(forces) != 1:
        return None
    force = forces[0]
    deferred = []
    reads = set()
    writes = {core._name_mapping[k] for k in core.head.output_fields if k != "atom_feature"}
    if core._name_mapping["atom_feature"] in writes:
        return None
    for i, layer in enumerate(layers):
        if i > force:
            deferred.append(i)
            continue
        if not _active(layer, core):
            continue
        ins = {layer._name_mapping[k] for k in layer._input_fields}
        outs = {layer._name_mapping[k] for k in layer._output_fields}
        if outs.intersection(private_fields):
            return None
        if type(layer) in (EquivariantDipoleLayer, EquivariantPolarizabilityLayer):
            # These deterministic heads may cross independent scalar operations.
            # User-defined modules retain their original execution order.
            deferred.append(i)
            reads.update(ins)
            writes.update(outs)
        else:
            if type(layer) not in (SimpleReadout, EnergyReduceLayer, ForceLayer):
                return None
            # EnergyReduce discovers per-atom contributions from their field names.
            if type(layer) is EnergyReduceLayer and any(k.startswith("E") and k.endswith("a") for k in writes):
                return None
            if ins & writes or outs & (reads | writes):
                return None
    if set(private_fields).intersection(core._name_mapping.values()):
        return None
    if any(set(private_fields).intersection(layer._name_mapping.values()) for layer in core.pre_sequence):
        return None
    return tuple(deferred)


def _scalar_prefix(core, topology, inputs):
    fields = {key: inputs.get(core._name_mapping[key]) for key in core._input_fields}
    if topology is not None:
        fields["_prepared_topology"] = topology
    ctx = core.context(fields)
    state = core.embed(fields, ctx)
    for i in core.schedule[:-1]:
        state = core.blocks[i](state, ctx)
    if core.architecture == "visnet_lsrm":
        return _fragment_scalar_outputs(core, inputs, state, ctx)
    receiver, sender, _, window, direction = ctx.cache["attention_edges"]
    last = core.blocks[core.schedule[-1]].module
    delta_scalar, vector_inputs = last._update(
        state.scalar, state.equi, state.edge, receiver, sender, window, direction, defer_vector=True
    )
    return {
        **inputs, core._name_mapping["atom_feature"]: core.head.out_norm(state.scalar + delta_scalar),
        **dict(zip(_VECTOR_INPUTS, (*vector_inputs, receiver, sender, direction, state.equi))),
    }


def _fragment_scalar_outputs(core, inputs, state, ctx):
    from ..visnet.lsr import fragment_inputs

    args = fragment_inputs(state, ctx)
    _, _, _, _, atoms, fragments, _, direction = args
    last = core.blocks[core.schedule[-1]].module
    delta_scalar, vector_inputs = last._update(*args, defer_vector=True)
    short = state.aux["short"]
    scalar = torch.cat((core.head.short.out_norm(short.scalar), core.head.out_norm_long(state.scalar + delta_scalar)), dim=-1)
    return {
        **inputs, core._name_mapping["atom_feature"]: scalar,
        **dict(zip(_FRAGMENT_INPUTS, (*vector_inputs, atoms, fragments, direction, state.equi, short.equi))),
    }


def _fragment_vector_outputs(core, outputs):
    message, group, aggregated, normalized, atoms, fragments, direction, vector, short = (
        outputs.pop(key) for key in _FRAGMENT_INPUTS
    )
    last = core.blocks[core.schedule[-1]].module
    messages = last._vector_message(message, group, fragments, direction)
    delta = last._vector_residual(scatter_sum(messages, atoms, vector.shape[0]), aggregated, normalized)
    if core.head.short.vec_out_norm is not None:
        short = core.head.short.vec_out_norm(short)
    long = core.head.vec_out_norm_long(vector + delta)
    outputs[core._name_mapping["atom_vector_feature"]] = torch.cat((short, long), dim=-1).contiguous()
    return outputs


def _vector_outputs(core, state):
    outputs = state.copy()
    if core.architecture == "visnet_lsrm":
        return _fragment_vector_outputs(core, outputs)
    message, vector_n, vec3, scale, receiver, sender, direction, residual = (
        outputs.pop(key) for key in _VECTOR_INPUTS
    )
    last = core.blocks[core.schedule[-1]].module
    vector_message = last._vector_message(message, vector_n, sender, direction)
    aggregated = scatter_sum(vector_message, receiver, residual.shape[0])
    vector = residual + last._vector_residual(vec3, scale, aggregated)
    if core.head.vec_out_norm is not None:
        vector = core.head.vec_out_norm(vector)
    if core.head.lmax == 1:
        outputs[core._name_mapping["atom_vector_feature"]] = vector
    else:
        outputs[core._name_mapping["atom_vector_feature"]] = vector[:, :3].contiguous()
        outputs[core._name_mapping["atom_l2_feature"]] = vector[:, 3:].contiguous()
    return outputs


@torch.compiler.disable(recursive=False)
def train_with_vector_outputs(core, inputs, edges, topology, post_pairs):
    """Separate differentiable graphs so unused auxiliary outputs keep absent gradients."""
    if core.architecture == "visnet_lsrm" and getattr(core, "_force_topology_block", None) is not None:
        from ..visnet.long_range import FragmentTopology

        if not isinstance(topology, FragmentTopology):
            # A preprocessing layer or live hook may replace positions or fragment labels.
            return native_forces(core, inputs)
        if not topology.edges.numel() or (not core.embed.include_self and edges is not None and not edges.numel()):
            # Empty fragment updates return disconnected zeros in eager mode.
            # Reuse the selected topology so native autograd does not recluster.
            return native_forces(core, inputs, topology)
    deferred = _deferred_layers(core, inputs)
    # With no messages, returning the last block's empty projection inputs
    # makes AOT send zero gradients to otherwise disconnected earlier layers.
    if deferred is None or (
        not getattr(core.embed, "include_self", False) and (edges is None or edges.numel() == 0)
    ):
        return forward_with_coordinate_gradients(core, inputs, edges, topology, post_pairs)
    out = forward_with_coordinate_gradients(
        core, _input_fields(inputs), edges, topology, post_pairs,
        core_stage=partial(_scalar_prefix, core, topology), skip_post=deferred,
    )
    out = _vector_outputs(core, out)
    for i in deferred:
        layer = core.post_sequence[i]
        if not _active(layer, core):
            continue
        if post_pairs and i in post_pairs:
            out = layer._forward(out, extra_inputs={"_pair_indices": post_pairs[i]})
        else:
            out = layer(out)
    return _restore_input_type(inputs, out)
