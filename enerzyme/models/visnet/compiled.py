"""Keep ViSNet's terminal vector-only weights outside the compiled energy VJP."""

from functools import partial

import torch

from ..layers.compiled_grad import _input_fields, _restore_input_type, forward_with_coordinate_gradients
from ..layers.gradient import ForceLayer
from ..layers.readout import SimpleReadout
from ..layers.reduce import EnergyReduceLayer
from ..layers.tensorial import EquivariantDipoleLayer, EquivariantPolarizabilityLayer
from ..mp.block import FnBlock
from ..mp.core import ComposableCore
from ..mp.message.vector import AngularFeatureHead, edge_attention_step
from ..mp.ops import scatter_sum
from .interaction import ViSInteraction


_VECTOR_INPUTS = tuple("_visnet_" + field for field in (
    "scalar_message", "vector_n", "vec3", "out_scale", "idx_i", "idx_j", "direction", "vector"
))


def _active(layer, core):
    return not ((layer.train_only and not layer.training) or (layer.eval_only and layer.training)
                or (layer.test_only and not core.test_mode) or (layer.test_exclude and core.test_mode))


def _has_hooks(module):
    return any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
               for m in module.modules())


def _deferred_layers(core, inputs):
    """Select independent built-in vector heads and the suffix already after Force."""
    if type(core) is not ComposableCore or not core.schedule or any(key in inputs for key in _VECTOR_INPUTS):
        return None
    last = core.blocks[core.schedule[-1]]
    if (type(last) is not FnBlock or last._fn is not edge_attention_step
            or type(last.module) is not ViSInteraction or not last.module.last_layer
            or type(core.head) is not AngularFeatureHead):
        return None
    layers = core.post_sequence
    forces = [i for i, layer in enumerate(layers) if type(layer) is ForceLayer and _active(layer, core)]
    if len(forces) != 1:
        return None
    force = forces[0]
    # All earlier message-passing blocks still run in their original order. Only the
    # final block, head and crossed post layers can observe a different hook boundary.
    if _has_hooks(last) or _has_hooks(core.head):
        return None
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
        if _has_hooks(layer):
            return None
        ins = {layer._name_mapping[k] for k in layer._input_fields}
        outs = {layer._name_mapping[k] for k in layer._output_fields}
        if outs.intersection(_VECTOR_INPUTS):
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
    if set(_VECTOR_INPUTS).intersection(core._name_mapping.values()):
        return None
    if any(set(_VECTOR_INPUTS).intersection(layer._name_mapping.values()) for layer in core.pre_sequence):
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
    receiver, sender, _, window, direction = ctx.cache["attention_edges"]
    last = core.blocks[core.schedule[-1]].module
    delta_scalar, vector_inputs = last._update(
        state.scalar, state.equi, state.edge, receiver, sender, window, direction, defer_vector=True
    )
    return {
        **inputs, core._name_mapping["atom_feature"]: core.head.out_norm(state.scalar + delta_scalar),
        **dict(zip(_VECTOR_INPUTS, (*vector_inputs, receiver, sender, direction, state.equi))),
    }


def _vector_outputs(core, state):
    outputs = state.copy()
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
    deferred = _deferred_layers(core, inputs)
    if deferred is None:
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
