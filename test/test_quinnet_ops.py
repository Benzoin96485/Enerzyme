"""QuinNet body-order channels versus the released formulas."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from enerzyme.models.mp.ops import scatter_sum
from enerzyme.models.layers.rejection import vector_rejection
from enerzyme.models.quinnet.interaction import BODY_TERMS, QuinNetInteraction


def test_channel_matched_rejection_and_bad_direction():
    direction = torch.zeros(2, 3, 2)
    direction[0, 0, 0] = 1.0
    direction[1, 1, 1] = 1.0
    features = torch.tensor(
        [
            [[2.0, -1.0], [0.3, 0.4], [0.0, 5.0]],
            [[0.2, 0.0], [4.0, -3.0], [1.0, 1.0]],
        ]
    )
    rejected = vector_rejection(features, direction)
    assert torch.allclose(rejected[0, 0, 0], torch.zeros(()), atol=1e-6)
    assert torch.allclose(rejected[0, 1:, 0], features[0, 1:, 0], atol=1e-6)
    assert torch.allclose(rejected[0, :, 1], features[0, :, 1], atol=1e-6)
    assert torch.allclose(rejected[1, 1, 1], torch.zeros(()), atol=1e-6)
    try:
        vector_rejection(features, torch.zeros(2, 4))
    except ValueError as exc:
        assert "direction" in str(exc)
    else:
        raise AssertionError("expected a bad direction shape to fail")


def test_body_term_order_and_distance_filters():
    assert BODY_TERMS == (
        "three_body",
        "five_body_i",
        "five_body_iii",
        "five_body_ii",
        "four_body_torsion",
        "four_body_improper",
    )
    both = QuinNetInteraction(4, 2, "both", last_layer=False)
    assert both.dk_proj is not None and both.dv_proj is not None
    keys = QuinNetInteraction(4, 2, "keys", last_layer=True)
    assert keys.dk_proj is not None and keys.dv_proj is None
    values = QuinNetInteraction(4, 2, "values", last_layer=True)
    assert values.dk_proj is None and values.dv_proj is not None
    none = QuinNetInteraction(4, 2, "none", last_layer=True)
    assert none.dk_proj is None and none.dv_proj is None
    none.ratio.data.copy_(torch.tensor([0.2, -0.5, 1.0, 0.0, 3.0, -1.0]))
    none.reset_parameters()
    assert torch.equal(none.ratio.data, torch.ones(6))
    try:
        QuinNetInteraction(4, 2, "edge", last_layer=True)
    except ValueError as exc:
        assert "distance_influence" in str(exc)
    else:
        raise AssertionError("expected an unknown distance influence to fail")
    try:
        QuinNetInteraction(5, 2, "both", last_layer=True)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected a bad head count to fail")


def _reference(layer, scalar, vector, edge, idx_i, idx_j, window, direction):
    n_atoms = scalar.shape[0]
    n_edges = idx_i.shape[0]
    hidden = layer.hidden_channels
    normed = F.layer_norm(
        scalar,
        (hidden,),
        layer.layernorm.weight,
        layer.layernorm.bias,
        layer.layernorm.eps,
    )
    query = F.linear(normed, layer.q_proj.weight, layer.q_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    key = F.linear(normed, layer.k_proj.weight, layer.k_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    value = F.linear(normed, layer.v_proj.weight, layer.v_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    projected = F.linear(vector, layer.vec_proj.weight, None)
    vec1, vec2, vec3 = torch.split(projected, hidden, dim=-1)
    three_body = (vec1 * vec2).sum(dim=1)
    normal_i = vector_rejection(
        F.linear(vector, layer.vec_proj_improper1.weight, None),
        F.linear(vector, layer.vec_proj_improper2.weight, None),
    )
    five_i = normal_i.square().sum(dim=1)
    improper = (
        vector_rejection(
            F.linear(vector, layer.vec_proj_improper3.weight, None),
            F.linear(vector, layer.vec_proj_improper4.weight, None),
        )
        * F.linear(vector, layer.vec_proj_improper5.weight, None)
    ).sum(dim=1)
    five_iii_edge = F.silu(F.linear(edge, layer.f_proj2.weight, layer.f_proj2.bias)) * (
        F.linear(normal_i[idx_j], layer.dst_proj1.weight, None)
        * F.linear(normal_i[idx_i], layer.src_proj1.weight, None)
    ).sum(dim=1)
    five_ii_edge = F.silu(F.linear(edge, layer.f_proj3.weight, layer.f_proj3.bias)) * (
        F.linear(normal_i[idx_i], layer.src_proj2.weight, None).square().sum(dim=1)
    )
    torsion_edge = F.silu(F.linear(edge, layer.f_proj1.weight, layer.f_proj1.bias)) * (
        vector_rejection(F.linear(vector[idx_i], layer.trg_proj.weight, None), direction)
        * vector_rejection(F.linear(vector[idx_j], layer.src_proj.weight, None), -direction)
    ).sum(dim=1)
    node = (n_atoms, hidden)
    five_iii = scatter_sum(five_iii_edge, idx_j, node[0])
    five_ii = scatter_sum(five_ii_edge, idx_j, node[0])
    torsion = scatter_sum(torsion_edge, idx_j, node[0])

    filter_key = None
    if layer.dk_proj is not None:
        filter_key = F.silu(F.linear(edge, layer.dk_proj.weight, layer.dk_proj.bias))
        filter_key = filter_key.reshape(n_edges, layer.num_heads, layer.head_dim)
    filter_value = None
    if layer.dv_proj is not None:
        filter_value = F.silu(F.linear(edge, layer.dv_proj.weight, layer.dv_proj.bias))
        filter_value = filter_value.reshape(n_edges, layer.num_heads, layer.head_dim)
    attention = query[idx_i] * key[idx_j]
    if filter_key is not None:
        attention = attention * filter_key
    attention = F.silu(attention.sum(dim=-1)) * window.reshape(-1, 1)
    value_j = value[idx_j]
    if filter_value is not None:
        value_j = value_j * filter_value
    scalar_message = (value_j * attention.unsqueeze(-1)).reshape(n_edges, hidden)
    gate, coeff = torch.split(
        F.silu(F.linear(scalar_message, layer.s_proj.weight, layer.s_proj.bias)),
        hidden,
        dim=-1,
    )
    vector_message = vector[idx_j] * gate.unsqueeze(1)
    vector_message = vector_message + coeff.unsqueeze(1) * direction.unsqueeze(-1)
    aggregated_scalar = scatter_sum(scalar_message, idx_i, node[0])
    aggregated_vector = scatter_sum(vector_message, idx_i, n_atoms)
    out_scale, out_gate, out_bias = torch.split(
        F.linear(aggregated_scalar, layer.o_proj.weight, layer.o_proj.bias),
        hidden,
        dim=-1,
    )
    terms = torch.stack(
        (three_body, five_i, five_iii, five_ii, torsion, improper), dim=0
    )
    mixed = (terms * layer.ratio.view(-1, 1, 1)).sum(dim=0)
    delta_scalar = mixed * out_gate + out_bias
    delta_vector = vec3 * out_scale.unsqueeze(1) + aggregated_vector
    delta_edge = None if layer.last_layer else torsion_edge + five_iii_edge + five_ii_edge
    return delta_scalar, delta_vector, delta_edge


def _block_inputs(seed: int, influence: str, last_layer: bool):
    generator = torch.Generator().manual_seed(seed)
    layer = QuinNetInteraction(4, 2, influence, last_layer=last_layer)
    scalar = torch.randn(3, 4, generator=generator)
    vector = torch.randn(3, 8, 4, generator=generator)
    edge = torch.randn(4, 4, generator=generator)
    idx_i = torch.tensor([0, 1, 2, 0])
    idx_j = torch.tensor([1, 2, 0, 0])
    window = torch.rand(4, generator=generator).clamp(0.2, 1)
    direction = torch.randn(4, 8, generator=generator)
    layer.ratio.data.copy_(torch.tensor([0.4, -0.2, 1.3, 0.7, -0.5, 0.9]))
    return layer, scalar, vector, edge, idx_i, idx_j, window, direction


def test_block_matches_the_released_mix_and_scatter():
    for influence, last_layer, seed in (
        ("both", False, 0),
        ("keys", False, 1),
        ("values", True, 2),
        ("none", True, 3),
    ):
        layer, scalar, vector, edge, idx_i, idx_j, window, direction = _block_inputs(
            seed, influence, last_layer
        )
        got = layer(scalar, vector, edge, idx_i, idx_j, window, direction)
        ref = _reference(layer, scalar, vector, edge, idx_i, idx_j, window, direction)
        assert torch.allclose(got[0], ref[0], atol=1e-5, rtol=1e-5)
        assert torch.allclose(got[1], ref[1], atol=1e-5, rtol=1e-5)
        if last_layer:
            assert got[2] is None and ref[2] is None
        else:
            assert torch.allclose(got[2], ref[2], atol=1e-5, rtol=1e-5)

    layer, scalar, vector, edge, idx_i, idx_j, window, direction = _block_inputs(
        4, "both", False
    )
    try:
        layer(scalar, vector, edge, idx_i, idx_j, window, direction[:, :3])
    except ValueError as exc:
        assert "direction" in str(exc)
    else:
        raise AssertionError("expected a mismatched angular width to fail")


def test_five_body_ii_lands_on_the_sender():
    torch.manual_seed(5)
    layer = QuinNetInteraction(2, 1, "none", last_layer=True)
    with torch.no_grad():
        for name, child in layer.named_parameters():
            if name != "ratio":
                child.zero_()
        layer.ratio.zero_()
        layer.ratio[3] = 1.0
        layer.f_proj3.bias.fill_(1.0)
        layer.f_proj3.weight.zero_()
        layer.vec_proj_improper1.weight.copy_(torch.eye(2))
        layer.vec_proj_improper2.weight.copy_(torch.eye(2))
        layer.src_proj2.weight.copy_(torch.eye(2))
        layer.o_proj.bias.zero_()
        layer.o_proj.bias[2:4] = 1.0
    vector = torch.zeros(2, 8, 2)
    vector[0, 0, 0] = 1.0
    vector[0, 1, 0] = 0.0
    direction_atom = torch.zeros(8)
    direction_atom[0] = 1.0
    normal = vector_rejection(vector, vector)
    assert torch.count_nonzero(normal[0, :, 0]) == 0
    vector[0, 1, 0] = 2.0
    scalar = torch.zeros(2, 2)
    edge = torch.zeros(1, 2)
    idx_i = torch.tensor([0])
    idx_j = torch.tensor([1])
    window = torch.ones(1)
    direction = direction_atom.view(1, 8)
    delta_scalar, delta_vector, delta_edge = layer(
        scalar, vector, edge, idx_i, idx_j, window, direction
    )
    assert delta_edge is None
    assert torch.count_nonzero(delta_vector) == 0
    assert torch.count_nonzero(delta_scalar[0]) == 0
    assert torch.count_nonzero(delta_scalar[1]) > 0
