"""ViSNet runtime geometry and one block versus the reference formulas."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from enerzyme.models.mp.ops import scatter_sum
from enerzyme.models.layers.polynomial_harmonics import polynomial_spherical_harmonics
from enerzyme.models.layers.rejection import vector_rejection
from enerzyme.models.visnet.interaction import (
    EdgeEmbedding,
    VecLayerNorm,
    ViSInteraction,
    normalize_activation,
    normalize_vecnorm_type,
    normalize_vertex_type,
)


def test_names_and_axis_harmonics():
    assert normalize_vertex_type(None) == "none"
    assert normalize_vertex_type("None") == "none"
    assert normalize_vertex_type("Edge") == "edge"
    assert normalize_vecnorm_type("max-min") == "max_min"
    try:
        normalize_vertex_type("triplet")
    except ValueError as exc:
        assert "vertex_type" in str(exc)
    else:
        raise AssertionError("expected an unknown vertex to fail")
    try:
        normalize_activation("tanh")
    except ValueError as exc:
        assert "activation" in str(exc)
    else:
        raise AssertionError("expected an unknown activation to fail")

    axis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = polynomial_spherical_harmonics(axis, 2)
    sqrt3 = math.sqrt(3.0)
    expected = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, -sqrt3 / 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert torch.allclose(got, expected, atol=1e-6)


def test_rejection_removes_the_parallel_part():
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    features = torch.tensor([
        [[2.0, -1.0], [0.3, 0.4], [0.0, 5.0]],
        [[0.2, 0.0], [4.0, -3.0], [1.0, 1.0]],
    ])
    rejected = vector_rejection(features, direction)
    assert torch.allclose(rejected[0, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(rejected[0, 1:], features[0, 1:], atol=1e-6)
    assert torch.allclose(rejected[1, 1], torch.zeros(2), atol=1e-6)


def test_vector_norm_is_channel_scale_and_rotates():
    torch.manual_seed(0)
    raw = torch.randn(4, 3, 5)
    plain = VecLayerNorm(5, trainable=False, norm_type="none")
    assert torch.allclose(plain(raw), raw, atol=1e-6)
    plain.weight.data.fill_(2.0)
    assert torch.allclose(plain(raw), raw * 2, atol=1e-6)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    for kind in ("rms", "max_min"):
        block = VecLayerNorm(5, trainable=True, norm_type=kind)
        block.weight.data.copy_(torch.rand(5) + 0.3)
        left = block(raw)
        right = block(torch.einsum("ncf,cd->ndf", raw, rotation))
        rotated = torch.einsum("ncf,cd->ndf", left, rotation)
        assert torch.allclose(rotated, right, atol=1e-5, rtol=1e-5)
    zeros = torch.zeros(2, 8, 5)
    assert torch.count_nonzero(VecLayerNorm(5, False, "rms")(zeros)) == 0
    assert torch.count_nonzero(VecLayerNorm(5, False, "max_min")(zeros)) == 0
    try:
        plain(torch.randn(2, 4, 5))
    except ValueError as exc:
        assert "angular" in str(exc)
    else:
        raise AssertionError("expected a bad angular width to fail")


def test_compiled_vector_norm_preserves_zero_field_force_derivatives():
    torch.manual_seed(29)
    field = torch.randn(4, 8, 5, dtype=torch.double)
    probe = torch.randn_like(field)
    for kind in ("rms", "max_min"):
        norm = VecLayerNorm(5, trainable=True, norm_type=kind).double()
        compiled = torch.compile(norm, backend="eager", fullgraph=True)
        for only_degree_two in (False, True):
            x = field.clone()
            if only_degree_two:
                x[:, :3].zero_()
            else:
                x.zero_()
            x.requires_grad_()
            def derivatives(layer):
                output = layer(x)
                # Connect all-zero reference fields to x with exact zero derivatives.
                loss = (output * probe).sum() + 0 * x.square().sum()
                gradient, = torch.autograd.grad(loss, x, create_graph=True)
                hessian, = torch.autograd.grad((gradient * probe).sum(), x)
                return output, gradient, hessian

            torch.testing.assert_close(derivatives(compiled), derivatives(norm))


def test_edge_embedding_is_sum_of_endpoints():
    torch.manual_seed(1)
    block = EdgeEmbedding(3, 4)
    scalar = torch.randn(3, 4)
    rbf = torch.randn(4, 3)
    idx_i = torch.tensor([0, 1, 2, 0])
    idx_j = torch.tensor([1, 2, 2, 0])
    got = block(idx_i, idx_j, rbf, scalar)
    weight = F.linear(rbf, block.edge_proj.weight, block.edge_proj.bias)
    reference = (scalar[idx_i] + scalar[idx_j]) * weight
    assert torch.allclose(got, reference, atol=1e-6)


def _reference(layer, scalar, vector, edge, idx_i, idx_j, window, direction):
    n_atoms = scalar.shape[0]
    scalar_n = F.layer_norm(
        scalar,
        (layer.hidden_channels,),
        layer.layernorm.weight,
        layer.layernorm.bias,
        layer.layernorm.eps,
    )
    vector_n = layer.vec_layernorm(vector)
    query = F.linear(scalar_n, layer.q_proj.weight, layer.q_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    key = F.linear(scalar_n, layer.k_proj.weight, layer.k_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    value = F.linear(scalar_n, layer.v_proj.weight, layer.v_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    n_edges = idx_i.shape[0]
    filter_key = F.silu(F.linear(edge, layer.dk_proj.weight, layer.dk_proj.bias))
    filter_value = F.silu(F.linear(edge, layer.dv_proj.weight, layer.dv_proj.bias))
    filter_key = filter_key.reshape(n_edges, layer.num_heads, layer.head_dim)
    filter_value = filter_value.reshape(n_edges, layer.num_heads, layer.head_dim)
    projected = F.linear(vector_n, layer.vec_proj.weight, None)
    vec1, vec2, vec3 = torch.split(projected, layer.hidden_channels, dim=-1)
    vector_dot = (vec1 * vec2).sum(dim=1)
    attention = (query[idx_i] * key[idx_j] * filter_key).sum(dim=-1)
    attention = F.silu(attention) * window.reshape(-1, 1)
    scalar_message = (value[idx_j] * filter_value * attention.unsqueeze(-1)).reshape(
        n_edges, layer.hidden_channels
    )
    gates = F.silu(F.linear(scalar_message, layer.s_proj.weight, layer.s_proj.bias))
    gate, coeff = torch.split(gates, layer.hidden_channels, dim=-1)
    vector_message = vector_n[idx_j] * gate.unsqueeze(1)
    vector_message = vector_message + coeff.unsqueeze(1) * direction.unsqueeze(-1)
    aggregated_scalar = scatter_sum(scalar_message, idx_i, n_atoms)
    angular = vector.shape[1]
    aggregated_vector = scatter_sum(vector_message, idx_i, n_atoms)
    mixed = F.linear(aggregated_scalar, layer.o_proj.weight, layer.o_proj.bias)
    if layer.vertex_type == "node":
        angle = _angle(layer, vector_n[idx_i], direction)
        aggregated_angle = scatter_sum(angle, idx_i, n_atoms)
        out_scale, out_dot, out_angle, out_bias = torch.split(
            mixed, layer.hidden_channels, dim=-1
        )
        delta_scalar = vector_dot * out_dot + aggregated_angle * out_angle + out_bias
    else:
        out_scale, out_dot, out_bias = torch.split(mixed, layer.hidden_channels, dim=-1)
        delta_scalar = vector_dot * out_dot + out_bias
    delta_vector = vec3 * out_scale.unsqueeze(1) + aggregated_vector
    delta_edge = None
    if not layer.last_layer:
        dihedral = _pair(
            F.linear(vector_n[idx_i], layer.w_trg_proj.weight, None),
            F.linear(vector_n[idx_j], layer.w_src_proj.weight, None),
            direction,
        )
        filtered = F.silu(F.linear(edge, layer.f_proj.weight, layer.f_proj.bias))
        if layer.vertex_type == "edge":
            angle = _angle(layer, vector_n[idx_i], direction)
            first, second = torch.split(filtered, layer.hidden_channels, dim=-1)
            delta_edge = first * dihedral + second * angle
        else:
            delta_edge = filtered * dihedral
    return delta_scalar, delta_vector, delta_edge


def _pair(left, right, direction):
    return (
        vector_rejection(left, direction) * vector_rejection(right, -direction)
    ).sum(dim=1)


def _angle(layer, receiver, direction):
    left = F.linear(receiver, layer.t_trg_proj.weight, None)
    right = F.linear(receiver, layer.t_src_proj.weight, None)
    return _pair(left, right, direction)


def test_block_matches_the_reference_for_each_vertex():
    torch.manual_seed(2)
    n, features, heads, angular = 5, 8, 2, 8
    idx_i = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    idx_j = torch.tensor([1, 2, 3, 4, 0, 0, 1])
    scalar = torch.randn(n, features)
    vector = torch.randn(n, angular, features)
    edge = torch.randn(idx_i.numel(), features)
    window = torch.rand(idx_i.numel())
    direction = torch.randn(idx_i.numel(), angular)
    for vertex in ("none", "edge", "node"):
        for last in (False, True):
            for vecnorm in ("none", "rms", "max_min"):
                layer = ViSInteraction(
                    features,
                    heads,
                    vertex,
                    vecnorm,
                    trainable_vecnorm=True,
                    last_layer=last,
                )
                layer.eval()
                got = layer(scalar, vector, edge, idx_i, idx_j, window, direction)
                reference = _reference(
                    layer, scalar, vector, edge, idx_i, idx_j, window, direction
                )
                assert torch.allclose(got[0], reference[0], atol=1e-5, rtol=1e-5)
                assert torch.allclose(got[1], reference[1], atol=1e-5, rtol=1e-5)
                if last:
                    assert got[2] is None and reference[2] is None
                    assert layer.f_proj is None
                else:
                    assert torch.allclose(got[2], reference[2], atol=1e-5, rtol=1e-5)
    node = ViSInteraction(features, heads, "node", "none", False, last_layer=True)
    assert node.t_trg_proj is not None and node.w_trg_proj is None
    assert node.o_proj.out_features == 4 * features
    edge_layer = ViSInteraction(features, heads, "edge", "none", False, last_layer=False)
    assert edge_layer.f_proj.out_features == 2 * features
    try:
        ViSInteraction(6, 4, "none", "none", False, False)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected a bad head count to fail")
