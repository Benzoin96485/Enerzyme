"""ET attention and neighbor embedding versus the TorchMD-Net formulas."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from enerzyme.models.et.interaction import (
    ETNeighborEmbedding,
    EquivariantMultiHeadAttention,
    normalize_distance_influence,
)
from enerzyme.models.functional import reduce_edge
from enerzyme.models.layers.self_pairs import insert_missing_self_pairs, unit_direction
from enerzyme.models.tensornet.ops import unit_vectors


def test_distance_influence_names():
    assert normalize_distance_influence("Both") == "both"
    try:
        normalize_distance_influence("softmax")
    except ValueError as exc:
        assert "distance_influence" in str(exc)
    else:
        raise AssertionError("expected an unknown distance influence to fail")


def test_unit_direction_matches_tensornet_and_zeros_contact():
    vij = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    dist = torch.tensor([3.0, 0.0, 2.0])
    got = unit_direction(vij, dist)
    assert torch.allclose(got, unit_vectors(vij, dist))
    assert torch.count_nonzero(got[1]) == 0
    assert torch.allclose(got[0], torch.tensor([1.0, 0.0, 0.0]))


def test_neighbor_embedding_drops_self_and_matches_formula():
    torch.manual_seed(0)
    n, features, num_rbf = 4, 6, 3
    block = ETNeighborEmbedding(features, num_rbf, max_Za=10)
    block.eval()
    atomic_numbers = torch.tensor([1, 6, 8, 1])
    scalar = torch.randn(n, features)
    idx_i = torch.tensor([0, 1, 2, 0, 1])
    idx_j = torch.tensor([1, 0, 2, 2, 1])
    rbf = torch.randn(5, num_rbf)
    window = torch.tensor([0.8, 0.8, 1.0, 0.4, 0.3])
    got = block(atomic_numbers, scalar, idx_i, idx_j, rbf, window)

    keep = idx_i != idx_j
    species = block.species.get_embedding(atomic_numbers)
    weight = F.linear(rbf[keep], block.distance_proj.weight, block.distance_proj.bias)
    weight = weight * window[keep].unsqueeze(-1)
    message = weight * species[idx_j[keep]]
    aggregated = reduce_edge(message, idx_i[keep], (n, features))
    mixed = torch.cat((scalar, aggregated), dim=-1)
    reference = F.linear(mixed, block.combine.weight, block.combine.bias)
    assert torch.allclose(got, reference, atol=1e-6, rtol=1e-6)
    # Atom 3 is not a receiver of any non-self edge, and atom 1's self pair is dropped.
    assert torch.count_nonzero(aggregated[3]) == 0


def _attention_reference(layer, scalar, vector, idx_i, idx_j, rbf, window, direction):
    n_atoms = scalar.shape[0]
    normed = F.layer_norm(
        scalar, (layer.hidden_channels,), layer.layernorm.weight, layer.layernorm.bias, layer.layernorm.eps
    )
    query = F.linear(normed, layer.q_proj.weight, layer.q_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    key = F.linear(normed, layer.k_proj.weight, layer.k_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim
    )
    value = F.linear(normed, layer.v_proj.weight, layer.v_proj.bias).reshape(
        n_atoms, layer.num_heads, layer.head_dim * 3
    )
    projected = F.linear(vector, layer.vec_proj.weight, None)
    vec1, vec2, vec3 = torch.split(projected, layer.hidden_channels, dim=-1)
    vector_dot = (vec1 * vec2).sum(dim=1)
    heads = vector.reshape(n_atoms, 3, layer.num_heads, layer.head_dim)
    query_i = query[idx_i]
    key_j = key[idx_j]
    value_j = value[idx_j]
    if layer.dk_proj is None:
        attention = (query_i * key_j).sum(dim=-1)
    else:
        filt = F.silu(F.linear(rbf, layer.dk_proj.weight, layer.dk_proj.bias))
        filt = filt.reshape(-1, layer.num_heads, layer.head_dim)
        attention = (query_i * key_j * filt).sum(dim=-1)
    attention = F.silu(attention)
    cutoff = window.reshape(-1, 1)
    if layer.vector_cutoff:
        value_j = value_j * cutoff.unsqueeze(-1)
    else:
        attention = attention * cutoff
    if layer.dv_proj is not None:
        filt = F.silu(F.linear(rbf, layer.dv_proj.weight, layer.dv_proj.bias))
        value_j = value_j * filt.reshape(-1, layer.num_heads, layer.head_dim * 3)
    scalar_message, gate, coeff = torch.split(value_j, layer.head_dim, dim=2)
    scalar_message = scalar_message * attention.unsqueeze(-1)
    vector_message = heads[idx_j] * gate.unsqueeze(1) + coeff.unsqueeze(1) * direction.reshape(
        -1, 3, 1, 1
    )
    aggregated_scalar = reduce_edge(
        scalar_message, idx_i, (n_atoms, layer.num_heads, layer.head_dim)
    ).reshape(n_atoms, layer.hidden_channels)
    aggregated_vector = reduce_edge(
        vector_message, idx_i, (n_atoms, 3, layer.num_heads, layer.head_dim)
    ).reshape(n_atoms, 3, layer.hidden_channels)
    mixed = F.linear(aggregated_scalar, layer.o_proj.weight, layer.o_proj.bias)
    out_scalar, out_gate, out_bias = torch.split(mixed, layer.hidden_channels, dim=-1)
    return vector_dot * out_gate + out_bias, vec3 * out_scalar.unsqueeze(1) + aggregated_vector


def test_attention_matches_torchmd_formula_for_each_filter():
    torch.manual_seed(1)
    n, features, heads, num_rbf = 5, 8, 2, 4
    idx_i = torch.tensor([0, 1, 2, 3, 4, 0])
    idx_j = torch.tensor([1, 2, 3, 4, 0, 0])
    scalar = torch.randn(n, features)
    vector = torch.randn(n, 3, features)
    rbf = torch.randn(idx_i.numel(), num_rbf)
    window = torch.rand(idx_i.numel())
    direction = torch.nn.functional.normalize(torch.randn(idx_i.numel(), 3), dim=-1)
    direction[-1] = 0
    for influence in ("both", "keys", "values", "none"):
        for vector_cutoff in (False, True):
            layer = EquivariantMultiHeadAttention(
                features, num_rbf, heads, influence, vector_cutoff
            )
            layer.eval()
            got = layer(scalar, vector, idx_i, idx_j, rbf, window, direction)
            reference = _attention_reference(
                layer, scalar, vector, idx_i, idx_j, rbf, window, direction
            )
            assert torch.allclose(got[0], reference[0], atol=1e-6, rtol=1e-6)
            assert torch.allclose(got[1], reference[1], atol=1e-6, rtol=1e-6)
            if influence == "both" and vector_cutoff:
                assert layer.dk_proj is not None and layer.dv_proj is not None
            if influence == "none":
                assert layer.dk_proj is None and layer.dv_proj is None


def test_vector_cutoff_changes_the_value_and_not_only_the_score():
    torch.manual_seed(2)
    features, heads, num_rbf = 8, 2, 3
    paper = EquivariantMultiHeadAttention(features, num_rbf, heads, "both", False)
    smooth = EquivariantMultiHeadAttention(features, num_rbf, heads, "both", True)
    smooth.load_state_dict(paper.state_dict())
    scalar = torch.randn(3, features)
    vector = torch.randn(3, 3, features)
    idx_i = torch.tensor([0, 1, 2])
    idx_j = torch.tensor([1, 2, 0])
    kwargs = dict(
        scalar=scalar,
        vector=vector,
        idx_i=idx_i,
        idx_j=idx_j,
        rbf=torch.randn(3, num_rbf),
        window=torch.tensor([0.2, 0.5, 0.9]),
        direction=F.normalize(torch.randn(3, 3), dim=-1),
    )
    left = paper(**kwargs)
    right = smooth(**kwargs)
    # One layer's scalar residual matches: the cutoff commutes through the
    # scalar value. The vector residual does not, which is the discontinuity
    # the flag removes.
    assert torch.allclose(left[0], right[0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(left[1], right[1], atol=1e-5, rtol=1e-5)


def test_insert_self_pairs_uses_the_radial_basis_once():
    class _Radial:
        def __init__(self):
            self.calls = 0

        def get_rbf(self, dist):
            self.calls += 1
            return torch.stack((dist, dist + 1), dim=-1)

    radial = _Radial()
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    vij = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    dist = torch.ones(2)
    rbf = torch.ones(2, 2)
    window = torch.full((2,), 0.5)
    out = insert_missing_self_pairs(
        3, idx_i, idx_j, vij, dist, rbf, window,
        enabled=True, radial_basis=radial, num_rbf=2,
    )
    assert radial.calls == 1
    assert out[0].tolist()[-1] == 2
    assert out[1].tolist()[-1] == 2
    assert torch.count_nonzero(out[2][-1]) == 0
    assert out[3][-1] == 0
    assert torch.allclose(out[4][-1], torch.tensor([0.0, 1.0]))
    assert out[5][-1] == 1

    try:
        insert_missing_self_pairs(
            1, idx_i[:0], idx_j[:0], vij[:0], dist[:0], rbf[:0], window[:0],
            enabled=True, radial_basis=None, num_rbf=2,
        )
    except RuntimeError as exc:
        assert "get_rbf" in str(exc)
    else:
        raise AssertionError("expected a missing radial basis to fail")
