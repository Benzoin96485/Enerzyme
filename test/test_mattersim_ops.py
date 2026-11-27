"""Graphormer centrality, segment softmax, and post-softmax cutoff mask."""

from __future__ import annotations

import torch
from numpy.testing import assert_allclose


def test_segment_softmax_is_one_per_receiver():
    from enerzyme.models.mp.ops import segment_softmax

    logits = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    index = torch.tensor([0, 0, 1])
    weights = segment_softmax(logits, index, 3, clamp_min=1e-12)
    assert_allclose(weights[0].numpy(), [0.5, 0.5], atol=1e-6)
    assert_allclose(weights[1].numpy(), [0.5, 0.5], atol=1e-6)
    assert_allclose(weights[2].numpy(), [1.0, 1.0], atol=1e-6)
    assert segment_softmax(logits[:0], index[:0], 0, clamp_min=1e-12).shape == (0, 2)


def test_centrality_matches_the_per_edge_affine_sum():
    from enerzyme.models.mp.basis.envelope import polynomial_envelope
    from enerzyme.models.mattersim.attention import CentralityEncoding

    torch.manual_seed(0)
    encoder = CentralityEncoding(4, 8)
    dist = torch.tensor([1.0, 2.5, 5.0, 6.0])
    basis = torch.randn(4, 4)
    index = torch.tensor([0, 0, 1, 1])
    cutoff = 5.0
    updated = encoder(basis, dist, index, n_atoms=3, cutoff=cutoff)
    mask = polynomial_envelope(dist, cutoff, 3)
    manual = torch.zeros(3, 8)
    for edge in range(4):
        manual[index[edge]] = manual[index[edge]] + encoder.linear(
            basis[edge] * mask[edge]
        )
    assert_allclose(updated.detach().numpy(), manual.detach().numpy(), atol=1e-6)
    assert_allclose(mask[2].detach().numpy(), 0.0, atol=1e-6)
    assert float(mask[3]) == 0.0
    empty = encoder(basis[:0], dist[:0], index[:0], n_atoms=2, cutoff=cutoff)
    assert_allclose(empty.detach().numpy(), torch.zeros(2, 8).numpy(), atol=1e-6)


def test_attention_mask_is_applied_after_softmax():
    from enerzyme.models.mp.ops import scatter_sum
    from enerzyme.models.mattersim.attention import GraphormerBlock
    from enerzyme.models.mp.ops import segment_softmax

    torch.manual_seed(0)
    block = GraphormerBlock(dim=8, num_heads=2, num_basis=4, ffn_dim=8, dropout=0.0)
    block.eval()
    node = torch.randn(3, 8)
    basis = torch.randn(2, 4)
    idx_i = torch.tensor([0, 0])
    idx_j = torch.tensor([1, 2])
    mask = torch.tensor([1.0, 0.0])

    zero = block.attend(node, basis, torch.zeros(2), idx_i, idx_j)
    assert_allclose(zero.detach().numpy(), torch.zeros(3, 8).numpy(), atol=1e-6)

    qkv = block.qkv(node).view(3, 3, 2, 4)
    query, key, value = qkv.unbind(dim=1)
    logits = (query[idx_i] * key[idx_j]).sum(dim=-1) * block.scale
    logits = logits + block.distance_bias(basis)
    weights = segment_softmax(logits, idx_i, 3, clamp_min=1e-12) * mask.unsqueeze(-1)
    message = (weights.unsqueeze(-1) * value[idx_j]).reshape(2, 8)
    expected = block.out(scatter_sum(message, idx_i, 3))
    got = block.attend(node, basis, mask, idx_i, idx_j)
    assert_allclose(got.detach().numpy(), expected.detach().numpy(), atol=1e-6)

    full = block.attend(node, basis, torch.ones(2), idx_i, idx_j)
    assert not torch.allclose(got, full, atol=1e-5, rtol=1e-5)
    # The masked edge still occupies softmax mass, so dropping it from the
    # graph is not the same as multiplying its weight by zero afterwards.
    only = block.attend(
        node, basis[:1], torch.ones(1), idx_i[:1], idx_j[:1]
    )
    assert not torch.allclose(got, only, atol=1e-5, rtol=1e-5)
