"""CHGNet basis, undirected bonds, bond graph, and gated MLP."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose


def test_fourier_matches_the_reference_order():
    from enerzyme.models.chgnet.basis import FourierExpansion

    expansion = FourierExpansion(order=2, learnable=False)
    angle = torch.tensor([0.0, math.pi / 2])
    values = expansion(angle)
    assert values.shape == (2, 5)
    scale = 1.0 / math.sqrt(math.pi)
    assert_allclose(values[0, 0].item(), scale / math.sqrt(2.0), atol=1e-6)
    assert_allclose(values[0, 1:3].detach().numpy(), [0.0, 0.0], atol=1e-6)
    assert_allclose(values[0, 3:].detach().numpy(), [scale, scale], atol=1e-6)
    assert_allclose(values[1, 1].item(), scale, atol=1e-6)
    assert_allclose(values[1, 2].item(), 0.0, atol=1e-6)
    assert_allclose(values[1, 3].item(), 0.0, atol=1e-6)
    assert_allclose(values[1, 4].item(), -scale, atol=1e-6)
    assert expansion(torch.zeros(0)).shape == (0, 5)

    learned = FourierExpansion(order=2, learnable=True)
    theta = torch.tensor([0.4, 1.1], requires_grad=True)
    learned(theta).sum().backward()
    assert learned.frequencies.grad is not None
    assert theta.grad is not None


def test_gemnet_bessel_is_the_chgnet_smooth_radial_basis():
    from enerzyme.models.mp.basis.envelope import polynomial_envelope
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 5.0
    exponent = 5
    layer = BesselRBFLayer(
        num_rbf=3,
        cutoff_sr=cutoff,
        flavor="gemnet",
        trainable=False,
        envelope_exponent=exponent,
    )
    dist = torch.tensor([0.5, 1.7, 4.9])
    got = layer._get_rbf(dist)
    scale = math.sqrt(2.0 / cutoff)
    env = polynomial_envelope(dist, cutoff, exponent).unsqueeze(-1)
    scaled = dist / cutoff
    frequencies = math.pi * torch.arange(1, 4)
    expected = env * scale * torch.sin(scaled.unsqueeze(-1) * frequencies) / dist.unsqueeze(-1)
    assert_allclose(got.detach().numpy(), expected.detach().numpy(), atol=1e-6)
    assert_allclose(layer._get_rbf(torch.tensor([cutoff])).detach().numpy(), 0.0, atol=1e-6)


def test_undirected_pairs_include_distinct_periodic_images():
    from enerzyme.models.chgnet.geometry import bond_graph, undirected_bond_index

    idx_i = torch.tensor([0, 1, 0, 1])
    idx_j = torch.tensor([1, 0, 1, 0])
    vij = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, -1.2, 0.0],
        ]
    )
    bonds = undirected_bond_index(idx_i, idx_j, vij)
    assert bonds.representative.numel() == 2
    assert int(bonds.directed_to_undirected.max()) == 1
    assert bonds.directed_to_undirected[0] == bonds.directed_to_undirected[1]
    assert bonds.directed_to_undirected[2] == bonds.directed_to_undirected[3]
    assert bonds.directed_to_undirected[0] != bonds.directed_to_undirected[2]

    dist = vij.norm(dim=-1)
    angles = bond_graph(idx_i, dist, n_atoms=2, bond_cutoff=2.0)
    assert angles.idx_ij.numel() == 4
    tight = bond_graph(idx_i, dist, n_atoms=2, bond_cutoff=1.1)
    assert tight.idx_ij.numel() == 0


def test_bond_graph_on_a_right_angle():
    from enerzyme.models.chgnet.geometry import bond_graph, directed_bond_angle

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    idx_i, idx_j = [], []
    for i in range(3):
        for j in range(3):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    wide = bond_graph(idx_i, dist, n_atoms=3, bond_cutoff=2.0)
    assert wide.idx_ij.numel() == 6
    narrow = bond_graph(idx_i, dist, n_atoms=3, bond_cutoff=1.2)
    assert narrow.idx_ij.numel() == 2
    theta = directed_bond_angle(vij, narrow.idx_ij, narrow.idx_ik)
    assert_allclose(theta.detach().numpy(), [math.pi / 2, math.pi / 2], atol=1e-5)


def test_closest_reverse_wins_when_several_images_match():
    from enerzyme.models.chgnet.geometry import undirected_bond_index

    idx_i = torch.tensor([0, 1, 1, 0])
    idx_j = torch.tensor([1, 0, 0, 1])
    vij = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 5.0e-5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -5.0e-5],
        ]
    )
    bonds = undirected_bond_index(idx_i, idx_j, vij)
    assert bonds.directed_to_undirected[0] == bonds.directed_to_undirected[2]
    assert bonds.directed_to_undirected[1] == bonds.directed_to_undirected[3]
    assert bonds.directed_to_undirected[0] != bonds.directed_to_undirected[1]
    assert int(bonds.representative[int(bonds.directed_to_undirected[0])]) == 0
    assert int(bonds.representative[int(bonds.directed_to_undirected[1])]) == 1


def test_missing_reverse_edge_is_rejected():
    from enerzyme.models.chgnet.geometry import undirected_bond_index

    try:
        undirected_bond_index(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
    except ValueError as exc:
        assert "reverse" in str(exc)
    else:
        raise AssertionError("a one-way neighbor list must be rejected")


def test_layer_norm_sits_before_the_final_activation():
    from enerzyme.models.mp.nn.mlp import GatedMLP

    torch.manual_seed(0)
    layer = GatedMLP(4, [5], 3, activate_last=True, use_bias=True, norm="layer")
    x = torch.randn(6, 4)
    value = torch.nn.functional.silu(layer.value_net[0](x))
    value = layer.value_norm(layer.value_net[2](value))
    value = torch.nn.functional.silu(value)
    gate = torch.nn.functional.silu(layer.gate_net[0](x))
    gate = torch.sigmoid(layer.gate_norm(layer.gate_net[2](gate)))
    assert_allclose(layer(x).detach().numpy(), (value * gate).detach().numpy(), atol=1e-6)

    bare = GatedMLP(4, [], 3, activate_last=True, use_bias=False)
    assert torch.count_nonzero(bare(torch.zeros(2, 4))) == 0
    assert isinstance(bare.value_net[-1], torch.nn.SiLU)
    assert isinstance(bare.gate_net[-1], torch.nn.Sigmoid)
