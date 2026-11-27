"""M3GNet basis, triplets, and gated MLP."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose


def test_l0_matches_sinc_bessel_and_vanishes_at_cutoff():
    from enerzyme.models.m3gnet.basis import SphericalBesselRBFLayer

    cutoff = 5.0
    layer = SphericalBesselRBFLayer(num_rbf=3, cutoff_sr=cutoff, max_l=3)
    dist = torch.tensor([0.0, 1.0, 2.5, cutoff])
    basis = layer._get_rbf(dist)
    assert basis.shape == (4, 9)
    assert torch.isfinite(basis).all()
    assert_allclose(basis[3].detach().numpy(), torch.zeros(9).numpy(), atol=1e-5)

    # l = 0, n = 1..3: sqrt(2/c) sin(n π r / c) / r, with the r→0 limit nπ/c.
    scale = math.sqrt(2.0 / cutoff)
    for index, order in enumerate((1, 2, 3)):
        arg = order * math.pi * dist / cutoff
        expected = torch.where(
            dist == 0,
            torch.full_like(dist, scale * order * math.pi / cutoff),
            scale * torch.sin(arg) / dist.clamp(min=1e-12),
        )
        assert_allclose(
            basis[:, index].detach().numpy(),
            expected.detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )


def test_joint_basis_uses_the_other_bond():
    from enerzyme.models.m3gnet.basis import (
        SphericalBesselRBFLayer,
        joint_zonal_basis,
    )

    layer = SphericalBesselRBFLayer(num_rbf=2, cutoff_sr=5.0, max_l=2)
    dist = torch.tensor([1.0, 1.7])
    rbf = layer._get_rbf(dist)
    angle = torch.tensor([0.8])
    base = joint_zonal_basis(rbf, angle, torch.tensor([1]), max_l=2, num_radial=2)
    swapped = rbf.clone()
    swapped[0] = rbf[1]
    swapped[1] = rbf[0]
    changed = joint_zonal_basis(
        swapped, angle, torch.tensor([1]), max_l=2, num_radial=2
    )
    assert not torch.allclose(base, changed)
    only_ij = rbf.clone()
    only_ij[0] = torch.randn_like(only_ij[0])
    # idx_ik = 1, so row 0 is the receiving bond and must not enter the basis.
    same = joint_zonal_basis(only_ij, angle, torch.tensor([1]), max_l=2, num_radial=2)
    assert_allclose(base.detach().numpy(), same.detach().numpy(), atol=1e-6)
    empty = joint_zonal_basis(
        rbf, angle.new_zeros(0), torch.zeros(0, dtype=torch.long), 2, 2
    )
    assert empty.shape == (0, 4)


def test_receiver_triplets_count_images_and_cutoff():
    from enerzyme.models.m3gnet.geometry import bond_angle, receiver_triplets

    idx_i = torch.tensor([0, 0, 1, 1, 2, 2])
    idx_j = torch.tensor([1, 2, 0, 2, 0, 1])
    dist = torch.ones(6)
    pairs = receiver_triplets(idx_i, dist, num_nodes=3, threebody_cutoff=4.0)
    assert pairs.idx_ij.numel() == 6
    assert torch.all(idx_i[pairs.idx_ij] == idx_i[pairs.idx_ik])
    assert torch.all(pairs.idx_ij != pairs.idx_ik)

    images_i = torch.tensor([0, 0])
    images = receiver_triplets(
        images_i, torch.tensor([1.0, 1.4]), num_nodes=2, threebody_cutoff=4.0
    )
    assert images.idx_ij.tolist() == [0, 1]
    assert images.idx_ik.tolist() == [1, 0]

    far = dist.clone()
    far[1] = 4.5
    far[5] = 4.5
    pruned = receiver_triplets(idx_i, far, num_nodes=3, threebody_cutoff=4.0)
    assert 1 not in pruned.idx_ij.tolist()
    assert 1 not in pruned.idx_ik.tolist()
    assert pruned.idx_ij.numel() == 2

    none = receiver_triplets(idx_i, dist, num_nodes=3, threebody_cutoff=0.5)
    assert none.idx_ij.numel() == 0

    vij = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    angle = bond_angle(vij, torch.tensor([0]), torch.tensor([1]))
    assert_allclose(angle.detach().numpy(), [math.pi / 2], atol=1e-5)
    assert bond_angle(vij, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)).numel() == 0


def test_gated_mlp_branches_and_zero_bias_free():
    from enerzyme.models.mp.nn.mlp import GatedMLP

    torch.manual_seed(0)
    layer = GatedMLP(4, [5], 3, activate_last=False, use_bias=True)
    x = torch.randn(2, 4)
    value = layer.value_net[0](x)
    value = torch.nn.functional.silu(value)
    value = layer.value_net[2](value)
    gate = torch.sigmoid(layer.gate_net[2](torch.nn.functional.silu(layer.gate_net[0](x))))
    assert_allclose(layer(x).detach().numpy(), (value * gate).detach().numpy(), atol=1e-6)

    bias_free = GatedMLP(4, [], 3, activate_last=True, use_bias=False)
    zeros = bias_free(torch.zeros(2, 4))
    assert torch.count_nonzero(zeros) == 0


def test_envelope_rejects_apply_cutoff():
    from enerzyme.models.m3gnet.basis import SphericalBesselRBFLayer

    try:
        SphericalBesselRBFLayer(num_rbf=2, cutoff_sr=5.0, max_l=2, apply_cutoff_fn=True)
    except ValueError as exc:
        assert "envelope" in str(exc)
    else:
        raise AssertionError("apply_cutoff_fn must be rejected")
