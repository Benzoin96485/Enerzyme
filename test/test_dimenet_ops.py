"""DimeNet operator tests: envelope, Bessel flavor, SBF, triplets, bilinear."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose

from enerzyme.models.dimenet.basis import (
    SphericalFourierBesselBasis,
    real_sph_harm_l0,
    spherical_bessel_zeros,
    spherical_jn,
)
from enerzyme.models.dimenet.triplets import directed_triplets, triplet_angles
from enerzyme.models.so3.envelope import DimeNetEnvelope


def test_spherical_bessel_j0_zeros_are_n_pi():
    zeros = spherical_bessel_zeros(1, 6)
    expected = math.pi * torch.arange(1, 7).double().numpy()
    assert_allclose(zeros[0], expected, rtol=0, atol=1e-12)


def test_dimenet_envelope_matches_closed_form():
    env = DimeNetEnvelope(exponent=5)
    x = torch.tensor([0.1, 0.5, 0.9, 1.0, 1.2])
    p = 6.0
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    x_safe = x.clamp(min=1e-12)
    closed = (1.0 / x_safe + a * x_safe.pow(p - 1) + b * x_safe.pow(p) + c * x_safe.pow(p + 1)) * (
        x < 1.0
    ).to(x.dtype)
    got = env(x)
    assert torch.allclose(got, closed, atol=1e-6)
    assert got[-2].abs().item() == 0.0
    assert got[-1].item() == 0.0


def test_bessel_rbf_dimenet_flavor_matches_envelope_sin():
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 5.0
    num_rbf = 4
    layer = BesselRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        flavor="dimenet",
        trainable=False,
        envelope_exponent=5,
    )
    dist = torch.tensor([0.5, 1.5, 4.9, 5.1])
    rbf = layer.get_rbf(dist)
    assert rbf.shape == (4, num_rbf)
    env = DimeNetEnvelope(5)
    x = dist / cutoff
    freq = math.pi * torch.arange(1, num_rbf + 1, dtype=dist.dtype)
    expected = env(x).unsqueeze(-1) * torch.sin(x.unsqueeze(-1) * freq)
    assert torch.allclose(rbf, expected, atol=1e-6)
    assert torch.all(rbf[-1].abs() == 0)


def test_bessel_rbf_default_flavor_unchanged_sin_over_r():
    from enerzyme.models.layers.rbf import BesselRBFLayer

    layer = BesselRBFLayer(num_rbf=3, cutoff_sr=4.0, trainable=False, apply_cutoff_fn=False)
    dist = torch.tensor([1.0, 2.0])
    rbf = layer._get_rbf(dist)
    pref = math.sqrt(2.0 / 4.0)
    n = torch.arange(1, 4, dtype=dist.dtype)
    expected = pref * torch.sin(n * math.pi / 4.0 * dist.unsqueeze(-1)) / dist.unsqueeze(-1)
    assert torch.allclose(rbf, expected, atol=1e-6)


def test_sbf_l0_matches_normalized_j0_times_y00():
    cutoff = 5.0
    sbf = SphericalFourierBesselBasis(
        num_spherical=1, num_radial=2, cutoff=cutoff, envelope_exponent=5
    )
    dist = torch.tensor([1.0, 2.0, 3.0])
    angle = torch.zeros(2)
    idx_kj = torch.tensor([0, 1])
    out = sbf(dist, angle, idx_kj)
    assert out.shape == (2, 2)
    y00 = real_sph_harm_l0(0, angle)
    env = DimeNetEnvelope(5)(dist / cutoff)
    zeros = sbf.zeros.to(dist.dtype)
    norm = sbf.normalizer.to(dist.dtype)
    col0 = env * norm[0, 0] * spherical_jn(0, zeros[0, 0] * dist / cutoff)
    expected = (col0[idx_kj] * y00).unsqueeze(-1)
    assert torch.allclose(out[:, :1], expected, atol=1e-5)


def test_directed_triplets_complete_graph_count():
    n = 4
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i, dtype=torch.long)
    idx_j = torch.tensor(idx_j, dtype=torch.long)
    idx_kj, idx_ji, trip_i, trip_j, trip_k = directed_triplets(idx_i, idx_j, n)
    assert idx_kj.numel() == n * (n - 1) * (n - 2)
    assert torch.all(trip_k != trip_i)
    assert torch.all(trip_j != trip_i)
    assert torch.all(idx_i[idx_ji] == trip_i)
    assert torch.all(idx_j[idx_kj] == trip_k)


def test_triplet_angle_right_angle():
    # i at (1,0,0), j at origin, k at (0,1,0) → 90° at j
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # edges: 0←1 (j=1,i=0), 1←2 (j=2,i=1) wait
    # receiver i, sender j; vij = R_j - R_i
    # edge ji: i=0, j=1 → vij = R1 - R0 = (-1, 0, 0)
    # edge kj: i=1, j=2 → vij = R2 - R1 = (0, 1, 0)
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 2])
    vij = pos[idx_j] - pos[idx_i]
    idx_kj = torch.tensor([1])
    idx_ji = torch.tensor([0])
    ang = triplet_angles(vij, idx_kj, idx_ji)
    assert torch.allclose(ang, torch.tensor([math.pi / 2]), atol=1e-5)


def test_interaction_bilinear_finite():
    from enerzyme.models.dimenet.interaction import DimeNetInteractionBlock

    torch.manual_seed(0)
    block = DimeNetInteractionBlock(
        dim_embedding=8,
        num_bilinear=4,
        num_sbf=6,
        num_rbf=3,
        num_before_skip=1,
        num_after_skip=1,
    )
    e = 5
    t = 3
    x = torch.randn(e, 8)
    rbf = torch.randn(e, 3)
    sbf = torch.randn(t, 6)
    idx_kj = torch.tensor([0, 1, 2])
    idx_ji = torch.tensor([1, 1, 4])
    out = block(x, rbf, sbf, idx_kj, idx_ji)
    assert out.shape == (e, 8)
    assert torch.isfinite(out).all()
