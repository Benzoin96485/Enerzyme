"""Real-space Gaussian multipoles: pair kernel, self energy, and l = 1 features."""

from __future__ import annotations

import math

import torch

from enerzyme.models.layers.gto_electrostatics import (
    FIELD_CONSTANT,
    GaussianMultipoleEnergy,
    GaussianPotentialFeatures,
    coulomb_prefactor,
)


def _pair_energy(q1: float, q2: float, distance: float, sigma: float) -> float:
    kernel = math.erf(distance / (2.0 * sigma)) / distance
    return coulomb_prefactor() * q1 * q2 * kernel


def test_pair_energy_matches_erf_kernel_and_limits():
    sigma = 1.5
    energy = GaussianMultipoleEnergy(max_l=0, sigma=sigma, include_self=False).double()
    charges = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    total = energy.atomic_energy(charges, positions, batch).sum()
    assert torch.allclose(
        total, torch.tensor(_pair_energy(1.0, 0.5, 2.0, sigma), dtype=torch.float64), atol=1e-8
    )
    assert abs(coulomb_prefactor() - FIELD_CONSTANT / (4.0 * math.pi)) < 1e-12

    far = positions.clone()
    far[1, 0] = 40.0
    far_total = energy.atomic_energy(charges, far, batch).sum()
    coulomb = coulomb_prefactor() * 1.0 * 0.5 / 40.0
    assert torch.allclose(far_total, torch.tensor(coulomb, dtype=torch.float64), rtol=1e-6, atol=1e-8)

    shifted = positions + torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64)
    assert torch.allclose(
        energy.atomic_energy(charges, shifted, batch).sum(), total, atol=1e-8
    )
    rotation, _ = torch.linalg.qr(torch.tensor(
        [[0.2, 0.9, -0.3], [0.7, -0.1, 0.4], [0.1, 0.2, 0.8]], dtype=torch.float64
    ))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    assert torch.allclose(
        energy.atomic_energy(charges, positions @ rotation, batch).sum(), total, atol=1e-8
    )


def test_separated_graphs_and_self_energy():
    energy = GaussianMultipoleEnergy(max_l=0, sigma=1.5, include_self=True).double()
    charges = torch.tensor([[1.0], [-0.5]], dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64)
    separate = torch.tensor([0, 1], dtype=torch.long)
    alone = energy.atomic_energy(charges[:1], positions[:1], torch.zeros(1, dtype=torch.long))
    both = energy.atomic_energy(charges, positions, separate)
    assert torch.allclose(both[0], alone[0], atol=1e-8)
    assert torch.isfinite(both).all()
    assert both[0] > 0
    no_self = GaussianMultipoleEnergy(max_l=0, sigma=1.5, include_self=False).double()
    assert torch.allclose(
        no_self.atomic_energy(charges[:1], positions[:1], torch.zeros(1, dtype=torch.long)),
        torch.zeros(1, dtype=torch.float64),
    )


def test_dipole_pair_forces_match_finite_difference():
    energy = GaussianMultipoleEnergy(
        max_l=1, sigma=1.2, include_self=True, energy_scale=1.0
    ).double()
    multipoles = torch.tensor(
        [[0.4, 0.1, -0.2, 0.05], [-0.3, 0.0, 0.15, -0.1]],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.7, -0.4, 0.3]], dtype=torch.float64, requires_grad=True
    )
    batch = torch.zeros(2, dtype=torch.long)
    total = energy.atomic_energy(multipoles, positions, batch).sum()
    analytical = torch.autograd.grad(total, positions)[0]
    step = 1e-5
    numerical = torch.zeros_like(positions)
    for atom in range(2):
        for coord in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = energy.atomic_energy(multipoles, plus, batch).sum()
            e_minus = energy.atomic_energy(multipoles, minus, batch).sum()
            numerical[atom, coord] = (e_plus - e_minus) / (2 * step)
    assert torch.allclose(analytical, numerical, rtol=1e-5, atol=1e-6)


def test_dipole_rotation_error_shrinks_with_offset():
    """The l = 1 stencil is a laboratory-frame finite difference.

    Rotating the dipole in e3nn order (q, y, z, x) and shrinking the offset
    drives the pair energy back to the rotated geometry.
    """
    multipoles = torch.tensor(
        [[0.2, 0.1, -0.05, 0.08], [-0.2, -0.04, 0.03, 0.01]],
        dtype=torch.float64,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.6, 0.3, -0.2]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    cartesian = multipoles[:, 1:4][:, [2, 0, 1]] @ rotation
    rotated_multipoles = torch.cat(
        [multipoles[:, :1], cartesian[:, [1, 2, 0]]], dim=-1
    )
    rotated_positions = positions @ rotation
    errors = []
    for offset in (0.2, 0.02, 0.002):
        energy = GaussianMultipoleEnergy(
            max_l=1, sigma=1.5, include_self=True, offset=offset
        ).double()
        reference = energy.atomic_energy(multipoles, positions, batch).sum()
        turned = energy.atomic_energy(rotated_multipoles, rotated_positions, batch).sum()
        errors.append(float((reference - turned).abs()))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1.0e-4


def test_repulsion_pushes_positive_charges_apart():
    energy = GaussianMultipoleEnergy(max_l=0, sigma=1.5, include_self=False).double()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    total = energy.atomic_energy(
        torch.ones(2, 1, dtype=torch.float64), positions, torch.zeros(2, dtype=torch.long)
    ).sum()
    force = -torch.autograd.grad(total, positions)[0]
    assert force[0, 0] < 0
    assert force[1, 0] > 0
    assert torch.allclose(force[0, 1:], torch.zeros(2, dtype=torch.float64), atol=1e-8)


def test_potential_feature_shape_axis_and_empty_pair():
    features = GaussianPotentialFeatures(
        density_max_l=1,
        density_sigma=1.5,
        projection_max_l=1,
        projection_sigmas=(1.5, 2.0),
        include_self=False,
    ).double()
    lone = features(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, dtype=torch.long),
    )
    assert lone.shape == (1, 8)
    assert torch.allclose(lone, torch.zeros_like(lone), atol=1e-8)

    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    multipoles = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    projected = features(multipoles, positions, torch.zeros(2, dtype=torch.long))
    assert projected.shape == (2, 8)
    at_origin = projected[0]
    # Layout is n x 0e + n x 1o: [σ0, σ1, y0, z0, x0, y1, z1, x1].
    # A finite offset makes the perpendicular channels small but not zero.
    assert at_origin[0] > 0
    assert at_origin[1] > 0
    assert torch.allclose(at_origin[2], at_origin[3], atol=1e-8)
    assert torch.allclose(at_origin[5], at_origin[6], atol=1e-8)
    assert at_origin[4] > at_origin[2].abs() * 5
    assert at_origin[7] > at_origin[5].abs() * 5
    try:
        GaussianPotentialFeatures(density_max_l=2)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("l > 1 should be rejected")
