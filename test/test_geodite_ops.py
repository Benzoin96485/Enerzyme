"""Geodite radial window, screened ZBL, and polynomial spherical harmonics."""

from __future__ import annotations

import math

import torch

from enerzyme.models.mp.basis.envelope import polynomial_envelope


def test_polynomial_envelope_is_flat_at_the_cutoff_and_accepts_per_edge_radii():
    distance = torch.tensor([0.0, 1.0, 5.999, 6.0], requires_grad=True)
    window = polynomial_envelope(distance, 6.0, 6)
    assert window[0].item() == 1.0
    assert window[-1].item() == 0.0
    assert window[-2].item() < 1e-6
    window.sum().backward()
    assert distance.grad[-2].abs().item() < 1e-3

    radii = torch.tensor([0.5, 1.5, 2.0])
    opened = polynomial_envelope(torch.tensor([0.25, 1.6, 1.0]), radii, 5)
    assert opened[0].item() > 0.0
    assert opened[1].item() == 0.0
    assert opened[2].item() > 0.0


def test_covalent_weight_is_bounded_and_suppresses_contact():
    from enerzyme.models.layers.covalent import (
        constrained_attenuation_exponents,
        covalent_distance_weight,
    )
    from enerzyme.models.layers.radial_window import RadialWindowLayer

    window = RadialWindowLayer(
        cutoff_sr=6.0, max_Za=10, attenuation="covalent", envelope="polynomial"
    )
    p, q = window.exponents()
    assert float(p.detach()) > 1.0
    assert float(q.detach()) > 1.0
    raw_p = torch.tensor(-15.2013)
    raw_q = torch.tensor(3.2139)
    p2, q2 = constrained_attenuation_exponents(raw_p, raw_q)
    assert torch.allclose(p, p2)
    assert torch.allclose(q, q2)

    radius = window.covalent_radii[1] * 2
    distance = torch.tensor([1e-3, 0.2, 1.0, 20.0])
    weight = covalent_distance_weight(distance, radius.expand_as(distance), p, q)
    assert torch.all(weight > 0)
    assert torch.all(weight < 1)
    assert weight[0] < weight[1] < weight[2] < weight[3]
    assert weight[0].item() < 0.05
    # With the published initialization p is almost 1, so y(r) reaches 1 only
    # at distances much larger than a typical neighbor cutoff.
    far = covalent_distance_weight(torch.tensor([1.0e4]), radius.view(1), p, q)
    assert far.item() > 0.99


def test_radial_window_vanishes_at_the_cutoff():
    from enerzyme.models.layers.radial_window import RadialWindowLayer

    window = RadialWindowLayer(
        cutoff_sr=6.0,
        max_Za=10,
        envelope="polynomial",
        envelope_p=6,
        attenuation="covalent",
    )
    basis = torch.ones(3, 4)
    distance = torch.tensor([1.0, 5.999, 6.0])
    atomic_numbers = torch.tensor([1, 1])
    receivers = torch.tensor([0, 0, 1])
    senders = torch.tensor([1, 1, 0])
    filtered = window.get_rbf(
        basis, distance, atomic_numbers, receivers, senders
    )
    assert filtered.shape == (3, 4)
    assert filtered[0].abs().min() > 0
    assert filtered[1].abs().max() < 1e-5
    assert torch.count_nonzero(filtered[2]) == 0


def test_screened_zbl_matches_the_pair_formula_and_the_covalent_cutoff():
    from enerzyme.models.layers.screened_zbl import ScreenedZBLRepulsionEnergyLayer

    layer = ScreenedZBLRepulsionEnergyLayer(
        max_Za=10,
        Bohr_in_R=0.5291772105638411,
        Hartree_in_E=27.211386245988,
        envelope_p=5,
    )
    atomic_numbers = torch.tensor([1, 1])
    receivers = torch.tensor([0, 1])
    senders = torch.tensor([1, 0])

    def expected(distance: float) -> torch.Tensor:
        z = torch.tensor(1.0)
        length = layer.a_factor.detach() * layer.bohr / (2 * z ** layer.z_power.detach())
        reduced = torch.tensor(distance) / length
        phi = (
            layer.screen_coefs.detach()
            * torch.exp(-layer.screen_exponents.detach() * reduced)
        ).sum()
        radius_sum = layer.covalent_radii[1] * 2
        env = polynomial_envelope(
            torch.tensor(distance), radius_sum.to(dtype=torch.float32), layer.envelope_p
        )
        return layer.bohr * layer.hartree * phi * env / distance

    short = torch.tensor([0.3, 0.3])
    energy = layer.get_E_zbl_a(atomic_numbers, short, receivers, senders)
    assert energy.shape == (2,)
    assert torch.allclose(energy.sum(), expected(0.3), rtol=1e-5, atol=1e-5)
    assert energy.sum() > 0

    far = torch.tensor([3.0, 3.0])
    silent = layer.get_E_zbl_a(atomic_numbers, far, receivers, senders)
    assert torch.count_nonzero(silent) == 0


def test_geodite_harmonics_match_the_polynomial_basis():
    from enerzyme.models.geodite.geometry import angular_dim, geodite_spherical_harmonics
    from enerzyme.models.geodite.interaction import vector_rejection

    axis = torch.tensor([[0.0, 0.0, 1.0]])
    harmonics = geodite_spherical_harmonics(axis, 4)
    assert harmonics.shape[-1] == angular_dim(4) == 24
    assert torch.allclose(harmonics[0, :3], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(harmonics[0, 3:8], torch.tensor([
        0.0, 0.0, -0.5, 0.0, math.sqrt(3.0) / 2.0
    ]))
    assert torch.allclose(
        harmonics[0, 12], torch.tensor(-math.sqrt(42.0) / 4.0), atol=1e-6
    )
    assert torch.allclose(harmonics[0, 19], torch.tensor(1.125), atol=1e-6)

    vector = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.7]]), dim=-1)
    rotated = vector @ torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(
        geodite_spherical_harmonics(rotated, 1)[0],
        rotated[0],
    )

    direction = geodite_spherical_harmonics(vector, 1)
    parallel = direction.unsqueeze(-1).expand(-1, -1, 5).contiguous()
    assert torch.allclose(
        vector_rejection(parallel, direction), torch.zeros_like(parallel), atol=1e-6
    )


def test_zero_radial_basis_matches_the_isolated_atom():
    from enerzyme.models.presets.cartesian_vector import geodite

    torch.manual_seed(0)
    core = geodite(
        dim_embedding=16,
        num_rbf=4,
        max_Za=10,
        num_layers=2,
        lmax=2,
        num_heads=4,
        attn_dropout=0.2,
    ).eval()
    embedding = torch.randn(3, 16)
    atomic_numbers = torch.tensor([6, 1, 8])
    receivers = torch.tensor([0, 1, 1, 2])
    senders = torch.tensor([1, 0, 2, 1])
    zeros = torch.zeros(4, 4)
    vectors = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, -1.2, 0.0],
    ])
    connected = core.get_output(
        embedding, zeros, receivers, senders, vectors, atomic_numbers
    )
    assert torch.allclose(
        connected["atom_feature"],
        connected["isolated_atom_feature"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(connected["atom_steerable_feature"]) == 0

    live = torch.randn(4, 4)
    interacting = core.get_output(
        embedding, live, receivers, senders, vectors, atomic_numbers
    )
    assert not torch.allclose(
        interacting["atom_feature"],
        interacting["isolated_atom_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
