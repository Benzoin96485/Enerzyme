"""CAMP moments: Chebyshev pairs, construction rules, and contractions."""

from __future__ import annotations

import torch

from enerzyme.models.camp.moment import (
    AtomicMoment,
    cartesian_dyads,
    contract_feature_dyad,
    contract_moments,
)
from enerzyme.models.camp.radial import SpeciesPairChebyshev, chebyshev_first
from enerzyme.models.camp.rules import atomic_paths, hyper_moment_rules
from enerzyme.models.cutoff import polynomial_envelope


def test_chebyshev_recurrence_uses_the_unit_interval_argument():
    degree = 4
    x = torch.tensor([0.0, 0.5, 1.0])
    values = chebyshev_first(degree, x)
    assert values.shape == (5, 3)
    assert torch.allclose(values[0], torch.ones(3))
    assert torch.allclose(values[1], x)
    assert torch.allclose(values[2], 2 * x**2 - 1)
    assert torch.allclose(values[3], 4 * x**3 - 3 * x)
    # T2(0) = -1, so a degree-2 basis at r = 0 is not just the constant term.
    assert torch.allclose(values[2, 0], torch.tensor(-1.0))


def test_species_pair_basis_is_zero_at_the_cutoff_and_depends_on_the_pair():
    torch.manual_seed(0)
    basis = SpeciesPairChebyshev(n_channels=2, n_species=2, degree=2, cutoff=4.0, envelope_p=6)
    with torch.no_grad():
        basis.coeff.zero_()
        basis.coeff[0, 1, :, 0] = 1.0
        basis.coeff[1, 0, :, 1] = 2.0
    distance = torch.tensor([1.0, 4.0, 5.0])
    zi = torch.tensor([0, 0, 1])
    zj = torch.tensor([1, 1, 0])
    out = basis(distance, zi, zj)
    window = polynomial_envelope(distance[:1], 4.0, 6)
    assert torch.allclose(out[0], window.expand(2))
    assert torch.allclose(out[1], torch.zeros(2))
    assert torch.allclose(out[2], torch.zeros(2))
    assert basis.coeff.grad is None
    out[0].sum().backward()
    assert basis.coeff.grad is not None
    assert torch.count_nonzero(basis.coeff.grad[0, 1]) > 0


def test_moment_rules_match_the_supplement_through_rank_3():
    assert atomic_paths(2, 2, 0) == ((0, 0), (1, 1), (2, 2))
    assert atomic_paths(2, 2, 1) == ((0, 1), (1, 2))
    assert atomic_paths(2, 2, 2) == ((0, 2),)
    assert atomic_paths(2, 0, 1) == ((0, 1),)
    assert set(hyper_moment_rules(3, 0)) == {
        (0,),
        (1, 1),
        (2, 2),
        (3, 3),
        (1, 1, 2),
        (1, 2, 3),
        (1, 1, 1, 3),
    }
    assert set(hyper_moment_rules(3, 1)) == {(1,), (1, 2), (2, 3), (1, 1, 3)}
    assert set(hyper_moment_rules(3, 2)) == {(2,), (1, 3)}
    assert set(hyper_moment_rules(3, 3)) == {(3,)}
    assert hyper_moment_rules(0, 0) == ((0,),)


def test_dyad_is_the_unit_outer_product_and_contraction_is_a_dot():
    vector = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dyads = cartesian_dyads(vector, 2)
    assert torch.allclose(dyads[0], torch.ones(2))
    assert torch.allclose(dyads[1][0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(dyads[1][1], torch.zeros(3))
    assert torch.allclose(dyads[2][0], torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    feature = torch.tensor([[[1.0, 0.0, 0.0]]])  # (C=1, E=1, 3)
    contracted = contract_feature_dyad(feature, dyads[2][:1], 1, 2)
    assert contracted.shape == (1, 1, 3)
    assert torch.allclose(contracted[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    scalar = contract_feature_dyad(feature, dyads[1][:1], 1, 1)
    assert torch.allclose(scalar, torch.ones(1, 1))


def test_hyper_moment_scalar_is_the_supplement_contraction():
    moment = {
        0: torch.tensor([[2.0]]),
        1: torch.tensor([[[0.0, 3.0, 0.0]]]),
        2: torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]),
    }
    # Shapes: rank 0 (C, N), rank 1 (C, N, 3), rank 2 (C, N, 3, 3). C = N = 1.
    copied = contract_moments(moment, (0,))
    dots = contract_moments(moment, (1, 1))
    frobenius = contract_moments(moment, (2, 2))
    assert torch.allclose(copied, torch.tensor([[2.0]]))
    assert torch.allclose(dots, torch.tensor([[9.0]]))
    assert torch.allclose(frobenius, torch.tensor([[3.0]]))


def test_atomic_moment_with_identity_maps_counts_one_neighbor():
    torch.manual_seed(0)
    moment = AtomicMoment(
        n_channels=1,
        n_species=1,
        max_v=1,
        max_feature_rank=0,
        avg_num_neighbors=1.0,
        degree=0,
        cutoff=5.0,
        envelope_p=6,
        radial_hidden=[],
    )
    with torch.no_grad():
        moment.radial.coeff.fill_(1.0)
        linear = moment.radial_mlp["0_0_0"][0]
        linear.weight.copy_(torch.eye(1))
        linear.bias.zero_()
        linear = moment.radial_mlp["1_0_1"][0]
        linear.weight.copy_(torch.eye(1))
        linear.bias.zero_()
        moment.channel_mix["0"].weight.copy_(torch.eye(1))
        moment.channel_mix["1"].weight.copy_(torch.eye(1))
    distance = torch.tensor([1.0])
    vectors = torch.tensor([[0.0, 2.0, 0.0]])
    species = torch.zeros(1, dtype=torch.long)
    receivers = torch.zeros(1, dtype=torch.long)
    senders = torch.ones(1, dtype=torch.long)
    features = {0: torch.ones(1, 2)}
    out = moment(distance, vectors, species, species, receivers, senders, features, 2)
    window = polynomial_envelope(distance, 5.0, 6)
    assert out[0].shape == (1, 2)
    assert out[1].shape == (1, 2, 3)
    assert torch.allclose(out[0][0, 0], window)
    assert torch.allclose(out[0][0, 1], torch.zeros(()))
    assert torch.allclose(out[1][0, 0], torch.tensor([0.0, 1.0, 0.0]) * window)
    # Rotating the neighbor rotates the vector moment and keeps the scalar.
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = moment(
        distance,
        vectors @ rotation,
        species,
        species,
        receivers,
        senders,
        features,
        2,
    )
    assert torch.allclose(rotated[0], out[0])
    assert torch.allclose(rotated[1][0, 0], out[1][0, 0] @ rotation)
