"""CACE monomials, invariant contractions, and the two-body identity."""

from __future__ import annotations

import torch


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_monomials_match_the_components_and_the_shell_count():
    from enerzyme.models.cace.basis import (
        CartesianAngularBasis,
        cartesian_monomials,
        multi_indices,
    )

    basis = CartesianAngularBasis(2)
    assert basis.n_angular == 10
    assert basis.shells.tolist() == [[0, 1], [1, 4], [4, 10]]
    unit = torch.tensor([[0.5, -0.3, 0.2]])
    values = cartesian_monomials(unit, basis.powers)
    expected = []
    for lx, ly, lz in multi_indices(2):
        expected.append(0.5**lx * (-0.3) ** ly * 0.2**lz)
    assert torch.allclose(values[0], torch.tensor(expected), atol=1e-6)
    zero = cartesian_monomials(torch.zeros(1, 3), basis.powers)
    assert zero[0, 0].item() == 1.0
    assert torch.count_nonzero(zero[0, 1:]) == 0
    empty = cartesian_monomials(torch.zeros(0, 3), basis.powers)
    assert empty.shape == (0, 10)


def test_invariant_counts_and_pair_contraction_is_the_power_of_the_dot_product():
    from enerzyme.models.cace.basis import (
        CartesianAngularBasis,
        InvariantSymmetrizer,
        invariant_keys,
    )

    assert len(invariant_keys(1, 2)) == 2
    assert len(invariant_keys(2, 3)) == 4
    assert len(invariant_keys(3, 3)) == 6
    assert len(invariant_keys(3, 4)) == 7
    assert invariant_keys(3, 3) == [
        (1, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1, 1),
        (3, 1, 2),
    ]
    symmetrizer = InvariantSymmetrizer(3, 4).double()
    assert symmetrizer.n_invariants == len(invariant_keys(3, 4))

    angular = CartesianAngularBasis(3)
    left = torch.tensor([[0.36, -0.8, 0.48]], dtype=torch.float64)
    right = torch.tensor([[0.8, 0.3, -0.51961524227]], dtype=torch.float64)
    left = left / torch.linalg.vector_norm(left)
    right = right / torch.linalg.vector_norm(right)
    monomials = angular.double()(torch.cat([left, right], dim=0))
    density = monomials.sum(dim=0).view(1, 1, -1, 1)
    invariants = symmetrizer(density)[0, 0, :, 0]
    dot = torch.dot(left[0], right[0])
    # ν = 1 is the sum of two scalar monomials.
    assert torch.allclose(invariants[0], torch.tensor(2.0, dtype=torch.float64))
    for degree in range(1, 4):
        expect = 2.0 + 2.0 * dot**degree
        assert torch.allclose(invariants[degree], expect, atol=1e-6, rtol=1e-6)

    scalar_only = torch.zeros_like(density)
    scalar_only[:, :, 0, :] = 3.0
    higher = symmetrizer(scalar_only)[0, 0, 1:, 0]
    assert torch.count_nonzero(higher) == 0


def test_invariants_are_unchanged_by_rotation_and_inversion():
    from enerzyme.models.cace.basis import CartesianAngularBasis, InvariantSymmetrizer

    torch.manual_seed(0)
    angular = CartesianAngularBasis(3).double()
    symmetrizer = InvariantSymmetrizer(3, 3).double()
    vectors = torch.randn(4, 3, dtype=torch.float64)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)

    def invariants(dirs: torch.Tensor) -> torch.Tensor:
        density = angular(dirs).sum(dim=0).view(1, 1, -1, 1)
        return symmetrizer(density)

    base = invariants(vectors)
    rotation = _rotation(1)
    rotated = invariants(vectors @ rotation)
    assert torch.allclose(base, rotated, atol=1e-8, rtol=1e-8)
    assert not torch.allclose(angular(vectors)[:, 1:4], angular(vectors @ rotation)[:, 1:4])
    inverted = invariants(-vectors)
    assert torch.allclose(base, inverted, atol=1e-8, rtol=1e-8)


def test_radial_mix_is_independent_per_shell():
    from enerzyme.models.cace.basis import CartesianAngularBasis
    from enerzyme.models.cace.interaction import RadialChannelMix

    angular = CartesianAngularBasis(1)
    mix = RadialChannelMix(2, 1, 1, angular.shells)
    with torch.no_grad():
        mix.weight.zero_()
        mix.weight[0, 0, 0, 0] = 1.0
        mix.weight[1, 1, 0, 0] = 2.0
    raw = torch.zeros(1, 2, angular.n_angular, 1)
    raw[0, 0, :, 0] = 1.0
    raw[0, 1, :, 0] = 3.0
    mixed = mix(raw)
    assert mixed.shape == (1, 1, 4, 1)
    assert torch.allclose(mixed[0, 0, 0, 0], torch.tensor(1.0))
    assert torch.allclose(mixed[0, 0, 1:, 0], torch.full((3,), 6.0))


def test_compiled_radial_shell_mix_with_force_loss():
    from enerzyme.models.cace.interaction import RadialChannelMix

    torch.manual_seed(14)
    model = RadialChannelMix(2, 3, 4, torch.tensor([[0, 1], [1, 4], [4, 10]])).double()
    raw = torch.randn(5, 2, 10, 4, dtype=torch.float64, requires_grad=True)

    def objective(x):
        mixed = model(x)
        return mixed.square().sum(), mixed

    def derivatives(x):
        _, pullback, mixed = torch.func.vjp(objective, x, has_aux=True)
        return mixed, pullback(x.new_ones(()))[0]

    eager = derivatives(raw)
    actual = torch.compile(derivatives, fullgraph=True, backend="aot_eager")(raw)
    for values in zip(actual, eager):
        torch.testing.assert_close(*values)
    actual_grad = torch.autograd.grad(actual[1].square().sum(), (raw, model.weight))
    expected_grad = torch.autograd.grad(eager[1].square().sum(), (raw, model.weight))
    for value, target in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, target)
