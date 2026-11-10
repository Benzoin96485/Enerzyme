"""GRACE radial basis, Clebsch–Gordan product, and energy-head algebra."""

from __future__ import annotations

import math

import torch


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_chebyshev_matches_the_closed_form_and_the_window_closes():
    from enerzyme.models.grace.radial import (
        chebyshev_basis,
        powered_cosine_window,
        spherical_bessel_basis,
    )

    distance = torch.tensor([0.0, 1.5, 3.0])
    basis = chebyshev_basis(distance, 3, 3.0)
    expect = torch.tensor(
        [
            [1.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(basis, expect, atol=1e-6)

    window = powered_cosine_window(torch.tensor([0.0, 3.0, 4.0]), 3.0, 4)
    assert torch.allclose(window, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    radius = torch.tensor(1.7, requires_grad=True)
    powered_cosine_window(radius, 3.0, 4).backward()
    assert torch.isfinite(radius.grad)

    bessel = spherical_bessel_basis(torch.tensor([0.0, 2.0]), 2, 4.0)
    limit = math.sqrt(2.0 / 4.0) * math.pi / 4.0
    assert torch.allclose(bessel[0], torch.tensor([limit, 2 * limit]), atol=1e-5)
    assert bessel.shape == (2, 2)
    assert torch.isfinite(bessel).all()


def test_radial_functions_vanish_at_the_cutoff():
    from enerzyme.models.grace.radial import RadialEmbedding

    radial = RadialEmbedding(
        n_basis=4, n_rad=3, lmax=2, cutoff=3.0, cutoff_order=4, hidden=8
    )
    values = radial(torch.tensor([0.4, 3.0, 3.5]))
    assert values.shape == (3, 3, 3)
    assert torch.count_nonzero(values[1]) == 0
    assert torch.count_nonzero(values[2]) == 0
    assert torch.isfinite(values).all()
    empty = radial(torch.zeros(0))
    assert empty.shape == (0, 3, 3)


def test_channel_aligned_product_is_rotation_equivariant():
    from e3nn import o3

    from enerzyme.models.grace.coupling import SparseCGProduct
    from enerzyme.models.grace.features import pack_features
    from enerzyme.models.so3.spherical_harmonics import spherical_harmonics

    torch.manual_seed(0)
    product = SparseCGProduct(channels=2, lmax_left=1, lmax_right=1, Lmax=1)
    product.eval()
    vectors = torch.randn(5, 3)
    scales = torch.tensor([1.0, -0.4])

    def features(dirs: torch.Tensor):
        harmonics = spherical_harmonics(
            dirs, [0, 1], layout="e3nn", normalization="component"
        )
        scalar = harmonics[:, :1].unsqueeze(1) * scales.view(1, 2, 1)
        vector = harmonics[:, 1:].unsqueeze(1) * scales.view(1, 2, 1)
        return [scalar.contiguous(), vector.contiguous()]

    original = product(*([features(vectors)] * 2))
    rotation = _rotation(1)
    # Row vectors transform as ``v @ R``. With this layout, e3nn's
    # ``D_from_matrix(R)`` acts on the right: ``Y(v @ R) = Y(v) @ D``.
    rotated = product(*([features(vectors @ rotation)] * 2))
    irreps = o3.Irreps("2x0e+2x1o")
    matrix = irreps.D_from_matrix(rotation)
    expected = pack_features(original) @ matrix
    assert torch.allclose(pack_features(rotated), expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(original[0], rotated[0], atol=1e-5, rtol=1e-5)


def test_density_powers_match_the_finnis_sinclair_cases():
    from enerzyme.models.grace.readout import GRACEEnergyReadout, _power

    values = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
    assert torch.allclose(_power(values, 1.0, 1e-8), values)
    assert torch.allclose(_power(values, 2.0, 1e-8), values.square())
    assert torch.allclose(_power(values, 3.0, 1e-8), values.pow(3))
    fractional = _power(values, 0.5, 1e-8)
    assert torch.allclose(fractional[3:], values[3:].sqrt(), atol=1e-6)
    assert fractional[0] < 0
    assert torch.isfinite(fractional).all()

    class _Core:
        is_grace_core = True
        n_densities = 3
        readout_head = "finnis_sinclair"
        readout_hidden = 0
        readout_activation = "silu"
        fs_terms = ((1.5, 1.0), (-0.5, 2.0), (1.0, 0.5))

    head = GRACEEnergyReadout(built_layers=[_Core()], max_Za=4)
    with torch.no_grad():
        head.shift.copy_(torch.tensor([0.0, 0.2, -0.1, 0.0, 0.4]))
    density = torch.tensor([[1.0, -2.0, 4.0], [0.0, 0.0, 0.0]])
    species = torch.tensor([1, 4])
    energy = head.get_Ea(density, species)
    manual = torch.tensor(
        [
            1.5 * 1.0 + (-0.5) * 4.0 + 1.0 * 2.0 + 0.2,
            0.4,
        ]
    )
    assert torch.allclose(energy, manual, atol=1e-6)


def test_linear_mlp_head_uses_only_the_first_channel_as_the_linear_term():
    from enerzyme.models.grace.readout import GRACEEnergyReadout

    class _Core:
        is_grace_core = True
        n_densities = 3
        readout_head = "linear_mlp"
        readout_hidden = 4
        readout_activation = "silu"
        fs_terms = None

    torch.manual_seed(0)
    head = GRACEEnergyReadout(built_layers=[_Core()], max_Za=2)
    with torch.no_grad():
        head.shift.zero_()
        # Zero the MLP so the energy is exactly the linear density.
        for parameter in head.mlp.parameters():
            parameter.zero_()
    density = torch.tensor([[0.3, 10.0, -7.0], [1.2, 4.0, 5.0]])
    energy = head.get_Ea(density, torch.tensor([0, 1]))
    assert torch.allclose(energy, density[:, 0], atol=1e-6)
