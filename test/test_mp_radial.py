"""Tensor and pipeline basis calls share parameters, field shapes, and derivatives."""

import pytest
import torch

from enerzyme.models.layers import rbf
from enerzyme.models.mp.basis.envelope import (
    C3CutoffEnvelope, Envelope, facto_polynomial_cutoff, polynomial_transition,
)
from enerzyme.models.mp.basis.radial import RadialBasis


@pytest.mark.parametrize(("kind", "layer"), [
    ("gaussian", rbf.GaussianRBFLayer), ("bernstein", rbf.BernsteinRBFLayer),
    ("exp_gaussian", rbf.ExponentialGaussianRBFLayer), ("exp_bernstein", rbf.ExponentialBernsteinRBFLayer),
    ("sinc", rbf.SincRBFLayer), ("bessel", rbf.BesselRBFLayer),
    ("expnormal", rbf.ExpNormalSmearing), ("spherical_bessel", rbf.SphericalBesselRBFLayer),
])
def test_tensor_basis_matches_layer(kind, layer):
    tensor_basis = RadialBasis(kind, num_rbf=3, cutoff_sr=4.).double()
    pipeline = layer(num_rbf=3, cutoff_sr=4.).double()
    pipeline.load_state_dict(tensor_basis.state_dict(), strict=True)
    distances = torch.tensor([0.7, 2., 3.9, 4.5], dtype=torch.float64, requires_grad=True)
    expected = pipeline({"Dij_sr": distances})["rbf"]
    actual = tensor_basis(distances)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), distances)[0], torch.autograd.grad(expected.sum(), distances)[0],
        rtol=0., atol=0.,
    )
    assert tensor_basis(distances[:0]).shape == (0, actual.shape[-1])


@pytest.mark.parametrize("kind", ["bessel", "gaussian", "sinc", "expnormal", "exp_gaussian", "exp_bernstein"])
def test_factory_trainable(kind):
    basis = RadialBasis(kind, 3, 5., trainable=True).double()
    basis(torch.tensor([1., 2.], dtype=torch.float64)).sum().backward()
    assert all(p.grad is not None for p in basis.parameters())
    assert any(p.requires_grad for p in basis.parameters())
    fixed = RadialBasis(kind, 3, 5., trainable=False)
    assert not any(p.requires_grad for p in fixed.parameters())


def test_chebyshev_shift_and_envelope_aliases():
    distances = torch.tensor([0., 0.5, 1., 1.1], dtype=torch.float64)
    x = 2 * distances - 1
    expected = torch.stack((torch.ones_like(x), x, 2 * x.square() - 1), dim=-1)
    torch.testing.assert_close(RadialBasis("chebyshev", 3, 1., shifted=True)(distances), expected)
    for kind, reference in [("smoothstep", polynomial_transition), ("poly9", facto_polynomial_cutoff)]:
        torch.testing.assert_close(Envelope(kind, cutoff=1.)(distances), reference(distances, 1.))
    torch.testing.assert_close(Envelope("c3", cutoff=1., exponent=7)(distances), C3CutoffEnvelope(1., 7)(distances))
