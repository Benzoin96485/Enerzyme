"""Check basis conventions against independent spherical harmonics and their gradients."""

import pytest
import torch
from e3nn import o3

from enerzyme.models.mp.basis.angular import spherical_harmonics


@pytest.mark.parametrize("layout,maximum", [("so3krates", 4), ("gemnet", 1), ("gemnet", 4), ("gemnet", 8)])
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
def test_tesseral_matches_e3nn(layout, maximum, normalization):
    directions = torch.randn(3, 5, 3, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
    directions.requires_grad_()
    degrees = list(range(maximum, -1, -1)) + [0]
    actual = spherical_harmonics(directions, degrees, layout=layout, normalization=normalization)
    expected = o3.spherical_harmonics(degrees, directions[..., [1, 2, 0]], normalize=True, normalization=normalization)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    weights = torch.sin(torch.arange(actual.shape[-1], dtype=actual.dtype))
    got = torch.autograd.grad((actual * weights).sum(), directions)[0]
    want = torch.autograd.grad((expected * weights).sum(), directions)[0]
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)


def test_dpa4c_solid_harmonics_preserve_degree_homogeneity_and_cartesian_order():
    v = torch.tensor([[1., 2., -0.4], [0., 0., 0.]], dtype=torch.float64)
    actual = spherical_harmonics(v, lmax=4, layout="dpa4c", normalize_input=False)
    expected = o3.spherical_harmonics(list(range(5)), v[:, [1, 2, 0]], normalize=False, normalization="norm")
    expected[:, 1:4] = v
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_polynomial_layout_accepts_batched_vectors_and_omits_unrequested_degrees():
    v = torch.tensor([[[1., 2., 3.]]], dtype=torch.float64)
    full = spherical_harmonics(v, lmax=4, layout="polynomial", normalization="none", normalize_input=False)
    torch.testing.assert_close(full[..., 1:4], v)
    partial = spherical_harmonics(v, [3, 1], layout="polynomial", normalize_input=False)
    torch.testing.assert_close(partial, torch.cat((full[..., 9:16], v), dim=-1))


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [False, True])
def test_single_degree_matches_e3nn_and_cartesian_gradient(normalization, normalize):
    from enerzyme.models.mp.basis.angular import spherical_harmonic_degree

    vectors = torch.randn(2, 3, 3, generator=torch.Generator().manual_seed(11), dtype=torch.float64,
                          requires_grad=True)
    for degree in (1, 2, 4, 6):
        expected = o3.spherical_harmonics(degree, vectors, normalize=normalize, normalization=normalization)
        actual = spherical_harmonic_degree(degree, vectors, normalize=normalize, normalization=normalization)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        probe = torch.arange(actual.numel(), dtype=vectors.dtype).reshape_as(actual)
        got, = torch.autograd.grad((actual * probe).sum(), vectors)
        ref, = torch.autograd.grad((expected * probe).sum(), vectors)
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
