import pytest
import torch
from e3nn import o3

from enerzyme.models.mp.equivariant.cg import coupling_paths, wigner_3j, wigner_6j


@pytest.mark.parametrize("degrees", [(0, 1, 1), (1, 1, 2), (2, 1, 2), (3, 2, 4)])
def test_real_couplings_follow_e3nn_and_orthonormal_columns(degrees):
    normalized = wigner_3j(*degrees, dtype=torch.float64)
    reference = o3.wigner_3j(*degrees, dtype=torch.float64)
    torch.testing.assert_close(normalized, reference, atol=0, rtol=0)
    cg = wigner_3j(*degrees, dtype=torch.float64, normalized=False).flatten(0, 1)
    eye = torch.eye(cg.shape[1], dtype=cg.dtype)
    torch.testing.assert_close(cg.T @ cg, eye, atol=1e-13, rtol=0)
    normalized.zero_()
    torch.testing.assert_close(wigner_3j(*degrees), reference.to(torch.get_default_dtype()), atol=0, rtol=0)


def test_path_table_matches_e3nn_with_repeated_irreps_and_both_parities():
    ir = [o3.Irrep(s) for s in ("0e", "2e", "1o", "2o", "1o")]
    expected = tuple((i, j, k) for i, a in enumerate(ir) for j, b in enumerate(ir)
                     for k, c in enumerate(ir) if c in a * b)
    assert coupling_paths(ir, ir, ir) == expected


def test_six_j_recoupling_coefficient():
    assert float(wigner_6j(1, 1, 1, 1, 1, 1)) == 1 / 6


@pytest.mark.parametrize("degrees", [(0, 0, 0), (1, 1, 2), (1, 2, 2), (3, 1, 2), (1, 1, 4)])
def test_gaunt_agrees_with_direct_sphere_quadrature(degrees):
    from math import pi
    from enerzyme.models.mp.equivariant.cg import gaunt
    from enerzyme.models.mp.equivariant.s2.lebedev import load_lebedev_rule

    xyz, weights = load_lebedev_rule(13)
    xyz = torch.as_tensor(xyz, dtype=torch.float64)
    harmonics = [o3.spherical_harmonics(l, xyz, normalize=False, normalization="integral") for l in degrees]
    integral = torch.einsum("n,ni,nj,nk->ijk", torch.as_tensor(weights) * (4 * pi), *harmonics)
    torch.testing.assert_close(gaunt(*degrees, dtype=torch.float64), integral, atol=1e-12, rtol=1e-12)
