import sys

import pytest
import torch

sys.path.extend(["..", "."])


def test_coefficient_mapping_shapes():
    from enerzyme.models.so3 import CoefficientMapping

    device = torch.device("cpu")
    mapping = CoefficientMapping([2], [2], device)
    # l=0..2 with |m|<=min(l,2): 1 + 3 + 5 = 9
    assert len(mapping.l_harmonic) == 9
    assert mapping.m_size[0].item() == 3  # l=0,1,2 real m=0
    assert mapping.to_m.shape == (9, 9)
    idx = mapping.coefficient_idx(1, 1)
    assert len(idx) == 1 + 3  # l=0 + l=1


def test_so2_conv_output_shape():
    from enerzyme.models.so3 import SO2Conv

    m = 1
    sphere_channels = 4
    hidden_channels = 8
    edge_channels = 6
    lmax_list = [2]
    mmax_list = [2]
    conv = SO2Conv(
        m, sphere_channels, hidden_channels, edge_channels, lmax_list, mmax_list, torch.nn.SiLU()
    )
    # for m=1, lmax=2: coefficients = lmax-m+1 = 2; channels = 2 * sphere_channels
    num_channels = (2 - 1 + 1) * sphere_channels
    num_edges = 5
    x_m = torch.randn(num_edges, 2, num_channels)
    x_edge = torch.randn(num_edges, edge_channels)
    out = conv(x_m, x_edge)
    assert out.shape == (num_edges, 2, num_channels)


def test_rotate_inverse_roundtrip():
    from enerzyme.models.so3 import SO3_Embedding, SO3_Rotation, init_edge_rot_mat

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    lmax = 2
    num_channels = 3
    num_edges = 4
    vij = torch.randn(num_edges, 3, dtype=dtype)
    vij = vij / torch.linalg.norm(vij, dim=1, keepdim=True).clamp(min=1e-8)
    rot_mat = init_edge_rot_mat(vij)
    so3_rot = SO3_Rotation(rot_mat, lmax)

    x = SO3_Embedding(num_edges, [lmax], num_channels, device, dtype)
    x.embedding = torch.randn(num_edges, (lmax + 1) ** 2, num_channels, dtype=dtype)
    original = x.embedding.clone()

    from enerzyme.models.so3 import CoefficientMapping

    mapping = CoefficientMapping([lmax], [lmax], device)
    x._rotate([so3_rot], [lmax], [lmax])
    x._rotate_inv([so3_rot], mapping)
    assert torch.allclose(x.embedding, original, atol=1e-5, rtol=1e-5)


def test_init_edge_rot_mat_orthonormal():
    from enerzyme.models.so3 import init_edge_rot_mat

    torch.manual_seed(1)
    vij = torch.randn(7, 3)
    R = init_edge_rot_mat(vij)
    assert R.shape == (7, 3, 3)
    eye = torch.eye(3).expand(7, -1, -1)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(7), atol=1e-5)


def test_real_sh_and_l0_shapes():
    from enerzyme.models.so3 import L0Contraction, RealSphericalHarmonics

    degrees = [1, 2, 3]
    sh = RealSphericalHarmonics(degrees)
    m_tot = sum(2 * l + 1 for l in degrees)
    y = sh(torch.randn(4, 3))
    assert y.shape == (4, m_tot)
    inv = L0Contraction(degrees)(y)
    assert inv.shape == (4, len(degrees))
    assert torch.isfinite(inv).all()

