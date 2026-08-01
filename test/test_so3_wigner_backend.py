"""Shared e3nn/Jd Wigner-D backend checks (eSCN / Equiformer / DPA4)."""

from __future__ import annotations

import pytest
import torch


def test_wigner_from_rotation_matrix_is_orthogonal():
    from enerzyme.models.so3.wigner_jd import wigner_from_rotation_matrix

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(5, 3, 3, dtype=torch.float64))
    q = q * torch.det(q).sign().view(-1, 1, 1)
    W = wigner_from_rotation_matrix(q, end_lmax=3)
    eye = torch.eye(16, dtype=torch.float64).expand(5, -1, -1)
    torch.testing.assert_close(W @ W.transpose(-1, -2), eye, atol=1e-6, rtol=1e-6)


def test_wigner_from_quaternion_matches_calculator():
    from enerzyme.models.so3.wigner_jd import wigner_from_rotation_matrix
    from enerzyme.models.so3.wigner_quaternion import (
        WignerDCalculator,
        _DPA4_CARTESIAN_BASIS,
        build_edge_quaternion,
        quaternion_normalize,
        quaternion_to_rotation_matrix,
    )

    torch.manual_seed(1)
    vij = torch.randn(7, 3, dtype=torch.float64)
    q = build_edge_quaternion(vij)
    D_calc, Dt_calc = WignerDCalculator(3).double()(q)
    R = quaternion_to_rotation_matrix(quaternion_normalize(q))
    A = _DPA4_CARTESIAN_BASIS.to(dtype=R.dtype)
    D_ref = wigner_from_rotation_matrix(A @ R @ A.T, end_lmax=3)
    torch.testing.assert_close(D_calc, D_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(Dt_calc, D_ref.transpose(-1, -2), atol=1e-10, rtol=1e-10)


def test_so3_rotation_matches_shared_packed_wigner():
    from enerzyme.models.so3.rotation import SO3_Rotation, init_edge_rot_mat
    from enerzyme.models.so3.wigner_jd import wigner_from_rotation_matrix

    torch.manual_seed(2)
    vij = torch.randn(4, 3, dtype=torch.float64)
    R = init_edge_rot_mat(vij)
    so3 = SO3_Rotation(R, lmax=2)
    W = wigner_from_rotation_matrix(R, end_lmax=2)
    torch.testing.assert_close(so3.wigner, W, atol=1e-10, rtol=1e-10)


def test_fused_packed_core_matches_shared_before_m_primary():
    from enerzyme.models.so3.rotation import init_edge_rot_mat
    from enerzyme.models.so3.rotation_fused import SO3RotationFused
    from enerzyme.models.so3.wigner_jd import wigner_from_rotation_matrix

    torch.manual_seed(3)
    vij = torch.randn(3, 3, dtype=torch.float64)
    R = init_edge_rot_mat(vij)
    fused = SO3RotationFused(lmax=2, mmax=1, use_rotation_mask=False)
    packed = fused._rotation_to_wigner_matrix(R, 0, 2)
    torch.testing.assert_close(
        packed, wigner_from_rotation_matrix(R, end_lmax=2), atol=1e-10, rtol=1e-10
    )


def test_wigner_d_rejects_l_above_jd_max():
    from enerzyme.models.so3.wigner_jd import max_wigner_lmax, wigner_D
    from enerzyme.models.so3.wigner_quaternion import WignerDCalculator

    bad = max_wigner_lmax() + 1
    with pytest.raises(NotImplementedError):
        wigner_D(bad, torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with pytest.raises(NotImplementedError):
        WignerDCalculator(bad)


def test_dpa4_lmax3_forward_and_geometry_grad():
    from enerzyme.models.dpa4 import DPA4Core

    torch.manual_seed(4)
    n = 4
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])
    positions = torch.randn(n, 3, requires_grad=True)
    core = DPA4Core(
        dim_embedding=8,
        channels=8,
        lmax=3,
        mmax=1,
        n_blocks=1,
        mixing_layers=1,
        n_radial=8,
        ffn_neurons=16,
    )
    out = core.get_output(
        torch.randn(n, 8),
        torch.tensor([1, 6, 8, 1]),
        idx_i,
        idx_j,
        positions[idx_i] - positions[idx_j],
        torch.zeros(n, dtype=torch.long),
    )
    assert out["atom_sphere_feature"].shape == (n, 16, 8)
    assert torch.isfinite(out["atom_feature"]).all()
    grad = torch.autograd.grad(out["atom_feature"].square().sum(), positions)[0]
    assert torch.isfinite(grad).all()
