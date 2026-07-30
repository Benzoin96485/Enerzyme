"""Tests for SphereSampleReadout."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])


def test_sphere_sample_readout_scalar_shapes():
    from enerzyme.models.escn import eSCNCore
    from enerzyme.models.layers import SphereSampleReadout

    torch.manual_seed(0)
    core = eSCNCore(
        dim_embedding=16,
        num_rbf=8,
        sphere_channels=16,
        hidden_channels=32,
        edge_channels=16,
        lmax=2,
        mmax=1,
        num_layers=1,
    )
    readout = SphereSampleReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[core],
        head_type="escn_mlp",
        num_sphere_samples=16,
    )
    N = 5
    feat = torch.randn(N, (2 + 1) ** 2, 16)
    out = readout.get_output(atom_sphere_feature=feat)
    assert out["Ea"].shape == (N,)
    assert out["Qa"].shape == (N,)
    assert torch.isfinite(out["Ea"]).all()


def test_sphere_sample_readout_vector_field():
    from enerzyme.models.escn import eSCNCore
    from enerzyme.models.layers import SphereSampleReadout

    torch.manual_seed(0)
    core = eSCNCore(
        dim_embedding=8,
        num_rbf=4,
        sphere_channels=8,
        hidden_channels=16,
        edge_channels=8,
        lmax=1,
        mmax=1,
        num_layers=1,
    )
    readout = SphereSampleReadout(
        output_fields=set(),
        vector_output_fields={"Fa"},
        built_layers=[core],
        head_type="escn_mlp",
        num_sphere_samples=12,
    )
    N = 4
    feat = torch.randn(N, 4, 8)
    out = readout.get_output(atom_sphere_feature=feat)
    assert out["Fa"].shape == (N, 3)


def test_sphere_sample_with_core_mmax_lt_lmax():
    """Core emits full (lmax+1)^2 node coeffs even when message SO2 uses mmax < lmax."""
    from enerzyme.models.escn import eSCNCore
    from enerzyme.models.layers import SphereSampleReadout

    torch.manual_seed(0)
    lmax, mmax = 2, 1
    sphere_channels = 16
    core = eSCNCore(
        dim_embedding=16,
        num_rbf=8,
        sphere_channels=sphere_channels,
        hidden_channels=32,
        edge_channels=16,
        lmax=lmax,
        mmax=mmax,
        num_layers=1,
    )
    readout = SphereSampleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="escn_mlp",
        num_sphere_samples=16,
    )
    N = 4
    Za = torch.tensor([1, 6, 8, 1])
    atom_embedding = torch.randn(N, 16)
    idx_i, idx_j = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    Ra = torch.randn(N, 3)
    vij = Ra[idx_j] - Ra[idx_i]
    Dij = torch.linalg.norm(vij, dim=1).clamp(min=1e-6)
    rbf = torch.exp(-((Dij.unsqueeze(1) - torch.linspace(0, 5, 8)) ** 2))
    out = core.get_output(
        atom_embedding=atom_embedding,
        Za=Za,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
    )
    assert out["atom_sphere_feature"].shape == (N, (lmax + 1) ** 2, sphere_channels)
    ea = readout.get_output(atom_sphere_feature=out["atom_sphere_feature"])["Ea"]
    assert ea.shape == (N,)
    assert torch.isfinite(ea).all()


def test_sphere_sample_with_core_energy_rotation_invariance():
    """Geometry rotation leaves SphereSampleReadout Ea approximately invariant."""
    from enerzyme.models.escn import eSCNCore
    from enerzyme.models.layers import SphereSampleReadout

    torch.manual_seed(0)
    core = eSCNCore(
        dim_embedding=16,
        num_rbf=8,
        sphere_channels=16,
        hidden_channels=32,
        edge_channels=16,
        lmax=2,
        mmax=2,
        num_layers=1,
    )
    readout = SphereSampleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="escn_mlp",
        num_sphere_samples=32,
    )
    N = 4
    Za = torch.tensor([1, 6, 8, 1])
    atom_embedding = torch.randn(N, 16)
    idx_i, idx_j = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    Ra = torch.randn(N, 3)
    vij = Ra[idx_j] - Ra[idx_i]
    Dij = torch.linalg.norm(vij, dim=1).clamp(min=1e-6)
    rbf = torch.exp(-((Dij.unsqueeze(1) - torch.linspace(0, 5, 8)) ** 2))

    out0 = core.get_output(
        atom_embedding=atom_embedding,
        Za=Za,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
    )
    e0 = readout.get_output(atom_sphere_feature=out0["atom_sphere_feature"])["Ea"]

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    Ra_r = Ra @ q.T
    vij_r = Ra_r[idx_j] - Ra_r[idx_i]
    out1 = core.get_output(
        atom_embedding=atom_embedding,
        Za=Za,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij_r,
    )
    e1 = readout.get_output(atom_sphere_feature=out1["atom_sphere_feature"])["Ea"]
    assert torch.allclose(e0, e1, atol=1e-4, rtol=1e-4)
