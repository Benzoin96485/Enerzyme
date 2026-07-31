"""EquiformerV2 Core smoke / equivariance / build_model tests."""

from __future__ import annotations

import sys

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])


def _complete_graph_edges(num_nodes: int):
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _random_so3(dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def test_equiformer_v2_core_atom_feature_shape():
    from enerzyme.models.equiformer_v2 import EquiformerV2Core

    torch.manual_seed(0)
    N = 6
    sphere_channels = 16
    dim_embedding = 16
    num_rbf = 8
    lmax = 2
    core = EquiformerV2Core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        attn_hidden_channels=16,
        num_heads=2,
        attn_alpha_channels=8,
        attn_value_channels=8,
        ffn_hidden_channels=32,
        lmax=lmax,
        mmax=1,
        num_layers=2,
        edge_channels=16,
    )
    idx_i, idx_j = _complete_graph_edges(N)
    atom_embedding = torch.randn(N, dim_embedding)
    Za = torch.tensor([1, 6, 7, 8, 1, 6])
    rbf = torch.randn(idx_i.shape[0], num_rbf)
    vij = torch.randn(idx_i.shape[0], 3)
    out = core.get_output(
        atom_embedding=atom_embedding,
        Za=Za,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
    )
    assert out["atom_feature"].shape == (N, sphere_channels)
    assert out["atom_sphere_feature"].shape == (N, (lmax + 1) ** 2, sphere_channels)
    assert core.feature_irreps == f"{sphere_channels}x0e"
    assert core.dim_feature_out == sphere_channels


def test_equiformer_v2_simple_readout_resolves_feature_irreps():
    from enerzyme.models.equiformer_v2 import EquiformerV2Core
    from enerzyme.models.layers import SimpleReadout

    core = EquiformerV2Core(
        dim_embedding=16,
        num_rbf=8,
        sphere_channels=16,
        attn_hidden_channels=16,
        num_heads=2,
        attn_alpha_channels=8,
        attn_value_channels=8,
        ffn_hidden_channels=32,
        lmax=2,
        mmax=1,
        num_layers=1,
        edge_channels=16,
    )
    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="dense",
    )
    assert ro.feature_irreps == "16x0e"
    assert ro.dim_feature_in == 16
    out = ro.get_output(torch.randn(4, 16))
    assert out["Ea"].shape == (4,)


def test_equiformer_v2_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("equiformer_v2", verbose=0)
    N = 5
    Ra = torch.randn(N, 3, requires_grad=True)
    Za = torch.tensor([1, 6, 8, 1, 6])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    out = model(
        {
            "Ra": Ra,
            "Za": Za,
            "batch_seg": batch_seg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert "E" in out and "Fa" in out
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (N, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()


def test_equiformer_v2_atom_feature_so3_invariance():
    from enerzyme.models.equiformer_v2 import EquiformerV2Core

    torch.manual_seed(0)
    dtype = torch.float64
    N = 5
    sphere_channels = 8
    dim_embedding = 8
    num_rbf = 8
    core = EquiformerV2Core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        attn_hidden_channels=8,
        num_heads=2,
        attn_alpha_channels=4,
        attn_value_channels=4,
        ffn_hidden_channels=16,
        lmax=2,
        mmax=2,
        num_layers=1,
        edge_channels=8,
    ).to(dtype)
    core.eval()

    idx_i, idx_j = _complete_graph_edges(N)
    atom_embedding = torch.randn(N, dim_embedding, dtype=dtype)
    Za = torch.tensor([1, 6, 8, 1, 6])
    pos = torch.randn(N, 3, dtype=dtype)
    vij = pos[idx_j] - pos[idx_i]
    # Fixed RBF so only geometry rotates
    rbf = torch.randn(idx_i.shape[0], num_rbf, dtype=dtype)

    with torch.no_grad():
        f0 = core.get_output(
            atom_embedding=atom_embedding,
            Za=Za,
            rbf=rbf,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=vij,
        )["atom_feature"]

        R = _random_so3(dtype)
        vij_r = (pos @ R.T)[idx_j] - (pos @ R.T)[idx_i]
        f1 = core.get_output(
            atom_embedding=atom_embedding,
            Za=Za,
            rbf=rbf,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=vij_r,
        )["atom_feature"]

    assert_allclose(
        f0.detach().cpu().numpy(),
        f1.detach().cpu().numpy(),
        atol=1e-4,
        rtol=1e-4,
    )
