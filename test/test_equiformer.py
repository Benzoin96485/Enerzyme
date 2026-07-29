"""Equiformer Core smoke tests: forward shapes, feature readout, SO(3) energy/force."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enerzyme.models.ff import build_model


def _random_so3(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    a = torch.randn(3, 3, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    if torch.linalg.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _complete_graph(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


@pytest.fixture(scope="module")
def equiformer_model():
    return build_model("equiformer", verbose=0)


def test_equiformer_build_and_forward_shapes(equiformer_model):
    torch.manual_seed(0)
    model = equiformer_model
    n = 6
    za = torch.tensor([1, 6, 1, 8, 7, 1], dtype=torch.long)
    ra = torch.randn(n, 3, dtype=torch.float32) * 0.4
    ra.requires_grad_(True)
    idx_i, idx_j = _complete_graph(n)
    batch_seg = torch.zeros(n, dtype=torch.long)
    out = model(
        {
            "Ra": ra,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
            "offsets": None,
        }
    )
    assert "E" in out and out["E"].shape == (1,)
    assert "Fa" in out and out["Fa"].shape == (n, 3)
    assert "Ea" in out and out["Ea"].shape == (n,)
    assert "Qa" in out and out["Qa"].shape == (n,)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    assert model.dim_feature_out == 64


def test_equiformer_core_feature_mode_direct():
    from enerzyme.models.equiformer.core import EquiformerCore
    from enerzyme.models.equiformer.node_embedding_layer import EquiformerNodeEmbedding
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    torch.manual_seed(1)
    n, e = 4, 12
    irreps = "16x0e+8x1e+4x2e"
    feat = "32x0e"
    embed = EquiformerNodeEmbedding(max_Za=16, irreps_node_embedding=irreps)
    rbf_layer = ExpNormalSmearing(num_rbf=8, cutoff_sr=5.0)
    core = EquiformerCore(
        num_rbf=8,
        irreps_node_embedding=irreps,
        irreps_feature=feat,
        irreps_sh="1x0e+1x1e+1x2e",
        irreps_head="8x0e+4x1o+2x2e",
        irreps_mlp_mid=irreps,
        num_layers=1,
        num_heads=2,
        fc_neurons=[32],
        nonlinear_message=True,
        output_mode="feature",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
    )
    za = torch.tensor([1, 6, 8, 1])
    idx_i = torch.randint(0, n, (e,))
    idx_j = torch.randint(0, n, (e,))
    vij = torch.randn(e, 3)
    dij = vij.norm(dim=-1).clamp_min(1e-6)
    rbf = rbf_layer.get_rbf(dij)
    atom_emb = embed.get_atom_embedding(za)
    batch = torch.zeros(n, dtype=torch.long)
    out = core.get_output(
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=rbf,
        atom_embedding=atom_emb,
        batch_seg=batch,
    )
    assert "atom_feature" in out
    assert out["atom_feature"].shape == (n, 32)
    assert torch.isfinite(out["atom_feature"]).all()


def test_equiformer_energy_invariant_force_equivariant():
    torch.manual_seed(2)
    model = build_model("equiformer", verbose=0).double()
    n = 5
    za = torch.tensor([1, 6, 8, 1, 7], dtype=torch.long)
    ra = torch.randn(n, 3, dtype=torch.float64) * 0.35
    idx_i, idx_j = _complete_graph(n)
    batch_seg = torch.zeros(n, dtype=torch.long)
    q = _random_so3(torch.float64)

    ra0 = ra.clone().requires_grad_(True)
    out0 = model(
        {
            "Ra": ra0,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
            "offsets": None,
        }
    )
    e0 = out0["E"].detach()
    f0 = out0["Fa"].detach()

    ra1 = (ra @ q.T).clone().requires_grad_(True)
    out1 = model(
        {
            "Ra": ra1,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
            "offsets": None,
        }
    )
    e1 = out1["E"].detach()
    f1 = out1["Fa"].detach()

    assert_allclose(e0.cpu().numpy(), e1.cpu().numpy(), atol=1e-5, rtol=1e-5)
    f0_rot = f0 @ q.T
    assert_allclose(f0_rot.cpu().numpy(), f1.cpu().numpy(), atol=1e-4, rtol=1e-4)
