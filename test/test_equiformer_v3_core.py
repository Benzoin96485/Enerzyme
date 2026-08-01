"""EquiformerV3 Core smoke / equivariance / build_model tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "equiformer_v3_layers_example.yaml"


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


def _tiny_core(**kwargs):
    from enerzyme.models.equiformer_v3 import EquiformerV3Core

    defaults = dict(
        dim_embedding=8,
        num_rbf=8,
        sphere_channels=8,
        attn_hidden_channels=8,
        num_heads=2,
        attn_alpha_channels=4,
        attn_value_channels=4,
        ffn_hidden_channels=16,
        lmax=2,
        mmax=1,
        num_layers=1,
        edge_channels=8,
        norm_type="merge_layer_norm",
        use_envelope=True,
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_grid_mlp=True,
        attn_grid_resolution=[8, 8],
        ffn_grid_resolution=[8, 8],
        cutoff_sr=5.0,
    )
    defaults.update(kwargs)
    return EquiformerV3Core(**defaults)


def test_equiformer_v3_core_atom_feature_shape():
    torch.manual_seed(0)
    N = 6
    sphere_channels = 16
    dim_embedding = 16
    num_rbf = 8
    lmax = 2
    core = _tiny_core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        attn_hidden_channels=16,
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
    assert torch.isfinite(out["atom_feature"]).all()


def test_equiformer_v3_simple_readout_resolves_feature_irreps():
    from enerzyme.models.layers import SimpleReadout

    core = _tiny_core(
        dim_embedding=16,
        sphere_channels=16,
        attn_hidden_channels=16,
        attn_alpha_channels=8,
        attn_value_channels=8,
        ffn_hidden_channels=32,
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


def test_equiformer_v3_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("equiformer_v3", verbose=0)
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


def test_equiformer_v3_so3_scalar_invariance():
    torch.manual_seed(0)
    dtype = torch.float64
    core = _tiny_core().to(dtype)
    core.eval()
    N = 4
    idx_i, idx_j = _complete_graph_edges(N)
    atom_embedding = torch.randn(N, 8, dtype=dtype)
    Za = torch.tensor([1, 6, 8, 1])
    rbf = torch.randn(idx_i.shape[0], 8, dtype=dtype)
    vij = torch.randn(idx_i.shape[0], 3, dtype=dtype)
    R = _random_so3(dtype)
    vij_rot = vij @ R.T
    with torch.no_grad():
        out0 = core.get_output(atom_embedding, Za, rbf, idx_i, idx_j, vij)
        out1 = core.get_output(atom_embedding, Za, rbf, idx_i, idx_j, vij_rot)
    assert_allclose(
        out0["atom_feature"].numpy(),
        out1["atom_feature"].numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_equiformer_v3_yaml_example_builds():
    from enerzyme.models.ff import build_model

    with open(EXAMPLE) as f:
        cfg = yaml.safe_load(f)
    ff = cfg["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "EquiformerV3Core"


def test_equiformer_v3_force_finite_difference_conservation():
    """Fa from ForceLayer must match central finite differences of E.

    Catches Wigner / edge-frame detach that still yields finite but wrong Fa.
    """
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("equiformer_v3", verbose=0).double()
    model.eval()
    N = 4
    Ra = (torch.randn(N, 3, dtype=torch.float64) * 0.4).requires_grad_(True)
    Za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    batch = {
        "Ra": Ra,
        "Za": Za,
        "batch_seg": batch_seg,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    fa = out["Fa"].detach()
    assert torch.isfinite(fa).all()
    assert fa.abs().sum() > 0

    eps = 1e-4
    fd = torch.zeros_like(fa)
    base = Ra.detach().clone()
    for i in range(N):
        for d in range(3):
            rp = base.clone()
            rm = base.clone()
            rp[i, d] += eps
            rm[i, d] -= eps
            rp = rp.requires_grad_(True)
            rm = rm.requires_grad_(True)
            ep = model(
                {
                    "Ra": rp,
                    "Za": Za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            em = model(
                {
                    "Ra": rm,
                    "Za": Za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            fd[i, d] = -(ep.detach() - em.detach()).sum() / (2 * eps)

    assert_allclose(fa.cpu().numpy(), fd.cpu().numpy(), atol=2e-3, rtol=2e-3)


def test_equiformer_v3_wigner_stays_in_autograd_graph():
    """SO3RotationFused Wigner must not be detached when use_rotation_mask=False."""
    from enerzyme.models.so3 import SO3RotationFused, init_edge_rot_mat

    torch.manual_seed(0)
    dtype = torch.float64
    vij = torch.randn(3, 3, dtype=dtype, requires_grad=True)
    rot = SO3RotationFused(lmax=2, mmax=1, use_rotation_mask=False)
    edge_rot = init_edge_rot_mat(vij)
    rot.set_wigner(edge_rot)
    assert rot.wigner.requires_grad
    assert rot.wigner_inv.requires_grad
    loss = rot.wigner.sum() + rot.wigner_inv.sum()
    loss.backward()
    assert vij.grad is not None
    assert torch.isfinite(vij.grad).all()


def test_equiformer_v3_drop_path_disabled_when_rate_zero():
    """drop_path_rate=0 must omit GraphDropPath (avoids empty batch_seg.max crash)."""
    core_off = _tiny_core(drop_path_rate=0.0)
    for blk in core_off.blocks:
        assert blk.drop_path is None
    core_on = _tiny_core(drop_path_rate=0.1)
    for blk in core_on.blocks:
        assert blk.drop_path is not None
