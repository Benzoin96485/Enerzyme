"""EquiformerV2 Core smoke / equivariance / build_model / public-switch tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

ROOT = Path(__file__).resolve().parents[1]
FFN_EXAMPLE = ROOT / "enerzyme" / "config" / "equiformer_v2_ffn_readout_example.yaml"
ENSEMBLE_EXAMPLE = ROOT / "enerzyme" / "config" / "equiformer_v2_shallow_ensemble_example.yaml"


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
    from enerzyme.models.equiformer_v2 import EquiformerV2Core

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
    )
    defaults.update(kwargs)
    return EquiformerV2Core(**defaults)


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
    from enerzyme.models.layers import SimpleReadout

    core = _tiny_core(dim_embedding=16, sphere_channels=16, attn_hidden_channels=16,
                      attn_alpha_channels=8, attn_value_channels=8,
                      ffn_hidden_channels=32, edge_channels=16)
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


def test_equiformer_v2_ffn_readout_keep_feature_false():
    from enerzyme.models.equiformer_v2.interaction import EquiformerV2FeedForwardReadout

    torch.manual_seed(0)
    core = _tiny_core()
    ro = EquiformerV2FeedForwardReadout(
        output_fields={"Ea"},
        built_layers=[core],
        keep_feature=False,
    )
    N = 3
    sphere = torch.randn(N, 9, 8)
    out = ro.get_output(sphere)
    assert out["Ea"].shape == (N,)
    assert torch.isfinite(out["Ea"]).all()
    assert "atom_feature" not in out
    assert "atom_sphere_feature" not in out
    assert set(ro._output_fields) == {"Ea"}


def test_equiformer_v2_ffn_readout_keep_feature_true():
    from enerzyme.models.equiformer_v2.interaction import EquiformerV2FeedForwardReadout

    torch.manual_seed(0)
    core = _tiny_core()
    ro = EquiformerV2FeedForwardReadout(
        output_fields={"Ea"},
        built_layers=[core],
        keep_feature=True,
    )
    N = 3
    sphere = torch.randn(N, 9, 8)
    out = ro.get_output(sphere)
    assert out["Ea"].shape == (N,)
    assert out["atom_sphere_feature"].shape == sphere.shape
    assert out["atom_feature"].shape == (N, 8)
    assert torch.allclose(out["atom_sphere_feature"], sphere)
    assert torch.allclose(out["atom_feature"], sphere[:, 0, :])
    assert {"Ea", "atom_feature", "atom_sphere_feature"} <= set(ro._output_fields)


def test_equiformer_v2_ffn_readout_shallow_ensemble():
    from enerzyme.models.equiformer_v2.interaction import EquiformerV2FeedForwardReadout

    torch.manual_seed(0)
    core = _tiny_core()
    ensemble = 3
    ro = EquiformerV2FeedForwardReadout(
        output_fields={"Ea"},
        built_layers=[core],
        shallow_ensemble_size=ensemble,
    )
    N = 3
    out = ro.get_output(torch.randn(N, 9, 8))
    assert out["Ea"].shape == (N, ensemble)
    assert torch.isfinite(out["Ea"]).all()


def test_equiformer_v2_core_public_switches_forward():
    """Public Core flags must construct and run (gate / grid MLP / atom-edge)."""
    torch.manual_seed(0)
    N = 4
    idx_i, idx_j = _complete_graph_edges(N)
    Za = torch.tensor([1, 6, 8, 1])
    atom_embedding = torch.randn(N, 8)
    rbf = torch.randn(idx_i.shape[0], 8)
    vij = torch.randn(idx_i.shape[0], 3)

    switch_sets = [
        dict(use_gate_act=True, use_grid_mlp=False, use_atom_edge_embedding=True),
        dict(use_gate_act=False, use_grid_mlp=True, use_atom_edge_embedding=True),
        dict(use_gate_act=False, use_grid_mlp=False, use_atom_edge_embedding=False),
    ]
    for switches in switch_sets:
        core = _tiny_core(**switches)
        core.eval()
        out = core.get_output(
            atom_embedding=atom_embedding,
            Za=Za,
            rbf=rbf,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=vij,
        )
        assert out["atom_feature"].shape == (N, 8)
        assert out["atom_sphere_feature"].shape == (N, 9, 8)
        assert torch.isfinite(out["atom_feature"]).all()
        assert torch.isfinite(out["atom_sphere_feature"]).all()


def test_equiformer_v2_ffn_example_yaml_build_model_smoke():
    from enerzyme.models.ff import build_model

    assert FFN_EXAMPLE.is_file()
    with open(FFN_EXAMPLE) as f:
        cfg = yaml.safe_load(f)
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    assert ff["architecture"] == "equiformer_v2"
    assert any(layer["name"] == "EquiformerV2FeedForwardReadout" for layer in ff["layers"])

    torch.manual_seed(0)
    model = build_model(
        architecture=ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
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


def test_equiformer_v2_shallow_ensemble_yaml_build_model_smoke():
    from enerzyme.models.ff import build_model

    assert ENSEMBLE_EXAMPLE.is_file()
    with open(ENSEMBLE_EXAMPLE) as f:
        cfg = yaml.safe_load(f)
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    assert ff["architecture"] == "equiformer_v2"
    assert any(
        layer["name"] == "SimpleReadout"
        and layer.get("params", {}).get("shallow_ensemble_size", 1) > 1
        for layer in ff["layers"]
    )

    torch.manual_seed(0)
    model = build_model(
        architecture=ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    N = 5
    ensemble = 4
    Ra = torch.randn(N, 3, requires_grad=True)
    Za = torch.tensor([1, 6, 8, 1, 6])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    batch = {
        "Ra": Ra,
        "Za": Za,
        "batch_seg": batch_seg,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    model.train()
    out_train = model(batch)
    assert out_train["E"].shape == (1, ensemble)
    assert out_train["Fa"].shape == (N, 3, ensemble)
    assert "E_var" in out_train
    assert out_train["E_var"].shape == (1,)
    assert torch.isfinite(out_train["E"]).all()
    assert torch.isfinite(out_train["Fa"]).all()
    assert torch.isfinite(out_train["E_var"]).all()

    model.eval()
    Ra2 = torch.randn(N, 3, requires_grad=True)
    batch["Ra"] = Ra2
    out_eval = model(batch)
    assert out_eval["E"].shape == (1, ensemble)
    assert "E_var" in out_eval and "Fa_var" in out_eval
    assert out_eval["E_var"].shape == (1,)
    assert out_eval["Fa_var"].shape == (N, 3)
    assert torch.isfinite(out_eval["E_var"]).all()
    assert torch.isfinite(out_eval["Fa_var"]).all()


def test_equiformer_v2_atom_feature_rotation_invariant():
    torch.manual_seed(0)
    dtype = torch.float64
    N = 5
    sphere_channels = 8
    dim_embedding = 8
    num_rbf = 8
    core = _tiny_core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        mmax=2,
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
