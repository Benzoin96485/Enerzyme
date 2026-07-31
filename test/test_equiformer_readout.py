"""Tests for equivariant SimpleReadout (0e extract + MLP) and GraphAttention readout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enerzyme.models.ff import build_model
from enerzyme.models.irreps_tools import extract_scalar_0e, scalar_0e_dim
from enerzyme.models.layers.readout import (
    SimpleReadout,
    EquiformerGraphAttentionReadout,
)


def test_extract_scalar_0e_identity_without_irreps():
    x = torch.randn(5, 7)
    assert torch.equal(extract_scalar_0e(x, None), x)


def test_extract_scalar_0e_from_mixed_irreps():
    irreps = o3.Irreps("4x0e+2x1e+1x2e")
    n = 3
    # Layout: [0e(4), 1e(2*3=6), 2e(1*5=5)] = 15
    feat = torch.randn(n, irreps.dim)
    scalars = extract_scalar_0e(feat, irreps)
    assert scalars.shape == (n, 4)
    assert torch.equal(scalars, feat[:, :4])
    assert scalar_0e_dim(irreps) == 4


def test_simple_readout_two_layer_without_irreps():
    torch.manual_seed(0)
    n, d = 6, 8
    # Mock prior scalar core
    class _Core:
        dim_feature_out = d

    ro = SimpleReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[_Core()],
        head_type="two_layer",
        activation_fn="swish",
    )
    assert ro.feature_irreps is None
    feat = torch.randn(n, d)
    out = ro.get_output(feat)
    assert out["Ea"].shape == (n,)
    assert out["Qa"].shape == (n,)
    assert torch.isfinite(out["Ea"]).all()


def test_simple_readout_extracts_0e_then_mlp():
    torch.manual_seed(1)
    irreps = o3.Irreps("8x0e+4x1e")
    mul0 = scalar_0e_dim(irreps)

    class _Core:
        dim_feature_out = mul0
        feature_irreps = str(irreps)

    ro = SimpleReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[_Core()],
        head_type="two_layer",
        activation_fn="swish",
    )
    assert ro.feature_irreps == str(irreps)
    assert ro.dim_feature_in == mul0
    feat = torch.randn(5, irreps.dim)
    out = ro.get_output(feat)
    assert out["Ea"].shape == (5,)
    assert out["Qa"].shape == (5,)


def test_equiformer_graph_attention_readout_multi_field():
    torch.manual_seed(2)
    n, e = 4, 10
    irreps = o3.Irreps("16x0e+8x1e+4x2e")
    n_rbf = 8

    class _Core:
        dim_feature_out = scalar_0e_dim(irreps)
        feature_irreps = str(irreps)
        num_rbf = 8
        fc_neurons = [8, 32]

    ro = EquiformerGraphAttentionReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[_Core()],
        irreps_head="8x0e+4x1o+2x2e",
        num_heads=2,
        fc_neurons=[32],
        irreps_sh="1x0e+1x1e+1x2e",
        nonlinear_message=True,
        num_rbf=n_rbf,
    )
    feat = torch.randn(n, irreps.dim)
    idx_i = torch.randint(0, n, (e,))
    idx_j = torch.randint(0, n, (e,))
    vij = torch.randn(e, 3)
    rbf = torch.randn(e, n_rbf)
    batch = torch.zeros(n, dtype=torch.long)
    out = ro.get_output(
        atom_feature=feat,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        rbf=rbf,
        batch_seg=batch,
    )
    assert set(out) == {"Ea", "Qa"}
    assert out["Ea"].shape == (n,)
    assert out["Qa"].shape == (n,)
    assert torch.isfinite(out["Ea"]).all()


def test_equiformer_graph_attention_readout_shallow_ensemble():
    torch.manual_seed(2)
    n, e, ensemble = 4, 10, 3
    irreps = o3.Irreps("16x0e+8x1e+4x2e")
    n_rbf = 8

    class _Core:
        dim_feature_out = scalar_0e_dim(irreps)
        feature_irreps = str(irreps)
        num_rbf = n_rbf

    ro = EquiformerGraphAttentionReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[_Core()],
        irreps_head="8x0e+4x1o+2x2e",
        num_heads=2,
        fc_neurons=[32],
        irreps_sh="1x0e+1x1e+1x2e",
        nonlinear_message=True,
        num_rbf=n_rbf,
        shallow_ensemble_size=ensemble,
    )
    feat = torch.randn(n, irreps.dim)
    idx_i = torch.randint(0, n, (e,))
    idx_j = torch.randint(0, n, (e,))
    vij = torch.randn(e, 3)
    rbf = torch.randn(e, n_rbf)
    batch = torch.zeros(n, dtype=torch.long)
    out = ro.get_output(
        atom_feature=feat,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        rbf=rbf,
        batch_seg=batch,
    )
    assert out["Ea"].shape == (n, ensemble)
    assert out["Qa"].shape == (n, ensemble)
    assert torch.isfinite(out["Ea"]).all()


def test_equiformer_graph_attention_readout_requires_irreps():
    class _Core:
        dim_feature_out = 8

    with pytest.raises(ValueError, match="feature_irreps"):
        EquiformerGraphAttentionReadout(
            output_fields={"Ea"},
            built_layers=[_Core()],
        )


def _complete_graph(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def test_equiformer_build_model_smoke_two_layer():
    model = build_model("equiformer", verbose=0)
    assert getattr(model, "feature_irreps", None) is not None
    assert model.dim_feature_out == scalar_0e_dim(model.irreps_feature)
    assert any(
        isinstance(m, SimpleReadout) and m.head_type == "two_layer"
        for m in model.post_sequence
    )

    n = 5
    za = torch.tensor([1, 6, 1, 8, 1], dtype=torch.long)
    ra = torch.randn(n, 3) * 0.3
    ra.requires_grad_(True)
    idx_i, idx_j = _complete_graph(n)
    out = model(
        {
            "Ra": ra,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "offsets": None,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Ea"].shape == (n,)
    assert out["Qa"].shape == (n,)


def test_equiformer_shallow_ensemble_yaml_build_model_smoke():
    import yaml

    root = Path(__file__).resolve().parents[1]
    example = root / "enerzyme" / "config" / "equiformer_shallow_ensemble_example.yaml"
    assert example.is_file()
    with open(example) as f:
        cfg = yaml.safe_load(f)
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    ensemble = 4
    model = build_model(
        architecture=ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    n = 5
    za = torch.tensor([1, 6, 1, 8, 1], dtype=torch.long)
    ra = torch.randn(n, 3) * 0.3
    ra.requires_grad_(True)
    idx_i, idx_j = _complete_graph(n)
    batch = {
        "Ra": ra,
        "Za": za,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "offsets": None,
    }
    model.train()
    out = model(batch)
    assert out["E"].shape == (1, ensemble)
    assert out["Fa"].shape == (n, 3, ensemble)
    assert "E_var" in out
    assert out["E_var"].shape == (1,)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    assert torch.isfinite(out["E_var"]).all()

    model.eval()
    ra2 = torch.randn(n, 3) * 0.3
    ra2.requires_grad_(True)
    batch["Ra"] = ra2
    out_eval = model(batch)
    assert "E_var" in out_eval and "Fa_var" in out_eval
    assert out_eval["Fa_var"].shape == (n, 3)
    assert torch.isfinite(out_eval["E_var"]).all()
    assert torch.isfinite(out_eval["Fa_var"]).all()


def test_equiformer_mixed_irreps_simple_readout_stack():
    """Full stack: mixed-irreps Core → SimpleReadout 0e extract → two_layer MLP."""
    import copy

    from enerzyme.models.equiformer.core import (
        DEFAULT_BUILD_PARAMS,
        DEFAULT_LAYER_PARAMS,
    )

    irreps_feature = "32x0e+16x1e+8x2e"
    mul0 = scalar_0e_dim(irreps_feature)
    layers = copy.deepcopy(DEFAULT_LAYER_PARAMS)
    for layer in layers:
        if layer["name"] == "Core":
            layer["params"]["irreps_feature"] = irreps_feature
            layer["params"]["num_layers"] = 1
        if layer["name"] == "SimpleReadout":
            layer["params"]["head_type"] = "two_layer"
            layer["params"]["activation_fn"] = "swish"

    model = build_model(
        "equiformer",
        layer_params=layers,
        build_params=dict(DEFAULT_BUILD_PARAMS),
        verbose=0,
    )
    assert model.feature_irreps == str(o3.Irreps(irreps_feature))
    assert model.dim_feature_out == mul0
    assert o3.Irreps(irreps_feature).dim > mul0

    readout = next(m for m in model.post_sequence if isinstance(m, SimpleReadout))
    assert readout.head_type == "two_layer"
    assert readout.feature_irreps == str(o3.Irreps(irreps_feature))
    assert readout.dim_feature_in == mul0

    n = 5
    torch.manual_seed(4)
    za = torch.tensor([1, 6, 1, 8, 1], dtype=torch.long)
    ra = torch.randn(n, 3) * 0.3
    ra.requires_grad_(True)
    idx_i, idx_j = _complete_graph(n)
    out = model(
        {
            "Ra": ra,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "offsets": None,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Ea"].shape == (n,)
    assert out["Qa"].shape == (n,)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    # Autograd through 0e extract + MLP + EnergyReduce + Force
    out["E"].sum().backward()
    assert ra.grad is not None
    assert torch.isfinite(ra.grad).all()


def test_equiformer_core_emits_full_irreps_feature():
    from enerzyme.models.equiformer.core import EquiformerCore
    from enerzyme.models.equiformer.node_embedding_layer import EquiformerNodeEmbedding
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    torch.manual_seed(3)
    n, e = 4, 12
    irreps = "16x0e+8x1e+4x2e"
    feat = "8x0e+4x1e+2x2e"
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
    assert core.feature_irreps == str(o3.Irreps(feat))
    assert core.dim_feature_out == scalar_0e_dim(feat)
    za = torch.tensor([1, 6, 8, 1])
    idx_i = torch.randint(0, n, (e,))
    idx_j = torch.randint(0, n, (e,))
    vij = torch.randn(e, 3)
    dij = vij.norm(dim=-1).clamp_min(1e-6)
    rbf = rbf_layer.get_rbf(dij)
    atom_emb = embed.get_atom_embedding(za)
    out = core.get_output(
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=rbf,
        atom_embedding=atom_emb,
        batch_seg=torch.zeros(n, dtype=torch.long),
    )
    assert out["atom_feature"].shape == (n, o3.Irreps(feat).dim)
