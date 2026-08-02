"""TECE Core smoke tests (registration, YAML, features, energy/forces)."""

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "tece_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.long),
        torch.tensor([p[1] for p in pairs], dtype=torch.long),
    )


def _tiny_core(**kwargs):
    from enerzyme.models.tece import TECECore

    params = dict(
        max_Za=10,
        dim_embedding=8,
        num_rbf=4,
        num_channel=8,
        num_layers=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        correlation=2,
        avg_num_neighbors=4.0,
        edge_embedding="identity",
        edge_update="identity",
        radial_mlp=[16],
        use_first_resnet=False,
        use_so2_edge_ace=True,
        use_graph_softmax=True,
        use_radial_phase=True,
        use_temperature=True,
        num_head=1,
        so2_linear_type="w1",
        gate_m0=False,
    )
    params.update(kwargs)
    return TECECore(**params)


def test_tece_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("tece")
    assert core_cls.__name__ == "TECECore"
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    layers = []
    for layer in ff["layers"]:
        layer = dict(layer)
        if layer.get("name") == "Core":
            params = dict(layer.get("params") or {})
            params.update(
                {
                    "num_channel": 8,
                    "num_layers": 1,
                    "Lmax": 1,
                    "lmax": 1,
                    "mmax": 1,
                    "correlation": 2,
                    "radial_mlp": [16],
                    "edge_embedding": "identity",
                    "edge_update": "identity",
                    "num_head": 1,
                }
            )
            layer["params"] = params
        layers.append(layer)
    bp = dict(ff["build_params"])
    bp.update({"dim_embedding": 8, "num_rbf": 4, "max_Za": 20})
    model = build_model(
        ff["architecture"], layer_params=layers, build_params=bp, verbose=0
    )
    assert model.__class__.__name__ == "TECECore"
    assert model.dim_feature_out == 8


def test_tece_feature_shape_and_grad():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _tiny_core()
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=positions[idx_i] - positions[idx_j],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
        cutoff_values_sr=torch.ones(idx_i.numel(), 1),
    )
    feat = out["atom_feature"]
    assert feat.ndim == 2 and feat.shape[0] == n
    assert core.dim_feature_out == 8
    feat.sum().backward()
    assert positions.grad is not None and positions.grad.abs().sum() > 0


def test_tece_requires_lmax_equals_Lmax():
    import pytest

    with pytest.raises(ValueError, match="Lmax == lmax"):
        _tiny_core(Lmax=2, lmax=1)


def test_tece_layer_stack_energy_and_force():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_layers": 1,
                "num_channel": 8,
                "Lmax": 1,
                "lmax": 1,
                "mmax": 1,
                "correlation": 2,
                "avg_num_neighbors": 4.0,
                "edge_embedding": "identity",
                "edge_update": "identity",
                "radial_mlp": [16],
                "use_so2_edge_ace": True,
                "use_graph_softmax": True,
                "use_radial_phase": True,
                "num_head": 1,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "keep_feature": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "tece",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 6.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    out = model(
        {
            "Ra": positions,
            "Za": torch.tensor([1, 6, 8, 1]),
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    out["E"].sum().backward()
    assert positions.grad is not None and positions.grad.abs().sum() > 0


def test_tece_ece_and_rra_flags_change_outputs():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    kwargs = dict(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
        cutoff_values_sr=torch.ones(idx_i.numel(), 1),
    )
    torch.manual_seed(1)
    a = _tiny_core(use_so2_edge_ace=True, use_graph_softmax=True, use_radial_phase=True)
    torch.manual_seed(1)
    b = _tiny_core(
        use_so2_edge_ace=False, use_graph_softmax=False, use_radial_phase=False
    )
    fa = a.get_output(**kwargs)["atom_feature"]
    fb = b.get_output(**kwargs)["atom_feature"]
    assert not torch.allclose(fa, fb, atol=1e-5)
