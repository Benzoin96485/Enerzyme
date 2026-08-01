"""TACE Core smoke tests (spherical + Cartesian)."""

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPH = ROOT / "enerzyme" / "config" / "tace_layers_example.yaml"
EXAMPLE_CART = ROOT / "enerzyme" / "config" / "tace_cartesian_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.long),
        torch.tensor([p[1] for p in pairs], dtype=torch.long),
    )


def _tiny_core(tensor_basis: str, **kwargs):
    from enerzyme.models.tace import TACECore

    params = dict(
        max_Za=10,
        dim_embedding=8,
        num_rbf=4,
        num_channel=8,
        num_layers=2,
        Lmax=1,
        lmax=2,
        correlation=2,
        avg_num_neighbors=4.0,
        tensor_basis=tensor_basis,
        edge_embedding="identity",
        edge_update="identity",
        radial_mlp=[16],
        use_first_resnet=False,
    )
    if tensor_basis == "spherical":
        params["nonlinear"] = "sigmoid_gate"
        params["parity"] = False
    params.update(kwargs)
    return TACECore(**params)


def test_tace_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("tace")
    assert core_cls.__name__ == "TACECore"
    for path in (EXAMPLE_SPH, EXAMPLE_CART):
        with path.open() as stream:
            ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
        # shrink for CI
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
                        "correlation": 2,
                        "radial_mlp": [16],
                        "edge_embedding": "identity",
                        "edge_update": "identity",
                    }
                )
                layer["params"] = params
            layers.append(layer)
        bp = dict(ff["build_params"])
        bp.update({"dim_embedding": 8, "num_rbf": 4, "max_Za": 20})
        model = build_model(
            ff["architecture"], layer_params=layers, build_params=bp, verbose=0
        )
        assert model.__class__.__name__ == "TACECore"
        assert model.dim_feature_out == 8


def test_tace_spherical_feature_shape_and_grad():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _tiny_core("spherical")
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=positions[idx_i] - positions[idx_j],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    feat = out["atom_feature"]
    assert feat.ndim == 2 and feat.shape[0] == n
    assert core.dim_feature_out == 8
    assert "0e" in core.feature_irreps
    grad = torch.autograd.grad(feat.square().sum(), positions)[0]
    assert torch.isfinite(grad).all()


def test_tace_cartesian_feature_shape_and_grad():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _tiny_core("cartesian")
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=positions[idx_i] - positions[idx_j],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    feat = out["atom_feature"]
    assert feat.shape == (n, 8)
    assert core.feature_irreps == "8x0e"
    grad = torch.autograd.grad(feat.square().sum(), positions)[0]
    assert torch.isfinite(grad).all()


def test_tace_layer_stack_energy_and_force_spherical():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "tensor_basis": "spherical",
                "num_layers": 1,
                "num_channel": 8,
                "Lmax": 1,
                "lmax": 1,
                "correlation": 2,
                "avg_num_neighbors": 4.0,
                "edge_embedding": "identity",
                "edge_update": "identity",
                "nonlinear": "sigmoid_gate",
                "radial_mlp": [16],
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
        "tace",
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


def test_tace_layer_stack_energy_and_force_cartesian():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "tensor_basis": "cartesian",
                "num_layers": 1,
                "num_channel": 8,
                "Lmax": 1,
                "lmax": 1,
                "correlation": 2,
                "avg_num_neighbors": 4.0,
                "edge_embedding": "identity",
                "edge_update": "identity",
                "radial_mlp": [16],
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
        "tace",
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
