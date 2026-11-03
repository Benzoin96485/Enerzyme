"""DimeNet++ Core smoke tests: registration, shapes, invariance, E/F."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "dimenetpp_layers_example.yaml"


def _complete_graph_edges(num_nodes: int):
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _tiny_core(**kwargs):
    from enerzyme.models.dimenet import DimeNetPPCore

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        cutoff_sr=5.0,
        num_blocks=2,
        out_emb_size=6,
        int_emb_size=4,
        basis_emb_size=3,
        num_spherical=3,
        num_before_skip=1,
        num_after_skip=1,
        envelope_exponent=5,
    )
    params.update(kwargs)
    return DimeNetPPCore(**params)


def _core_inputs(n=4, dim_embedding=8, num_rbf=4, requires_grad=False):
    idx_i, idx_j = _complete_graph_edges(n)
    pos = torch.randn(n, 3, requires_grad=requires_grad)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1).clamp(min=1e-8)
    return {
        "atom_embedding": torch.randn(n, dim_embedding),
        "rbf": torch.randn(idx_i.numel(), num_rbf),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "pos": pos,
    }


def test_dimenetpp_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    for name in ("dimenet++", "dimenetpp", "DimeNet++"):
        core_cls, _, _ = get_ff_core(name)
        assert core_cls.__name__ == "DimeNetPPCore"
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    layers = []
    for layer in ff["layers"]:
        layer = dict(layer)
        if layer.get("name") == "Core":
            params = dict(layer.get("params") or {})
            params.update(
                {
                    "num_blocks": 2,
                    "out_emb_size": 6,
                    "int_emb_size": 4,
                    "basis_emb_size": 3,
                    "num_spherical": 3,
                    "num_before_skip": 1,
                    "num_after_skip": 1,
                }
            )
            layer["params"] = params
        layers.append(layer)
    bp = dict(ff["build_params"])
    bp.update({"dim_embedding": 8, "num_rbf": 4, "max_Za": 20})
    model = build_model(
        ff["architecture"], layer_params=layers, build_params=bp, verbose=0
    )
    assert model.__class__.__name__ == "DimeNetPPCore"
    assert model.num_output_blocks == 3
    assert model.dim_feature_out == 6
    assert not hasattr(model.interactions[0], "W")
    assert model.interactions[0].down.bias is None
    assert model.atom_readouts[0].up.bias is None


def test_dimenetpp_feature_shape_hierarchical_and_last():
    torch.manual_seed(0)
    n = 4
    core = _tiny_core()
    inp = _core_inputs(n=n)
    out = core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )
    feat = out["atom_feature"]
    assert feat.shape == (n, 6, 3)
    last = _tiny_core(output_mode="last")
    out_last = last.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )
    assert out_last["atom_feature"].shape == (n, 6)


def test_dimenetpp_rotation_invariance():
    torch.manual_seed(1)
    n = 5
    core = _tiny_core()
    core.eval()
    idx_i, idx_j = _complete_graph_edges(n)
    pos = torch.randn(n, 3)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    pos_r = pos @ q
    atom_embedding = torch.randn(n, 8)
    rbf = torch.randn(idx_i.numel(), 4)

    def _feat(p):
        vij = p[idx_j] - p[idx_i]
        dist = vij.norm(dim=-1).clamp(min=1e-8)
        return core.get_output(
            atom_embedding=atom_embedding,
            rbf=rbf,
            Dij_sr=dist,
            vij_sr=vij,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
        )["atom_feature"]

    f0 = _feat(pos)
    f1 = _feat(pos_r)
    assert torch.allclose(f0, f1, atol=1e-5, rtol=1e-5)


def test_dimenetpp_layer_stack_energy_and_force():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "dimenet", "trainable": False, "envelope_exponent": 5},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_blocks": 2,
                "out_emb_size": 6,
                "int_emb_size": 4,
                "basis_emb_size": 3,
                "num_spherical": 3,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "envelope_exponent": 5,
            },
        },
        {
            "name": "HierachicalReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "mlp",
                "num_hidden_layers": 2,
                "activation_fn": "swish",
                "activation_params": {
                    "dim_feature": 1,
                    "initial_alpha": 1,
                    "initial_beta": 1,
                    "learnable": False,
                },
                "use_bias_out": False,
                "keep_feature": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "dimenet++",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    n = 4
    idx_i, idx_j = _complete_graph_edges(n)
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
    energy = out["E"].sum()
    force_autograd = torch.autograd.grad(energy, positions, retain_graph=True)[0]
    assert_allclose(
        out["Fa"].detach().numpy(),
        (-force_autograd).detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dimenetpp_pair_graph_has_no_triplets():
    """Two atoms have edges and no triplets; forces must still be finite."""
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "dimenet", "trainable": False, "envelope_exponent": 5},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_blocks": 1,
                "out_emb_size": 4,
                "int_emb_size": 3,
                "basis_emb_size": 2,
                "num_spherical": 2,
                "num_before_skip": 1,
                "num_after_skip": 0,
            },
        },
        {
            "name": "HierachicalReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "mlp",
                "num_hidden_layers": 1,
                "use_bias_out": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "dimenetpp",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 5.0,
        },
        verbose=0,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], requires_grad=True)
    out = model(
        {
            "Ra": positions,
            "Za": torch.tensor([1, 8]),
            "batch_seg": torch.zeros(2, dtype=torch.long),
            "idx_i": torch.tensor([0, 1]),
            "idx_j": torch.tensor([1, 0]),
        }
    )
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()


def test_dimenet_defaults_unchanged():
    from enerzyme.models.dimenet import DEFAULT_LAYER_PARAMS, DimeNetCore
    from enerzyme.models.ff import get_ff_core

    core_cls, _, layers = get_ff_core("dimenet")
    assert core_cls is DimeNetCore
    core = next(layer for layer in layers if layer["name"] == "Core")
    assert core["params"]["num_bilinear"] == 8
    assert "out_emb_size" not in core["params"]
    assert layers is DEFAULT_LAYER_PARAMS
