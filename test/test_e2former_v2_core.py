"""E2Former-V2 Core smoke / equivariance / build_model tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "e2former_v2_layers_example.yaml"


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


def _tiny_v2_core(**kwargs):
    from enerzyme.models.e2former import E2FormerCore

    defaults = dict(
        dim_embedding=16,
        num_rbf=8,
        max_Za=20,
        irreps_node_embedding="16x0e+16x1e+16x2e",
        irreps_head="8x0e+8x1e+8x2e",
        num_layers=1,
        num_attn_heads=2,
        attn_scalar_head=8,
        ffn_hidden_channels=32,
        max_neighbors=16,
        attn_type="so2-first-order",
        # Force PyTorch path on CPU / login nodes.
        tp_type="QK_alpha",
        alpha_drop=0.0,
    )
    defaults.update(kwargs)
    return E2FormerCore(**defaults)


def test_e2former_v2_core_atom_feature_shape():
    torch.manual_seed(0)
    n = 6
    dim_embedding = 16
    num_rbf = 8
    lmax = 2
    c = 16
    core = _tiny_v2_core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        irreps_node_embedding=f"{c}x0e+{c}x1e+{c}x2e",
    )
    core.eval()
    idx_i, idx_j = _complete_graph_edges(n)
    out = core.get_output(
        atom_embedding=torch.randn(n, dim_embedding),
        Za=torch.tensor([1, 6, 7, 8, 1, 6]),
        Ra=torch.randn(n, 3),
        rbf=torch.randn(idx_i.shape[0], num_rbf),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
    )
    assert out["atom_feature"].shape == (n, c)
    assert out["atom_sphere_feature"].shape == (n, (lmax + 1) ** 2, c)
    assert core.feature_irreps == f"{c}x0e"
    assert core.dim_feature_out == c


def test_e2former_v2_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model(
        "e2former_v2",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing"},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "irreps_node_embedding": "16x0e+16x1e+16x2e",
                    "irreps_head": "8x0e+8x1e+8x2e",
                    "num_layers": 1,
                    "num_attn_heads": 2,
                    "attn_scalar_head": 8,
                    "ffn_hidden_channels": 32,
                    "max_neighbors": 16,
                    "attn_type": "so2-first-order",
                    "tp_type": "QK_alpha",
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
        ],
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    n = 5
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3, requires_grad=True)
    out = model(
        {
            "Ra": ra,
            "Za": torch.tensor([8, 1, 1, 6, 1]),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "n_atoms": torch.tensor([n]),
        }
    )
    assert out["atom_feature"].shape[0] == n
    assert out["atom_sphere_feature"].shape[1] == 9
    e = out["E"].sum()
    assert torch.isfinite(e)
    e.backward()
    assert torch.isfinite(ra.grad).all()
    assert torch.isfinite(out["Fa"]).all()


def test_e2former_v2_energy_invariance_and_force_equivariance():
    from enerzyme.models.ff import build_model

    torch.manual_seed(1)
    model = build_model(
        "e2former_v2",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing"},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "irreps_node_embedding": "16x0e+16x1e+16x2e",
                    "irreps_head": "8x0e+8x1e+8x2e",
                    "num_layers": 1,
                    "num_attn_heads": 2,
                    "attn_scalar_head": 8,
                    "ffn_hidden_channels": 32,
                    "max_neighbors": 16,
                    "attn_type": "so2-first-order",
                    "tp_type": "QK_alpha",
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
        ],
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    model.eval()
    n = 4
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3)
    za = torch.tensor([1, 6, 8, 1])
    batch = torch.zeros(n, dtype=torch.long)
    r = _random_so3(dtype=torch.float32)

    def _run(pos):
        pos = pos.clone().requires_grad_(True)
        out = model(
            {
                "Ra": pos,
                "Za": za,
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": batch,
                "n_atoms": torch.tensor([n]),
            }
        )
        e = out["E"].sum()
        fa = torch.autograd.grad(e, pos, create_graph=False)[0]
        return e.detach(), fa.detach()

    e0, f0 = _run(ra)
    e1, f1 = _run(ra @ r.T)
    assert_allclose(e0.cpu().numpy(), e1.cpu().numpy(), atol=1e-4, rtol=1e-4)
    f0_r = f0 @ r.T
    assert_allclose(f1.cpu().numpy(), f0_r.cpu().numpy(), atol=2e-3, rtol=2e-3)

    # Translation invariance of energy.
    shift = torch.tensor([3.0, -2.0, 1.5])
    e2, _ = _run(ra + shift)
    assert_allclose(e0.cpu().numpy(), e2.cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_e2former_v2_example_yaml_build_model_smoke():
    import yaml
    from enerzyme.models.ff import build_model

    with open(EXAMPLE) as f:
        cfg = yaml.safe_load(f)
    ff = cfg["Modelhub"]["internal_FFs"]["FF01"]
    # Override to tiny + non-Triton for CPU smoke.
    layers = ff["layers"]
    for layer in layers:
        if layer.get("name") == "Core":
            layer["params"].update(
                {
                    "irreps_node_embedding": "16x0e+16x1e+16x2e",
                    "irreps_head": "8x0e+8x1e+8x2e",
                    "num_layers": 1,
                    "num_attn_heads": 2,
                    "attn_scalar_head": 8,
                    "ffn_hidden_channels": 32,
                    "max_neighbors": 16,
                    "tp_type": "QK_alpha",
                }
            )
    model = build_model(
        "e2former_v2",
        layer_params=layers,
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    model.eval()
    n = 4
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3, requires_grad=True)
    out = model(
        {
            "Ra": ra,
            "Za": torch.tensor([1, 6, 8, 1]),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "n_atoms": torch.tensor([n]),
        }
    )
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
