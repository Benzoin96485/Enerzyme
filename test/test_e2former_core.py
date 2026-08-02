"""E2Former Core smoke / equivariance / build_model tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "e2former_layers_example.yaml"


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
        attn_type="first-order",
        tp_type="QK_alpha",
        alpha_drop=0.0,
    )
    defaults.update(kwargs)
    return E2FormerCore(**defaults)


def test_e2former_core_atom_feature_shape():
    torch.manual_seed(0)
    n = 6
    dim_embedding = 16
    num_rbf = 8
    lmax = 2
    c = 16
    core = _tiny_core(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        irreps_node_embedding=f"{c}x0e+{c}x1e+{c}x2e",
    )
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


def test_e2former_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("e2former", verbose=0)
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


def test_e2former_energy_invariance_and_force_equivariance():
    from enerzyme.models.ff import build_model

    torch.manual_seed(1)
    model = build_model(
        "e2former",
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
    assert_allclose(e0.numpy(), e1.numpy(), rtol=1e-3, atol=1e-4)
    # Forces transform as vectors: F' ≈ F R^T
    assert_allclose(f1.numpy(), (f0 @ r.T).numpy(), rtol=5e-3, atol=5e-3)


def test_e2former_energy_translation_invariance():
    """Wigner-6j uses absolute solid harmonics; COM-centering must keep E(R+t)=E(R)."""
    from enerzyme.models.ff import build_model

    torch.manual_seed(2)
    model = build_model(
        "e2former",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing"},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "irreps_node_embedding": "16x0e+16x1e+16x2e",
                    "irreps_head": "8x0e+8x1e+8x2e",
                    "num_layers": 2,
                    "num_attn_heads": 2,
                    "attn_scalar_head": 8,
                    "ffn_hidden_channels": 32,
                    "max_neighbors": 16,
                    "attn_type": "first-order",
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
    n = 5
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3)
    za = torch.tensor([1, 6, 8, 1, 7])
    batch = torch.zeros(n, dtype=torch.long)
    shift = torch.tensor([100.0, -70.0, 35.0])

    def _energy(pos):
        with torch.no_grad():
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
        return out["E"].sum().detach()

    e0 = _energy(ra)
    e1 = _energy(ra + shift)
    assert_allclose(e0.numpy(), e1.numpy(), rtol=1e-5, atol=1e-5)


def test_center_positions_by_batch_is_shift_invariant():
    from enerzyme.models.e2former.graph import center_positions_by_batch

    torch.manual_seed(0)
    ra = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    shift = torch.tensor([50.0, -20.0, 10.0])
    c0 = center_positions_by_batch(ra, batch)
    c1 = center_positions_by_batch(ra + shift, batch)
    # Large |t| makes float32 COM subtraction slightly noisy.
    assert_allclose(c0.numpy(), c1.numpy(), rtol=1e-4, atol=1e-5)
    # Each graph COM near origin
    for g in (0, 1):
        assert float(c0[batch == g].mean(dim=0).abs().max()) < 1e-5


def test_select_closest_neighbors_truncates_by_distance():
    """Regression: degree > max_neighbors must keep nearest edges, not crash.

    Earlier CI only used N<=6 complete graphs with max_neighbors>=16, so the
    hard ValueError path was never exercised.
    """
    from enerzyme.models.e2former.graph import (
        build_topk_neighborhood,
        select_closest_neighbors,
    )

    # Star: center 0 connected to 1..4 at increasing distances
    src = torch.tensor([1, 2, 3, 4])
    dst = torch.tensor([0, 0, 0, 0])
    dist = torch.tensor([4.0, 1.0, 3.0, 2.0])
    vij = torch.stack([dist, torch.zeros(4), torch.zeros(4)], dim=-1)
    rbf = torch.randn(4, 3)
    src_k, dst_k, dist_k, vij_k, rbf_k = select_closest_neighbors(
        src, dst, dist, 2, 5, vij, rbf
    )
    assert src_k.tolist() == [2, 4]  # distances 1.0 then 2.0
    assert dist_k.tolist() == [1.0, 2.0]
    assert vij_k.shape == (2, 3) and rbf_k.shape == (2, 3)

    # Full model path: complete graph N=8 => degree 7 > max_neighbors=3
    n = 8
    k = 3
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3)
    vij = ra[idx_j] - ra[idx_i]
    rbf = torch.randn(idx_i.shape[0], 4)
    neigh = build_topk_neighborhood(
        ra, idx_i, idx_j, vij, rbf, max_neighbors=k
    )
    assert neigh["f_sparse_idx_node"].shape == (n, k)
    assert int(neigh["present"].sum(dim=-1).max().item()) == k
    # Each kept neighbor is among the true k closest
    for i in range(n):
        true_dist = torch.linalg.norm(ra - ra[i], dim=-1)
        true_dist[i] = float("inf")
        true_topk = set(torch.topk(true_dist, k, largest=False).indices.tolist())
        kept = neigh["f_sparse_idx_node"][i][neigh["present"][i]].tolist()
        assert set(kept) == true_topk


def test_e2former_dense_graph_respects_max_neighbors():
    """Default-style Core must not crash when cutoff degree exceeds max_neighbors."""
    from enerzyme.models.ff import build_model

    torch.manual_seed(3)
    n = 10  # complete-graph degree 9 > max_neighbors 4
    model = build_model(
        "e2former",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing"},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "irreps_node_embedding": "8x0e+8x1e+8x2e",
                    "irreps_head": "4x0e+4x1e+4x2e",
                    "num_layers": 1,
                    "num_attn_heads": 2,
                    "attn_scalar_head": 4,
                    "ffn_hidden_channels": 16,
                    "max_neighbors": 4,
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
        ],
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 20.0,  # keep all pairs so degree = N-1
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    model.eval()
    idx_i, idx_j = _complete_graph_edges(n)
    with torch.no_grad():
        out = model(
            {
                "Ra": torch.randn(n, 3) * 0.5,
                "Za": torch.arange(1, n + 1),
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": torch.zeros(n, dtype=torch.long),
                "n_atoms": torch.tensor([n]),
            }
        )
    assert torch.isfinite(out["E"]).all()


def test_e2former_example_yaml_loads():
    import yaml
    from enerzyme.models.ff import build_model

    assert EXAMPLE.is_file()
    cfg = yaml.safe_load(EXAMPLE.read_text())
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    model = build_model(
        ff["architecture"],
        layer_params=ff.get("layers"),
        build_params=ff.get("build_params"),
        verbose=0,
    )
    assert model.dim_feature_out == 64
