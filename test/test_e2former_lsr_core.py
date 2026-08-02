"""E2Former-LSR Core smoke / equivariance / cluster graph tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "e2former_lsr_layers_example.yaml"


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


def _tiny_lsr_core(**kwargs):
    from enerzyme.models.e2former import E2FormerLSRCore

    defaults = dict(
        dim_embedding=16,
        num_rbf=8,
        max_Za=20,
        irreps_node_embedding="16x0e+16x1e+16x2e",
        irreps_head="8x0e+8x1e+8x2e",
        num_layers=1,
        long_layers=1,
        num_attn_heads=2,
        attn_scalar_head=8,
        ffn_hidden_channels=32,
        max_neighbors=16,
        long_max_neighbors=8,
        attn_type="first-order",
        tp_type="QK_alpha",
        alpha_drop=0.0,
        cutoff_lr=15.0,
        fragment_mode="kmeans",
        min_nodes_per_group=2,
    )
    defaults.update(kwargs)
    return E2FormerLSRCore(**defaults)


def _tiny_lsr_layer_params(**core_overrides):
    params = {
        "irreps_node_embedding": "16x0e+16x1e+16x2e",
        "irreps_head": "8x0e+8x1e+8x2e",
        "num_layers": 1,
        "long_layers": 1,
        "num_attn_heads": 2,
        "attn_scalar_head": 8,
        "ffn_hidden_channels": 32,
        "max_neighbors": 16,
        "long_max_neighbors": 8,
        "fragment_mode": "kmeans",
        "min_nodes_per_group": 2,
        "cutoff_lr": 15.0,
    }
    params.update(core_overrides)
    return [
        {"name": "RangeSeparation"},
        {"name": "GaussianSmearing"},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": params},
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


def test_pool_and_atom_fragment_topk_batch_isolation():
    from enerzyme.models.e2former.cluster import (
        build_atom_fragment_topk,
        pool_fragment_irreps,
        resolve_fragments,
    )

    torch.manual_seed(0)
    # Two graphs: 4 + 3 atoms
    ra = torch.randn(7, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    # Precomputed: graph0 → 2 fragments, graph1 → 2 fragments
    local = torch.tensor([0, 0, 1, 1, 0, 1, 1])
    flat, cpos, cbatch, _remapped = resolve_fragments(
        ra, batch, fragment_mode="precomputed", cluster_ids=local
    )
    assert cpos.shape[0] == 4
    assert cbatch.tolist() == [0, 0, 1, 1]
    assert flat.max().item() == 3

    feats = torch.randn(7, 9, 4)
    pooled = pool_fragment_irreps(feats, flat)
    assert pooled.shape == (4, 9, 4)
    # Mean of atoms 0,1 equals cluster 0
    assert_allclose(
        pooled[0].numpy(),
        feats[:2].mean(dim=0).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    graph = build_atom_fragment_topk(
        atom_pos=ra,
        cluster_pos=cpos,
        flat_cluster_ids=flat,
        batch_seg=batch,
        cluster_batch=cbatch,
        radius=100.0,
        max_neighbors=4,
        remove_self_cluster=True,
    )
    # Neighbors must stay within the same graph's cluster index range
    for i in range(4):
        nbrs = graph["f_sparse_idx_expnode"][i][graph["present"][i]]
        assert (nbrs < 2).all()
    for i in range(4, 7):
        nbrs = graph["f_sparse_idx_expnode"][i][graph["present"][i]]
        assert ((nbrs >= 2) & (nbrs < 4)).all()


def test_num_clusters_guards_nonpositive_min_nodes():
    from enerzyme.models.e2former.cluster import (
        _num_clusters_for_graph,
        build_kmeans_fragments,
    )

    # Misconfigured YAML with 0 / negative must not divide by zero.
    assert _num_clusters_for_graph(10, 0) == _num_clusters_for_graph(10, 1)
    assert _num_clusters_for_graph(10, -3) == _num_clusters_for_graph(10, 1)
    assert _num_clusters_for_graph(0, 24) == 0

    torch.manual_seed(0)
    ra = torch.randn(5, 3)
    batch = torch.zeros(5, dtype=torch.long)
    local, centers, cbatch = build_kmeans_fragments(
        ra, batch, min_nodes_per_group=0, random_state=0
    )
    assert local.shape == (5,)
    assert centers.shape[0] >= 1
    assert cbatch.shape[0] == centers.shape[0]


def test_e2former_lsr_single_atom_graph():
    """Single-atom graphs have no short/long neighbors; fuse must still run."""
    torch.manual_seed(0)
    core = _tiny_lsr_core(min_nodes_per_group=0)
    assert core.min_nodes_per_group == 1
    empty_i = torch.zeros(0, dtype=torch.long)
    empty_j = torch.zeros(0, dtype=torch.long)
    out = core.get_output(
        atom_embedding=torch.randn(1, 16),
        Za=torch.tensor([6]),
        Ra=torch.randn(1, 3),
        rbf=torch.zeros(0, 8),
        idx_i_sr=empty_i,
        idx_j_sr=empty_j,
        vij_sr=torch.zeros(0, 3),
    )
    assert out["atom_feature"].shape == (1, 16)
    assert out["atom_sphere_feature"].shape == (1, 9, 16)
    assert torch.isfinite(out["atom_feature"]).all()


def test_e2former_lsr_core_atom_feature_shape():
    torch.manual_seed(0)
    n = 6
    c = 16
    core = _tiny_lsr_core()
    idx_i, idx_j = _complete_graph_edges(n)
    out = core.get_output(
        atom_embedding=torch.randn(n, 16),
        Za=torch.tensor([1, 6, 7, 8, 1, 6]),
        Ra=torch.randn(n, 3),
        rbf=torch.randn(idx_i.shape[0], 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
    )
    assert out["atom_feature"].shape == (n, c)
    assert out["atom_sphere_feature"].shape == (n, 9, c)


def test_e2former_lsr_precomputed_cluster_ids():
    torch.manual_seed(0)
    n = 6
    core = _tiny_lsr_core(fragment_mode="precomputed")
    idx_i, idx_j = _complete_graph_edges(n)
    cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    out = core.get_output(
        atom_embedding=torch.randn(n, 16),
        Za=torch.tensor([1, 6, 7, 8, 1, 6]),
        Ra=torch.randn(n, 3),
        rbf=torch.randn(idx_i.shape[0], 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
        cluster_ids=cluster_ids,
    )
    assert out["atom_feature"].shape == (n, 16)
    assert torch.isfinite(out["atom_sphere_feature"]).all()


def test_e2former_lsr_so2_first_order_with_fewer_fragments():
    """LSR + so2-first-order with N_atoms != N_frag (cluster value path).

    Default LSR tests use first-order TP; V2 so2 tests use equal atom/value
    batch sizes. This combo is the only place the alpha-batch bug surfaces.
    """
    torch.manual_seed(0)
    n = 6
    core = _tiny_lsr_core(
        attn_type="so2-first-order",
        fragment_mode="precomputed",
        tp_type="QK_alpha",
    )
    idx_i, idx_j = _complete_graph_edges(n)
    # 6 atoms → 3 fragments (strictly fewer values than alpha rows).
    cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    out = core.get_output(
        atom_embedding=torch.randn(n, 16),
        Za=torch.tensor([1, 6, 7, 8, 1, 6]),
        Ra=torch.randn(n, 3),
        rbf=torch.randn(idx_i.shape[0], 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
        cluster_ids=cluster_ids,
    )
    assert out["atom_feature"].shape == (n, 16)
    assert out["atom_sphere_feature"].shape == (n, 9, 16)
    assert torch.isfinite(out["atom_feature"]).all()
    assert torch.isfinite(out["atom_sphere_feature"]).all()


def test_resolve_fragments_ignores_absolute_cluster_centers():
    """Regression: absolute BRICS centers must not mix with COM-centered Ra."""
    from enerzyme.models.e2former.cluster import resolve_fragments
    from enerzyme.models.e2former.graph import center_positions_by_batch

    torch.manual_seed(0)
    ra = torch.randn(6, 3) + 50.0  # large COM
    batch = torch.zeros(6, dtype=torch.long)
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    # Absolute means (wrong frame if used with COM-centered atoms)
    abs_centers = torch.stack(
        [ra[ids == g].mean(dim=0) for g in (0, 1, 2)], dim=0
    )
    ra_c = center_positions_by_batch(ra, batch)
    _, cpos, _, _ = resolve_fragments(
        ra_c,
        batch,
        fragment_mode="precomputed",
        cluster_ids=ids,
        cluster_centers=abs_centers,
    )
    expected = torch.stack(
        [ra_c[ids == g].mean(dim=0) for g in (0, 1, 2)], dim=0
    )
    assert_allclose(cpos.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
    # Absolute centers would be ~50 away from COM-centered means
    assert float((cpos - abs_centers).norm()) > 1.0


def test_e2former_lsr_absolute_centers_input_translation_invariant():
    """Providing absolute cluster_centers must not break E(R)=E(R+t)."""
    from enerzyme.models.ff import build_model

    torch.manual_seed(3)
    layers = _tiny_lsr_layer_params(fragment_mode="precomputed")
    layers = [layer for layer in layers if layer.get("name") != "Force"]
    model = build_model(
        "e2former_lsr",
        layer_params=layers,
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_lr": 15.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    model.eval()
    n = 6
    idx_i, idx_j = _complete_graph_edges(n)
    ra = torch.randn(n, 3)
    za = torch.tensor([1, 6, 8, 1, 7, 6])
    batch = torch.zeros(n, dtype=torch.long)
    cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    abs_centers = torch.stack(
        [ra[cluster_ids == g].mean(dim=0) for g in (0, 1, 2)], dim=0
    )
    shift = torch.tensor([100.0, -70.0, 35.0])

    def _energy(pos, centers):
        with torch.no_grad():
            out = model(
                {
                    "Ra": pos,
                    "Za": za,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                    "batch_seg": batch,
                    "n_atoms": torch.tensor([n]),
                    "cluster_ids": cluster_ids,
                    "cluster_centers": centers,
                }
            )
        return out["E"].sum().detach()

    e0 = _energy(ra, abs_centers)
    e1 = _energy(ra + shift, abs_centers + shift)
    assert_allclose(e0.numpy(), e1.numpy(), rtol=1e-5, atol=1e-5)


def test_empty_long_neighbors_still_runs_fuse():
    """Regression: all-masked long graph must still apply FFN + late fuse."""
    torch.manual_seed(0)
    n = 4
    # One fragment + tiny cutoff ⇒ remove_self leaves no long neighbors.
    core = _tiny_lsr_core(
        fragment_mode="precomputed",
        cutoff_lr=1e-8,
        long_layers=1,
        min_nodes_per_group=100,
    )
    called = []
    core.final_linear.register_forward_hook(lambda *_a, **_k: called.append(True))
    idx_i, idx_j = _complete_graph_edges(n)
    out = core.get_output(
        atom_embedding=torch.randn(n, 16),
        Za=torch.tensor([1, 6, 8, 1]),
        Ra=torch.randn(n, 3),
        rbf=torch.randn(idx_i.shape[0], 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
        cluster_ids=torch.zeros(n, dtype=torch.long),
    )
    assert called, "final_linear fuse was skipped on empty long neighborhood"
    assert out["atom_sphere_feature"].shape == (n, 9, 16)
    assert torch.isfinite(out["atom_sphere_feature"]).all()


def test_e2former_lsr_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model(
        "e2former_lsr",
        layer_params=_tiny_lsr_layer_params(),
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_lr": 15.0,
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
    e = out["E"].sum()
    assert torch.isfinite(e)
    e.backward()
    assert torch.isfinite(ra.grad).all()
    assert torch.isfinite(out["Fa"]).all()


def test_e2former_lsr_energy_invariance_and_force_equivariance():
    from enerzyme.models.ff import build_model

    torch.manual_seed(1)
    model = build_model(
        "e2former_lsr",
        layer_params=_tiny_lsr_layer_params(),
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_lr": 15.0,
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
    assert_allclose(f1.numpy(), (f0 @ r.T).numpy(), rtol=5e-3, atol=5e-3)


def test_e2former_lsr_energy_translation_invariance():
    from enerzyme.models.ff import build_model

    torch.manual_seed(2)
    layers = _tiny_lsr_layer_params()
    # Force needs autograd through Ra; drop it for a pure energy invariance check.
    layers = [layer for layer in layers if layer.get("name") != "Force"]
    model = build_model(
        "e2former_lsr",
        layer_params=layers,
        build_params={
            "dim_embedding": 16,
            "num_rbf": 8,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_lr": 15.0,
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


def test_e2former_lsr_example_yaml_loads():
    import yaml
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
    assert model is not None
