"""Unit tests for Euclidean Fast Attention (ERoPE, attention, hooks, Core wiring)."""

from __future__ import annotations

import math
import sys

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])


def _complete_graph_edges(num_nodes: int):
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def test_lebedev_and_frequency_lookup():
    from enerzyme.models.efa import (
        LEBEDEV_FREQUENCY_LOOKUP,
        lebedev_quadrature,
        recommend_max_frequency,
    )

    pts, w = lebedev_quadrature(50)
    assert pts.shape == (50, 3)
    assert w.shape == (50,)
    assert_allclose(float(w.sum()), 1.0, atol=1e-10)
    assert recommend_max_frequency(146) == LEBEDEV_FREQUENCY_LOOKUP[146]


def test_eropes_even_dim_and_shapes():
    from enerzyme.models.efa import (
        apply_rotary_position_embedding,
        calculate_rotary_position_embedding,
        frequency_init,
    )

    theta = frequency_init(8, max_frequency=math.pi, max_length=10.0)
    assert theta.shape == (4,)
    x_proj = torch.randn(5, 50)
    sin, cos = calculate_rotary_position_embedding(x_proj, theta)
    assert sin.shape == (5, 50, 8)
    feats = torch.randn(5, 8)
    out = apply_rotary_position_embedding(feats, sin, cos)
    assert out.shape == (5, 50, 8)


def test_efa_batch_isolation():
    from enerzyme.models.efa import EuclideanFastAttention

    torch.manual_seed(0)
    efa = EuclideanFastAttention(
        16,
        num_features_qk=8,
        num_features_v=8,
        lebedev_num=50,
        max_frequency=math.pi,
        max_length=10.0,
    )
    # Two molecules of 3 atoms. Changing the *relative* geometry of molecule 1
    # must not affect molecule 0 (EFA is translation-invariant, so a pure
    # shift of molecule 1 is not a valid isolation probe).
    pos = torch.randn(6, 3)
    seg = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    x = torch.randn(6, 16)
    y1 = efa(x, pos, seg)
    pos2 = pos.clone()
    pos2[3:] = pos2[3:] * 1.7 + torch.tensor([0.0, 0.5, -0.3])
    y2 = efa(x, pos2, seg)
    assert torch.allclose(y1[:3], y2[:3], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y1[3:], y2[3:], atol=1e-3)


def test_efa_translation_invariance_of_kernel_path():
    """Integrated EFA on scalars should be translation invariant (relative geometry)."""
    from enerzyme.models.efa import EuclideanFastAttention

    torch.manual_seed(1)
    efa = EuclideanFastAttention(
        8,
        num_features_qk=8,
        num_features_v=8,
        lebedev_num=50,
        max_frequency=math.pi,
        max_length=10.0,
    )
    efa.eval()
    x = torch.randn(4, 8)
    pos = torch.randn(4, 3)
    seg = torch.zeros(4, dtype=torch.long)
    y0 = efa(x, pos, seg)
    y1 = efa(x, pos + 3.7, seg)
    assert torch.allclose(y0, y1, atol=1e-4, rtol=1e-4)


def test_efa_block_zero_at_init_and_force_grad():
    from enerzyme.models.efa import EFABlock

    torch.manual_seed(0)
    blk = EFABlock(
        12,
        num_features_qk=8,
        num_features_v=8,
        lebedev_num=50,
        max_frequency=math.pi,
        max_length=10.0,
        as_delta=True,
    )
    x = torch.randn(5, 12)
    pos = torch.randn(5, 3, requires_grad=True)
    seg = torch.zeros(5, dtype=torch.long)
    delta = blk(x, pos, seg)
    assert delta.shape == (5, 12)
    assert float(delta.detach().abs().max()) < 1e-6
    # Non-zero last layer so gradients flow for the smoke check
    torch.nn.init.normal_(blk.mlp_2.weight, std=0.01)
    delta = blk(x, pos, seg)
    delta.sum().backward()
    assert pos.grad is not None
    assert pos.grad.abs().sum() > 0


def test_apply_efa_if_configured_hook():
    from enerzyme.models.efa import EFABlock, apply_efa_if_configured, build_efa_blocks

    blocks = build_efa_blocks(
        3,
        8,
        era_use_in_iterations=[1],
        num_features_qk=8,
        num_features_v=8,
        lebedev_num=50,
        max_frequency=math.pi,
        max_length=5.0,
    )
    x = torch.randn(4, 8)
    pos = torch.randn(4, 3)
    # inactive layers -> zeros
    z0 = apply_efa_if_configured(x, pos, None, blocks[0])
    assert torch.equal(z0, torch.zeros_like(x))
    # active layer is an EFABlock
    assert isinstance(blocks[1], EFABlock)
    z1 = apply_efa_if_configured(x, pos, None, blocks[1])
    assert z1.shape == x.shape


def test_so3krates_with_efa_get_output():
    from enerzyme.models.so3krates import So3kratesCore

    torch.manual_seed(0)
    N, F = 5, 12
    core = So3kratesCore(
        dim_embedding=F,
        num_rbf=8,
        degrees=[1, 2, 3],
        num_features=F,
        num_heads=3,
        num_layers=2,
        avg_num_neighbors=4.0,
        era_use_in_iterations=[0, 1],
        era_lebedev_num=50,
        era_max_frequency=math.pi,
        era_max_length=10.0,
        era_qk_num_features=8,
        era_v_num_features=8,
    )
    idx_i, idx_j = _complete_graph_edges(N)
    P = idx_i.shape[0]
    out = core.get_output(
        atom_embedding=torch.randn(N, F),
        rbf=torch.randn(P, 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(P, 3),
        Dij_sr=torch.rand(P) * 4 + 0.1,
        Ra=torch.randn(N, 3),
        batch_seg=torch.zeros(N, dtype=torch.long),
    )
    assert out["atom_feature"].shape == (N, F)


def test_build_model_efa_energy_force():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    layers = [
        {"name": "RangeSeparation"},
        {"name": "BernsteinRBF", "params": {"cutoff_fn": "cosine"}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "degrees": [1, 2, 3],
                "num_features": 12,
                "num_heads": 3,
                "num_layers": 2,
                "avg_num_neighbors": 4.0,
                "era_use_in_iterations": [0],
                "era_lebedev_num": 50,
                "era_max_frequency": math.pi,
                "era_max_length": 10.0,
                "era_qk_num_features": 8,
                "era_v_num_features": 8,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {"output_fields": ["Ea"], "head_type": "dense", "keep_feature": False},
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "efa",
        layer_params=layers,
        build_params={
            "dim_embedding": 12,
            "num_rbf": 8,
            "max_Za": 94,
            "cutoff_sr": 5.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    )
    N = 4
    Ra = torch.randn(N, 3, requires_grad=True)
    net_in = {
        "Ra": Ra,
        "Za": torch.tensor([1, 6, 1, 8], dtype=torch.long),
        "batch_seg": torch.zeros(N, dtype=torch.long),
        "idx_i": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long),
        "idx_j": torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=torch.long),
    }
    # denser complete graph
    idx_i, idx_j = _complete_graph_edges(N)
    net_in["idx_i"] = idx_i
    net_in["idx_j"] = idx_j
    out = model(net_in)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    out["E"].sum().backward()
    assert Ra.grad is not None


def test_build_model_so3lr_efa_smoke():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    # Shrink Core for speed while keeping SO3LR post-core stack from defaults.
    from enerzyme.models.so3krates.so3lr_efa import DEFAULT_LAYER_PARAMS

    layers = []
    for item in DEFAULT_LAYER_PARAMS:
        item = dict(item)
        if item["name"] == "Core":
            item["params"] = dict(item.get("params", {}))
            item["params"].update(
                {
                    "degrees": [1, 2],
                    "num_features": 8,
                    "num_heads": 2,
                    "num_layers": 2,
                    "avg_num_neighbors": 4.0,
                    "era_use_in_iterations": [0],
                    "era_lebedev_num": 50,
                    "era_max_frequency": math.pi,
                    "era_max_length": 10.0,
                    "era_qk_num_features": 8,
                    "era_v_num_features": 8,
                }
            )
        layers.append(item)
    model = build_model(
        "so3lr_efa",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 8,
            "max_Za": 118,
            "cutoff_sr": 4.5,
            "cutoff_fn": "phys",
            "cutoff_lr": 12.0,
            "Bohr_in_R": 0.5291772105638411,
            "Hartree_in_E": 27.211386245988,
        },
        verbose=0,
    )
    N = 4
    Ra = torch.randn(N, 3, requires_grad=True)
    idx_i, idx_j = _complete_graph_edges(N)
    out = model(
        {
            "Ra": Ra,
            "Za": torch.tensor([1, 6, 1, 8], dtype=torch.long),
            "Q": torch.zeros(1),
            "S": torch.ones(1),
            "batch_seg": torch.zeros(N, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert torch.isfinite(out["E"]).all()
    assert "Fa" in out


def test_spookynet_use_efa_smoke():
    from enerzyme.models.spookynet.core import SpookyNetCore

    torch.manual_seed(0)
    F = 16
    core = SpookyNetCore(
        dim_embedding=F,
        num_rbf=8,
        num_modules=1,
        num_residual_pre=1,
        num_residual_local_x=1,
        num_residual_local_s=1,
        num_residual_local_p=1,
        num_residual_local_d=1,
        num_residual_local=1,
        num_residual_nonlocal_q=1,
        num_residual_nonlocal_k=1,
        num_residual_nonlocal_v=1,
        num_residual_post=1,
        num_residual_output=1,
        activation_fn="swish",
        use_irreps=True,
        use_efa=True,
        efa_lebedev_num=50,
        efa_max_frequency=math.pi,
        efa_max_length=10.0,
        efa_num_features_qk=8,
        efa_num_features_v=8,
        output_mode="feature",
    )
    N = 4
    idx_i, idx_j = _complete_graph_edges(N)
    P = idx_i.shape[0]
    Dij = torch.rand(P) * 3 + 0.5
    vij = torch.randn(P, 3)
    vij = vij / vij.norm(dim=-1, keepdim=True) * Dij.unsqueeze(-1)
    out = core.get_output(
        Dij_sr=Dij,
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(P, 8),
        atom_embedding=torch.randn(N, F),
        batch_seg=torch.zeros(N, dtype=torch.long),
        Ra=torch.randn(N, 3),
    )
    assert out["atom_feature"].shape == (N, F)
