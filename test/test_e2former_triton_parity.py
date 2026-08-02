"""Regression / parity tests for E2Former-V2 Triton sparse path.

CPU CI exercises the PyTorch fallback and index-convention guards.
CUDA machines additionally compare Triton kernels to the PyTorch reference.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])


def _cuda_triton_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    from enerzyme.models.e2former.triton_sparse import triton_kernels_available

    return triton_kernels_available()


def test_euler_singularity_mask_stays_bool_and_includes_pole():
    """Regression: pole mask must stay bool (``|``, not ``+``)."""
    from enerzyme.models.e2former.wigner_otf import init_edge_rot_euler_angles

    # Near-north-pole + a generic vector.
    vec = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.9999999, 1e-8],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    alpha, beta, gamma = init_edge_rot_euler_angles(vec, training=False)
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(beta).all()
    assert torch.isfinite(gamma).all()
    # Pole cases → beta ≈ 0.
    assert float(beta[0].abs()) < 1e-5
    assert float(beta[1].abs()) < 1e-4


def test_sparse_qk_matches_qk_alpha_gather_convention():
    """PyTorch sparse_qk must match QKAlphaModule's gather (f_sparse_idx_node)."""
    from enerzyme.models.e2former.attention import QKAlphaModule
    from enerzyme.models.e2former.triton_sparse import sparse_qk

    torch.manual_seed(0)
    n, k, h, d = 6, 4, 2, 8
    lmax = 2
    c = 16
    irreps = __import__("e3nn").o3.Irreps(f"{c}x0e+{c}x1e+{c}x2e")
    # Deliberately diverge expnode vs node indices — Triton must not use expnode.
    idx_node = torch.randint(0, n, (n, k))
    idx_exp = (idx_node + 1) % n
    node = torch.randn(n, (lmax + 1) ** 2, c)
    edge_dim = 8 + 128 * 2
    x_edge = torch.randn(n, k, edge_dim)
    mod = QKAlphaModule(
        irreps,
        num_attn_heads=h,
        attn_scalar_head=d,
        edge_channel_list=[edge_dim, 32, 32],
        lmax=lmax,
    )
    ref = mod(
        x_edge=x_edge,
        node_irreps_input=node,
        f_sparse_idx_node=idx_node,
    )

    query = mod.query_linear(node).reshape(n, h, -1)
    key = mod.key_linear(node).reshape(n, h, -1)
    gate = mod.fc_easy(x_edge)
    scale = 1.0 / math.sqrt(d)
    # Correct convention (node idx).
    got = mod.alpha_act(
        sparse_qk(query, key, idx_node, gate, scale, use_triton=False)
    )
    assert_allclose(got.detach().numpy(), ref.detach().numpy(), atol=1e-6, rtol=1e-6)

    # Wrong convention (exp idx) must differ when indices differ.
    wrong = mod.alpha_act(
        sparse_qk(query, key, idx_exp, gate, scale, use_triton=False)
    )
    assert not torch.allclose(wrong, ref, atol=1e-5)


def test_want_triton_cpu_fallback_matches_pytorch_sparse_ops():
    """With want_triton but no CUDA, sparse_qk/v must equal explicit use_triton=False."""
    from enerzyme.models.e2former.triton_sparse import (
        sparse_qk,
        sparse_v_agg,
        triton_kernels_available,
    )

    if triton_kernels_available():
        pytest.skip("CUDA Triton available; CPU fallback path not exercised here")

    torch.manual_seed(1)
    n, k, h, d, feat = 5, 3, 2, 8, 6
    query = torch.randn(n, h, d)
    key = torch.randn(n, h, d)
    idx = torch.randint(0, n, (n, k))
    gate = torch.randn(n, k, h)
    # use_triton=None → auto; on CPU must equal forced False.
    a_auto = sparse_qk(query, key, idx, gate, 0.1, use_triton=None)
    a_ref = sparse_qk(query, key, idx, gate, 0.1, use_triton=False)
    assert_allclose(a_auto.numpy(), a_ref.numpy(), atol=0, rtol=0)

    value = torch.randn(n, feat, h)
    alpha = torch.softmax(a_ref, dim=1)
    v_auto = sparse_v_agg(value, alpha, idx, use_triton=None)
    v_ref = sparse_v_agg(value, alpha, idx, use_triton=False)
    assert_allclose(v_auto.numpy(), v_ref.numpy(), atol=0, rtol=0)


@pytest.mark.skipif(not _cuda_triton_ready(), reason="CUDA + Triton required")
def test_triton_sparse_qk_v_match_pytorch_on_cuda():
    from enerzyme.models.e2former.triton_sparse import sparse_qk, sparse_v_agg

    torch.manual_seed(0)
    device = torch.device("cuda")
    n, k, h, d, feat = 8, 5, 4, 16, 12
    query = torch.randn(n, h, d, device=device)
    key = torch.randn(n, h, d, device=device)
    idx = torch.randint(0, n, (n, k), device=device)
    gate = torch.randn(n, k, h, device=device)
    scale = 0.25

    qk_ref = sparse_qk(query, key, idx, gate, scale, use_triton=False)
    qk_tri = sparse_qk(query, key, idx, gate, scale, use_triton=True)
    assert_allclose(
        qk_tri.detach().cpu().numpy(),
        qk_ref.detach().cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )

    value = torch.randn(n, feat, h, device=device)
    alpha = torch.softmax(qk_ref, dim=1)
    v_ref = sparse_v_agg(value, alpha, idx, use_triton=False)
    v_tri = sparse_v_agg(value, alpha, idx, use_triton=True)
    assert_allclose(
        v_tri.detach().cpu().numpy(),
        v_ref.detach().cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.skipif(not _cuda_triton_ready(), reason="CUDA + Triton required")
def test_attention_triton_qk_matches_qk_alpha_module_on_cuda():
    """End-to-end: E2AttentionSparse Triton QK vs QKAlphaModule (same indices)."""
    from e3nn import o3

    from enerzyme.models.e2former.attention import E2AttentionSparse

    torch.manual_seed(2)
    device = torch.device("cuda")
    n, k = 6, 4
    c = 16
    irreps = f"{c}x0e+{c}x1e+{c}x2e"
    attn = E2AttentionSparse(
        irreps_node_input=irreps,
        attn_weight_input_dim=8,
        num_attn_heads=2,
        attn_scalar_head=8,
        irreps_head="8x0e+8x1e+8x2e",
        tp_type="QK_alpha+triton",
        attn_type="zero-order",
    ).to(device)
    attn.eval()

    idx_node = torch.randint(0, n, (n, k), device=device)
    # Poisoned exp indices — must be ignored by Triton QK after the fix.
    idx_exp = (idx_node + 2) % n
    node = torch.randn(n, 9, c, device=device)
    za = torch.randint(1, 10, (n,), device=device)
    edge_dis = torch.rand(n, k, device=device) + 0.5
    edge_vec = torch.randn(n, k, 3, device=device)
    attn_weight = torch.randn(n, k, 8, device=device)
    attn_mask = torch.zeros(n, k, 1, dtype=torch.bool, device=device)
    batched = {
        "f_sparse_idx_node": idx_node,
        "f_sparse_idx_expnode": idx_exp,
        "f_outcell_index": torch.arange(n, device=device),
        "f_exp_node_pos": torch.randn(n, 3, device=device),
    }
    pos = torch.randn(n, 3, device=device)

    # Force Triton path.
    assert attn._use_triton()
    out_tri, _ = attn(
        pos, node, edge_dis, edge_vec, attn_weight, za, attn_mask, batched
    )

    # Reference: same module with Triton disabled.
    attn.want_triton = False
    out_ref, _ = attn(
        pos, node, edge_dis, edge_vec, attn_weight, za, attn_mask, batched
    )
    assert_allclose(
        out_tri.detach().cpu().numpy(),
        out_ref.detach().cpu().numpy(),
        atol=5e-4,
        rtol=5e-4,
    )
