"""E2Former-V2 SO2 / EAAS tensor-product unit tests."""

from __future__ import annotations

import sys

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])


def _random_so3(dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def test_so2_tp_givenl2_shapes():
    from enerzyme.models.e2former.so2_tensor_product import SO2_TP_givenl2

    lmax = 2
    in_c, out_c = 8, 8
    tp = SO2_TP_givenl2(
        range(lmax + 1),
        1,
        range(lmax + 1),
        in_c=in_c,
        out_c=out_c,
        with_linear=True,
    )
    n = 5
    x = torch.randn(n, (lmax + 1) ** 2, in_c)
    y = tp(x, with_linear=False)
    assert y.shape[0] == n
    assert y.shape[2] == in_c
    assert y.shape[1] == sum((2 * l + 1) * c for l, c in tp.l3_sequential)
    z = tp.forward_linear(y)
    assert z.shape == (n, (lmax + 1) ** 2, out_c)


def test_so2_first_order_rotation_equivariance():
    """Rotate positions + irreps → rotated SO2 TP output (eval, gamma=0)."""
    from enerzyme.models.e2former.so2_tensor_product import E2TensorProductSO2_FirstOrder
    from enerzyme.models.e2former.wigner_otf import build_so2_wigner_frames

    torch.manual_seed(0)
    dtype = torch.float64
    lmax = 2
    c = 8
    heads = 2
    n = 4
    k = 3
    irreps = f"{c}x0e+{c}x1e+{c}x2e"
    irreps_out = f"{c}x0e+{c}x1e+{c}x2e"
    tp = E2TensorProductSO2_FirstOrder(irreps, irreps_out, head=heads, order=1).to(
        dtype=dtype
    )
    tp.eval()

    pos = torch.randn(n, 3, dtype=dtype)
    # Avoid near-zero norms for solid-harmonic scaling.
    pos = pos + torch.tensor([2.0, 0.0, 0.0], dtype=dtype)
    h = torch.randn(n, (lmax + 1) ** 2, c, dtype=dtype)
    idx = torch.randint(0, n, (n, k))
    alpha = torch.rand(n, k, heads, dtype=dtype)
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _run(p, feat):
        frames = build_so2_wigner_frames(
            p,
            p,
            lmax,
            l3_sequential=tp.so2_tp.l3_sequential,
            training=False,
        )
        frames["f_sparse_idx_expnode"] = idx
        return tp(
            p,
            p,
            None,
            feat,
            alpha,
            f_sparse_idx_expnode=idx,
            batched_data=frames,
        )

    out0 = _run(pos, h)
    r = _random_so3(dtype=dtype)
    # Rotate SH features degree-wise with Wigner-D of R.
    from e3nn import o3

    pos_r = pos @ r.T
    angles = o3.matrix_to_angles(r.unsqueeze(0))
    h_r = torch.zeros_like(h)
    for ell in range(lmax + 1):
        d = o3.wigner_D(ell, angles[0], angles[1], angles[2])[0].to(dtype)
        sl = slice(ell**2, (ell + 1) ** 2)
        h_r[:, sl, :] = torch.einsum("ij,bjc->bic", d, h[:, sl, :])

    out1 = _run(pos_r, h_r)
    out0_r = torch.zeros_like(out0)
    for ell in range(lmax + 1):
        d = o3.wigner_D(ell, angles[0], angles[1], angles[2])[0].to(dtype)
        sl = slice(ell**2, (ell + 1) ** 2)
        out0_r[:, sl, :] = torch.einsum("ij,bjc->bic", d, out0[:, sl, :])

    assert_allclose(
        out1.detach().cpu().numpy(),
        out0_r.detach().cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )


def test_sparse_qk_v_pytorch_fallback():
    from enerzyme.models.e2former.triton_sparse import sparse_qk, sparse_v_agg

    torch.manual_seed(0)
    n, k, h, d, feat = 5, 4, 2, 8, 6
    query = torch.randn(n, h, d)
    key = torch.randn(n, h, d)
    idx = torch.randint(0, n, (n, k))
    gate = torch.randn(n, k, h)
    scores = sparse_qk(query, key, idx, gate, scale=0.1, use_triton=False)
    assert scores.shape == (n, k, h)
    value = torch.randn(n, feat, h)
    alpha = torch.softmax(scores, dim=1)
    out = sparse_v_agg(value, alpha, idx, use_triton=False)
    assert out.shape == (n, feat, h)


def test_so2_first_order_attention_alpha_batch_is_atoms_not_fragments():
    """Cluster path: alpha/x_edge are [N_atoms,K,*]; value is [N_frag,*].

    Short-range V2 tests keep N_atoms == N_value, so ``f_n = value.shape[0]``
    never diverges. Choose N_atoms * K divisible by N_frag so a buggy reshape
    would *succeed* with a wrong layout instead of raising at reshape time.
    """
    from e3nn import o3

    from enerzyme.models.e2former.attention import So2FirstOrderAttention
    from enerzyme.models.e2former.wigner_otf import build_so2_wigner_frames

    torch.manual_seed(0)
    n_atoms, n_frag, k, heads = 6, 3, 4, 2
    assert (n_atoms * k) % n_frag == 0
    c, lmax = 8, 2
    irreps = o3.Irreps(f"{c}x0e+{c}x1e+{c}x2e")
    irreps_head = o3.Irreps(f"4x0e+4x1e+4x2e")
    edge_channel_list = [32, 16, 16]
    attn = So2FirstOrderAttention(
        irreps, irreps_head, heads, edge_channel_list, lmax
    )
    attn.eval()

    alpha = torch.rand(n_atoms, k, heads)
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
    value = torch.randn(n_frag, (lmax + 1) ** 2, c)
    x_edge = torch.randn(n_atoms, k, edge_channel_list[0])
    node_pos = torch.randn(n_atoms, 3) + 2.0
    frag_pos = torch.randn(n_frag, 3) + 2.0
    edge_dis = torch.rand(n_atoms, k) + 0.5
    idx = torch.randint(0, n_frag, (n_atoms, k))
    frames = build_so2_wigner_frames(
        node_pos,
        frag_pos,
        lmax,
        l3_sequential=attn.l3_sequential,
        training=False,
    )
    # ClusterSparse wires fragment positions as f_exp_node_pos (≠ node_pos).
    frames["f_exp_node_pos"] = frag_pos
    frames["f_sparse_idx_expnode"] = idx

    out = attn(
        alpha=alpha,
        value=value,
        x_edge=x_edge,
        node_pos=node_pos,
        edge_dis=edge_dis,
        batched_data=frames,
        use_triton=False,
    )
    assert out.shape[0] == n_atoms
    assert out.shape[1:] == ((lmax + 1) ** 2, c)
    assert torch.isfinite(out).all()
