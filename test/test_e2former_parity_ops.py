"""Numerical parity: E2Former Wigner-6j / SO2 TP vs upstream / vanilla refs."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from e2former_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    copy_state_dict,
)


def test_wigner6j_arbitrary_matches_vanilla_orders_1_2():
    """Factorized Wigner-6j path vs edge-wise vanilla TP (upstream self-check)."""
    from enerzyme.models.e2former.wigner6j import E2TensorProductArbitraryOrder

    torch.manual_seed(0)
    for order in (1, 2):
        head, hidden = 4, 8
        f_n1, f_n2 = 5, 6
        alpha_ij = torch.randn(f_n1, f_n2, head)
        h = torch.randn(f_n1, (order + 1) ** 2, head * hidden)
        exp_h = torch.randn(f_n2, (order + 1) ** 2, head * hidden)
        pos = torch.randn(f_n1, 3)
        exp_pos = torch.randn(f_n2, 3)
        irreps = "+".join([f"{head * hidden}x{l}e" for l in range(order + 1)])
        model = E2TensorProductArbitraryOrder(
            irreps,
            irreps,
            head,
            order=order,
            learnable_weight=True,
            connection_mode="uvw",
            path_normalization="element",
        )
        out_arb = model(pos, exp_pos, h, exp_h, alpha_ij)
        out_van = model.vanilla_forward(pos, exp_pos, h, exp_h, alpha_ij)
        denom = out_van.abs().clamp_min(1e-6)
        rel = ((out_arb - out_van) / denom).abs()
        assert float(rel.mean()) < 0.05, (order, float(rel.mean()), float(rel.max()))
        assert torch.isfinite(out_arb).all()


def test_so2_tp_givenl2_matches_ubio_fixture():
    from enerzyme.models.e2former.so2_tensor_product import SO2_TP_givenl2 as EZ
    from upstream_so2_tensor_product import SO2_TP_givenl2 as Off

    torch.manual_seed(0)
    h = PARITY_HPARAMS
    lmax, c = h["lmax"], h["channels"]
    ez = EZ(
        range(lmax + 1),
        1,
        range(lmax + 1),
        in_c=c,
        out_c=c,
        with_linear=True,
    )
    off = Off(
        range(lmax + 1),
        1,
        range(lmax + 1),
        in_c=c,
        out_c=c,
        with_linear=True,
    )
    copy_state_dict(ez, off)
    x = torch.randn(h["num_nodes"], (lmax + 1) ** 2, c)
    assert_close(ez(x, with_linear=False), off(x, with_linear=False))
    y = ez(x, with_linear=False)
    assert_close(ez.forward_linear(y), off.forward_linear(y))


def test_so2_first_order_tp_matches_ubio_fixture():
    from enerzyme.models.e2former.so2_tensor_product import (
        E2TensorProductSO2_FirstOrder as EZ,
    )
    from enerzyme.models.e2former.wigner_otf import build_so2_wigner_frames
    from upstream_so2_tensor_product import E2TensorProductSO2_FirstOrder as Off

    torch.manual_seed(1)
    h = PARITY_HPARAMS
    lmax, c, heads = h["lmax"], h["channels"], h["heads"]
    n, k = h["num_nodes"], h["topk"]
    irreps = f"{c}x0e+{c}x1e+{c}x2e"
    ez = EZ(irreps, irreps, head=heads, order=1)
    off = Off(irreps, irreps, head=heads, order=1)
    copy_state_dict(ez, off)
    ez.eval()
    off.eval()

    pos = torch.randn(n, 3) + torch.tensor([1.5, 0.0, 0.0])
    feat = torch.randn(n, (lmax + 1) ** 2, c)
    idx = torch.randint(0, n, (n, k))
    alpha = torch.rand(n, k, heads)
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
    frames = build_so2_wigner_frames(
        pos,
        pos,
        lmax,
        l3_sequential=ez.so2_tp.l3_sequential,
        training=False,
    )
    frames["f_sparse_idx_expnode"] = idx

    out_ez = ez(
        pos,
        pos,
        None,
        feat,
        alpha,
        f_sparse_idx_expnode=idx,
        batched_data=frames,
    )
    out_off = off(
        pos,
        pos,
        None,
        feat,
        alpha,
        f_sparse_idx_expnode=idx,
        batched_data=frames,
    )
    assert_close(out_ez, out_off, atol=1e-5, rtol=1e-5)
