"""Wigner-6j TP parity: arbitrary-order path vs edge-wise vanilla TP."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])


def test_e2_tensor_product_arbitrary_vs_vanilla_orders_1_2():
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
        irreps = "+".join(
            [f"{head * hidden}x{l}e" for l in range(order + 1)]
        )
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
        # Upstream self-check uses ratio; allow small relative error
        denom = out_van.abs().clamp_min(1e-6)
        rel = ((out_arb - out_van) / denom).abs()
        assert float(rel.mean()) < 0.05, (order, float(rel.mean()), float(rel.max()))
        assert torch.isfinite(out_arb).all()
