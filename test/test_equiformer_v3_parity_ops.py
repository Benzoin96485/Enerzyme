"""Numerical parity: EquiformerV3 ops vs upstream fixture."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from equiformer_v3_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    copy_state_dict,
)


def test_merge_layer_norm_matches_upstream():
    from enerzyme.models.so3.layer_norm import EquivariantMergeLayerNorm as EZNorm
    from layer_norm import EquivariantMergeLayerNorm as OffNorm

    torch.manual_seed(0)
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax, c = h["lmax"], h["sphere_channels"]
    ez = EZNorm(lmax, c).to(dtype)
    off = OffNorm(lmax, c).to(dtype)
    copy_state_dict(ez, off)
    x = torch.randn(5, (lmax + 1) ** 2, c, dtype=dtype)
    assert_close(ez(x.clone()), off(x.clone()))


def test_so3_linear_matches_upstream():
    from enerzyme.models.so3.linear import SO3Linear as EZLin
    from eqv3_so3 import SO3Linear as OffLin

    torch.manual_seed(1)
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax = h["lmax"]
    cin, cout = h["sphere_channels"], h["ffn_hidden_channels"]
    ez = EZLin(cin, cout, lmax=lmax).to(dtype)
    off = OffLin(cin, cout, lmax=lmax).to(dtype)
    copy_state_dict(ez, off)
    x = torch.randn(4, (lmax + 1) ** 2, cin, dtype=dtype)
    assert_close(ez(x.clone()), off(x.clone()))


def test_so2_linear_matches_upstream():
    from enerzyme.models.so3.so2_ops import SO2Linear as EZLin
    from so2_ops import SO2Linear as OffLin

    torch.manual_seed(2)
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax, mmax = h["lmax"], h["mmax"]
    cin, cout = h["sphere_channels"], h["attn_hidden_channels"]
    # m-primary component count
    n_m = 0
    for m in range(mmax + 1):
        n_m += lmax + 1 - m
        if m > 0:
            n_m += lmax + 1 - m
    ez = EZLin(cin, cout, lmax, mmax, extra_m0_out_channels=8).to(dtype)
    off = OffLin(cin, cout, lmax, mmax, extra_m0_out_channels=8).to(dtype)
    copy_state_dict(ez, off)
    x = torch.randn(6, n_m, cin, dtype=dtype)
    ez_y, ez_extra = ez(x.clone())
    off_y, off_extra = off(x.clone())
    assert_close(ez_y, off_y)
    assert_close(ez_extra, off_extra)


def test_polynomial_envelope_matches_upstream():
    from enerzyme.models.so3.envelope import PolynomialEnvelope as EZEnv
    from envelope import PolynomialEnvelope as OffEnv

    torch.manual_seed(3)
    dtype = torch.float64
    ez = EZEnv(cutoff=5.0, exponent=5).to(dtype)
    off = OffEnv(cutoff=5.0, exponent=5).to(dtype)
    d = torch.linspace(0.1, 6.0, 20, dtype=dtype)
    assert_close(ez(d), off(d))


def test_graph_softmax_envelope_matches_upstream():
    from enerzyme.models.so3.softmax import GraphSoftmax as EZSoft
    from softmax import GraphSoftmax as OffSoft

    torch.manual_seed(4)
    dtype = torch.float64
    ez = EZSoft(eps=1e-16).to(dtype)
    off = OffSoft(eps=1e-16).to(dtype)
    src = torch.randn(12, dtype=dtype)
    index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.long)
    rescale = torch.rand(12, dtype=dtype)
    assert_close(
        ez(src.clone(), index, num_nodes=3, exp_rescale=rescale.clone()),
        off(src.clone(), index, num_nodes=3, exp_rescale=rescale.clone()),
    )
