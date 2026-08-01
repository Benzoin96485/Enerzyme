"""Numerical parity: Enerzyme so3 SH / L0 and so3krates blocks vs fixtures."""

from __future__ import annotations

import sys

import torch
from torch.nn import Identity

sys.path.extend(["..", "."])

from so3krates_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    copy_state_dict,
)


def test_real_spherical_harmonics_matches_upstream():
    from enerzyme.models.so3 import RealSphericalHarmonics as EZSH
    from spherical_harmonics import RealSphericalHarmonics as OffSH

    torch.manual_seed(0)
    degrees = PARITY_HPARAMS["degrees"]
    ez = EZSH(degrees)
    off = OffSH(degrees)
    vecs = torch.randn(7, 3, dtype=torch.float64)
    assert_close(ez(vecs), off(vecs), atol=1e-12, rtol=1e-12)


def test_l0_contraction_matches_upstream():
    from enerzyme.models.so3 import L0Contraction as EZL0
    from so3_conv_invariants import L0Contraction as OffL0

    torch.manual_seed(1)
    degrees = PARITY_HPARAMS["degrees"]
    m_tot = sum(2 * l + 1 for l in degrees)
    ez = EZL0(degrees, dtype=torch.float64)
    off = OffL0(degrees, dtype=torch.float64)
    assert_close(ez.cg_rep.double(), off.cg_rep.double(), atol=1e-12, rtol=1e-12)
    x = torch.randn(5, m_tot, dtype=torch.float64)
    assert_close(ez(x), off(x), atol=1e-12, rtol=1e-12)


def test_filternet_matches_upstream():
    from enerzyme.models.so3krates.interaction import FilterNet as EZFN
    from upstream_blocks import FilterNet as OffFN

    torch.manual_seed(2)
    h = PARITY_HPARAMS
    ez = EZFN(h["degrees"], h["num_rbf"], h["num_features"])
    off = OffFN(h["degrees"], h["num_rbf"], h["num_features"])
    copy_state_dict(ez, off)
    rbf = torch.randn(6, h["num_rbf"])
    dgamma = torch.randn(6, len(h["degrees"]))
    assert_close(ez(rbf, dgamma), off(rbf, dgamma))


def test_interaction_block_matches_upstream():
    from enerzyme.models.so3krates.interaction import (
        InteractionBlock as EZIB,
    )
    from upstream_blocks import InteractionBlock as OffIB

    torch.manual_seed(3)
    h = PARITY_HPARAMS
    m_tot = sum(2 * l + 1 for l in h["degrees"])
    ez = EZIB(h["degrees"], h["num_features"])
    off = OffIB(h["degrees"], h["num_features"])
    copy_state_dict(ez, off)
    inv = torch.randn(4, h["num_features"])
    ev = torch.randn(4, m_tot)
    ez_d_inv, ez_d_ev = ez(inv, ev)
    off_d_inv, off_d_ev = off(inv, ev)
    assert_close(ez_d_inv, off_d_inv)
    assert_close(ez_d_ev, off_d_ev)


def test_attention_block_matches_upstream():
    from enerzyme.models.so3krates.interaction import (
        EuclideanAttentionBlock as EZAtt,
        FilterNet as EZFN,
    )
    from upstream_blocks import (
        EuclideanAttentionBlock as OffAtt,
        FilterNet as OffFN,
    )

    torch.manual_seed(4)
    h = PARITY_HPARAMS
    m_tot = sum(2 * l + 1 for l in h["degrees"])
    N, P = 5, 8
    ez_f_inv = EZFN(h["degrees"], h["num_rbf"], h["num_features"])
    ez_f_ev = EZFN(h["degrees"], h["num_rbf"], h["num_features"])
    off_f_inv = OffFN(h["degrees"], h["num_rbf"], h["num_features"])
    off_f_ev = OffFN(h["degrees"], h["num_rbf"], h["num_features"])
    copy_state_dict(ez_f_inv, off_f_inv)
    copy_state_dict(ez_f_ev, off_f_ev)

    ez = EZAtt(
        degrees=h["degrees"],
        num_heads=h["num_heads"],
        num_features=h["num_features"],
        filter_net_inv=ez_f_inv,
        filter_net_ev=ez_f_ev,
        message_normalization="avg_num_neighbors",
        avg_num_neighbors=h["avg_num_neighbors"],
        qk_non_linearity=Identity,
    )
    off = OffAtt(
        degrees=h["degrees"],
        num_heads=h["num_heads"],
        num_features=h["num_features"],
        filter_net_inv=off_f_inv,
        filter_net_ev=off_f_ev,
        message_normalization="avg_num_neighbors",
        avg_num_neighbors=h["avg_num_neighbors"],
        qk_non_linearity=Identity,
    )
    # Copy attention weights (filters already synced).
    with torch.no_grad():
        for name in ("W_q_inv", "W_k_inv", "W_v_inv", "W_q_ev", "W_k_ev"):
            getattr(ez, name).copy_(getattr(off, name))

    inv = torch.randn(N, h["num_features"])
    ev = torch.randn(N, m_tot)
    rbf = torch.randn(P, h["num_rbf"])
    senders = torch.randint(0, N, (P,))
    receivers = torch.randint(0, N, (P,))
    sh = torch.randn(P, m_tot)
    cutoffs = torch.rand(P)

    ez_di, ez_de = ez(inv, ev, rbf, senders, receivers, sh, cutoffs)
    off_di, off_de = off(inv, ev, rbf, senders, receivers, sh, cutoffs)
    assert_close(ez_di, off_di)
    assert_close(ez_de, off_de)
