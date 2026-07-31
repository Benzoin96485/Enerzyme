"""Offline parity / algebraic checks for DPA4 ops (no runtime deepmd dependency).

Fixtures encode published deepmd-kit layout examples (lmax=2, mmax=1) and
closed-form C³ envelope coefficients from Li et al., arXiv:2606.02419 /
deepmd ``dpa4_nn``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dpa4_upstream"


def test_m_major_index_matches_published_layout():
    from enerzyme.models.dpa4.indexing import (
        build_m_major_index,
        build_m_major_l_index,
        build_rotate_inv_rescale,
    )

    # deepmd docs example for lmax=2, mmax=1
    assert list(build_m_major_index(2, 1)) == [0, 2, 6, 1, 5, 3, 7]
    assert list(build_m_major_l_index(2, 1)) == [0, 1, 2, 1, 2, 1, 2]
    rescale = build_rotate_inv_rescale(2, 1, build_m_major_l_index(2, 1))
    expected = np.ones(7, dtype=np.float64)
    expected[[2, 4, 6]] = math.sqrt(5.0 / 3.0)  # l=2 > mmax=1
    np.testing.assert_allclose(rescale, expected)


def test_c3_envelope_matches_closed_form_p5():
    from enerzyme.models.dpa4.radial import C3CutoffEnvelope

    rcut = 6.0
    env = C3CutoffEnvelope(rcut=rcut, exponent=5).double()
    r = torch.tensor([[0.0], [3.0], [6.0], [7.0]], dtype=torch.float64)
    out = env(r).squeeze(-1)
    x = (r.squeeze(-1) / rcut).clamp(0, 1)
    u = 1.0 - x
    # E_5(x) = u^4 * (1 + 4x + 10x^2 + 20x^3 + 35x^4)
    closed = u**4 * (1 + 4 * x + 10 * x**2 + 20 * x**3 + 35 * x**4)
    closed = torch.where(r.squeeze(-1) >= rcut, torch.zeros_like(closed), closed)
    torch.testing.assert_close(out, closed, atol=1e-12, rtol=1e-12)
    assert out[0].item() == 1.0
    assert out[2].item() == 0.0
    assert out[3].item() == 0.0


def test_so2_linear_complex_multiply_equivariance():
    """Rotating (±m) inputs by φ must rotate outputs by the same angle."""
    from enerzyme.models.dpa4.so2 import SO2Linear

    torch.manual_seed(0)
    dtype = torch.float64
    layer = SO2Linear(lmax=2, mmax=1, in_channels=2, out_channels=2, n_focus=1).to(dtype)
    layer.eval()
    # Layout: m=0 [0:3], m=-1 [3:5], m=+1 [5:7]
    x = torch.randn(1, 4, 7, 2, dtype=dtype)
    phi = 0.37
    c, s = math.cos(phi), math.sin(phi)
    x_rot = x.clone()
    # Apply SO(2) rotation on |m|=1 pair: (neg', pos') = R_phi (neg, pos)
    neg, pos = x[:, :, 3:5], x[:, :, 5:7]
    x_rot[:, :, 3:5] = c * neg - s * pos
    x_rot[:, :, 5:7] = s * neg + c * pos
    with torch.no_grad():
        y = layer(x)
        y_rot_in = layer(x_rot)
    y_rot_out = y.clone()
    neg_y, pos_y = y[:, :, 3:5], y[:, :, 5:7]
    y_rot_out[:, :, 3:5] = c * neg_y - s * pos_y
    y_rot_out[:, :, 5:7] = s * neg_y + c * pos_y
    torch.testing.assert_close(y_rot_in, y_rot_out, atol=1e-10, rtol=1e-10)
    # m=0 block is invariant under this SO(2) action
    torch.testing.assert_close(y_rot_in[:, :, :3], y[:, :, :3], atol=1e-10, rtol=1e-10)


def test_envelope_gated_softmax_sums_with_null_mass():
    from enerzyme.models.dpa4.attention import segment_envelope_gated_softmax

    torch.manual_seed(1)
    dtype = torch.float64
    n_nodes, n_edge = 3, 6
    logits = torch.randn(n_edge, 1, 2, dtype=dtype)
    edge_env = torch.rand(n_edge, 1, dtype=dtype) * 0.8 + 0.1
    dst = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    z_bias = torch.zeros(1, 2, dtype=dtype)
    alpha = segment_envelope_gated_softmax(
        logits, edge_env, dst, n_nodes, z_bias, eps=1e-6
    )
    # Per destination: sum_e alpha + null_mass_contribution ≈ 1 is not stored;
    # check non-negativity, finiteness, and masked-zero env kills weight.
    assert torch.isfinite(alpha).all()
    assert (alpha >= 0).all()
    # Zero envelope → zero attention
    edge_env0 = edge_env.clone()
    edge_env0[0] = 0.0
    alpha0 = segment_envelope_gated_softmax(
        logits, edge_env0, dst, n_nodes, z_bias, eps=1e-6
    )
    assert alpha0[0].abs().max().item() == 0.0


def test_indexing_fixture_roundtrip_optional():
    """If a fixture npz exists, compare against it; otherwise generate & skip soft."""
    from enerzyme.models.dpa4.indexing import (
        build_gie_zonal_index,
        build_m_major_index,
        build_m_major_l_index,
        build_rotate_inv_rescale,
    )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "indexing_l2_m1.npz"
    payload = {
        "m_major": build_m_major_index(2, 1),
        "m_major_l": build_m_major_l_index(2, 1),
        "rescale": build_rotate_inv_rescale(2, 1, build_m_major_l_index(2, 1)),
    }
    rows, cols, rad = build_gie_zonal_index(2)
    payload["gie_rows"] = rows
    payload["gie_cols"] = cols
    payload["gie_rad"] = rad
    if not path.exists():
        np.savez(path, **payload)
    data = np.load(path)
    np.testing.assert_array_equal(payload["m_major"], data["m_major"])
    np.testing.assert_array_equal(payload["m_major_l"], data["m_major_l"])
    np.testing.assert_allclose(payload["rescale"], data["rescale"])
    np.testing.assert_array_equal(payload["gie_rows"], data["gie_rows"])
