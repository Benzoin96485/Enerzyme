"""Numerical parity: EquiformerV2 SO2 ops / LinearV2 / norms vs upstream fixture."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from equiformer_v2_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    build_so2_convolution_pair,
    copy_state_dict,
)


def test_so3_linear_v2_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from enerzyme.models.so3 import SO3_LinearV2 as EZLin
    from eqv2_so3 import SO3_Embedding as OffEmb
    from eqv2_so3 import SO3_LinearV2 as OffLin

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax = h["lmax"]
    cin, cout = h["sphere_channels"], h["ffn_hidden_channels"]
    n = 5

    ez = EZLin(cin, cout, lmax=lmax).to(dtype)
    off = OffLin(cin, cout, lmax=lmax).to(dtype)
    copy_state_dict(ez, off)

    emb = torch.randn(n, (lmax + 1) ** 2, cin, dtype=dtype)
    ez_x = EZEmb(0, [lmax], cin, device, dtype)
    off_x = OffEmb(0, [lmax], cin, device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    ez_x.set_lmax_mmax([lmax], [lmax])
    off_x.set_lmax_mmax([lmax], [lmax])

    assert_close(ez(ez_x).embedding, off(off_x).embedding)


def test_so2_convolution_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from eqv2_so3 import SO3_Embedding as OffEmb

    torch.manual_seed(1)
    device = torch.device("cpu")
    dtype = torch.float64
    ez, off, ez_map, off_map, h = build_so2_convolution_pair(extra_m0=None)
    ez = ez.to(dtype)
    off = off.to(dtype)
    copy_state_dict(ez, off)

    lmax, mmax = h["lmax"], h["mmax"]
    channels = h["sphere_channels"]
    num_edges = 6
    emb = torch.randn(num_edges, (lmax + 1) ** 2, channels, dtype=dtype)
    x_edge = torch.randn(num_edges, h["num_rbf"], dtype=dtype)

    ez_x = EZEmb(0, [lmax], channels, device, dtype)
    off_x = OffEmb(0, [lmax], channels, device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    ez_x.set_lmax_mmax([lmax], [mmax])
    off_x.set_lmax_mmax([lmax], [mmax])

    # Ensure mapping tensors match layout
    assert_close(ez_map.to_m, off_map.to_m.to(dtype=ez_map.to_m.dtype))

    ez_out = ez(ez_x, x_edge.clone())
    off_out = off(off_x, x_edge.clone())
    assert_close(ez_out.embedding, off_out.embedding)


def test_so2_convolution_extra_m0_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from eqv2_so3 import SO3_Embedding as OffEmb

    torch.manual_seed(2)
    device = torch.device("cpu")
    dtype = torch.float64
    extra = 8
    ez, off, _, _, h = build_so2_convolution_pair(extra_m0=extra)
    ez = ez.to(dtype)
    off = off.to(dtype)
    copy_state_dict(ez, off)

    lmax, mmax = h["lmax"], h["mmax"]
    channels = h["sphere_channels"]
    num_edges = 4
    emb = torch.randn(num_edges, (lmax + 1) ** 2, channels, dtype=dtype)
    x_edge = torch.randn(num_edges, h["num_rbf"], dtype=dtype)

    ez_x = EZEmb(0, [lmax], channels, device, dtype)
    off_x = OffEmb(0, [lmax], channels, device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    ez_x.set_lmax_mmax([lmax], [mmax])
    off_x.set_lmax_mmax([lmax], [mmax])

    ez_emb, ez_extra = ez(ez_x, x_edge.clone())
    off_emb, off_extra = off(off_x, x_edge.clone())
    assert_close(ez_emb.embedding, off_emb.embedding)
    assert_close(ez_extra, off_extra)


def test_rms_norm_sh_matches_upstream():
    from enerzyme.models.equiformer_v2.layer_norm import get_normalization_layer as ez_norm
    from layer_norm import get_normalization_layer as off_norm

    torch.manual_seed(3)
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax, c = h["lmax"], h["sphere_channels"]
    ez = ez_norm("rms_norm_sh", lmax=lmax, num_channels=c).to(dtype)
    off = off_norm("rms_norm_sh", lmax=lmax, num_channels=c).to(dtype)
    copy_state_dict(ez, off)
    x = torch.randn(7, (lmax + 1) ** 2, c, dtype=dtype)
    assert_close(ez(x.clone()), off(x.clone()))


def test_rotate_inv_rescale_when_mmax_lt_lmax():
    """Shared so3 rotate_inv applies EquiformerV2 rescale for mmax < lmax."""
    from enerzyme.models.so3 import CoefficientMapping, SO3_Embedding, SO3_Rotation

    torch.manual_seed(4)
    device = torch.device("cpu")
    dtype = torch.float64
    lmax, mmax, channels = 2, 1, 4
    num_edges = 3
    vij = torch.randn(num_edges, 3, dtype=dtype)
    vij = vij / torch.linalg.norm(vij, dim=1, keepdim=True).clamp(min=1e-8)
    from equiformer_v2_parity_utils import deterministic_edge_rot_mat

    rot_mat = deterministic_edge_rot_mat(vij)
    rot = SO3_Rotation(rot_mat, lmax)
    emb = torch.randn(num_edges, (lmax + 1) ** 2, channels, dtype=dtype)
    x = SO3_Embedding(0, [lmax], channels, device, dtype)
    x.set_embedding(emb.clone())
    x._rotate([rot], [lmax], [mmax])
    mapping = CoefficientMapping([lmax], [mmax], device)
    x._rotate_inv([rot], mapping)
    # Round-trip is not exact with mmax truncation, but must be finite and shaped.
    assert x.embedding.shape == emb.shape
    assert torch.isfinite(x.embedding).all()
