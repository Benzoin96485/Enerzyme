"""Numerical parity: Enerzyme so3 ops vs vendored fairchem_core-1.10.0 so3."""

from __future__ import annotations

import sys

import torch
from torch.nn import SiLU

sys.path.extend(["..", "."])

from escn_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    build_so2_pair,
    build_so3_grids,
    deterministic_edge_rot_mat,
)


def test_coefficient_mapping_matches_upstream():
    from enerzyme.models.so3 import CoefficientMapping as EZMap
    from so3 import CoefficientMapping as OffMap

    device = torch.device("cpu")
    lmax, mmax = PARITY_HPARAMS["lmax"], PARITY_HPARAMS["mmax"]
    ez = EZMap([lmax], [mmax], device)
    off = OffMap([lmax], [mmax], device)
    assert_close(ez.l_harmonic.float(), off.l_harmonic.float())
    assert_close(ez.m_harmonic.float(), off.m_harmonic.float())
    assert_close(ez.m_complex.float(), off.m_complex.float())
    assert_close(ez.to_m, off.to_m)
    assert_close(ez.m_size.float(), off.m_size.float())


def test_wigner_rotate_inverse_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from enerzyme.models.so3 import SO3_Rotation as EZRot
    from so3 import SO3_Embedding as OffEmb
    from so3 import SO3_Rotation as OffRot
    from so3 import CoefficientMapping as OffMap
    from enerzyme.models.so3 import CoefficientMapping as EZMap

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    lmax = PARITY_HPARAMS["lmax"]
    channels = PARITY_HPARAMS["sphere_channels"]
    num_edges = 5

    vij = torch.randn(num_edges, 3, dtype=dtype)
    vij = vij / torch.linalg.norm(vij, dim=1, keepdim=True).clamp(min=1e-8)
    rot_mat = deterministic_edge_rot_mat(vij)

    ez_rot = EZRot(rot_mat, lmax)
    off_rot = OffRot(rot_mat, lmax)

    emb = torch.randn(num_edges, (lmax + 1) ** 2, channels, dtype=dtype)
    ez_x = EZEmb(0, [lmax], channels, device, dtype)
    off_x = OffEmb(0, [lmax], channels, device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())

    ez_x._rotate([ez_rot], [lmax], [lmax])
    off_x._rotate([off_rot], [lmax], [lmax])
    assert_close(ez_x.embedding, off_x.embedding)

    ez_map = EZMap([lmax], [lmax], device)
    off_map = OffMap([lmax], [lmax], device)
    ez_x._rotate_inv([ez_rot], ez_map)
    off_x._rotate_inv([off_rot], off_map)
    assert_close(ez_x.embedding, off_x.embedding)
    assert_close(ez_x.embedding, emb, atol=1e-5, rtol=1e-5)


def test_so2_block_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from so3 import SO3_Embedding as OffEmb
    from so3 import CoefficientMapping as OffMap
    from enerzyme.models.so3 import CoefficientMapping as EZMap

    torch.manual_seed(1)
    device = torch.device("cpu")
    dtype = torch.float64
    ez, off, h = build_so2_pair()
    lmax, mmax = h["lmax"], h["mmax"]
    channels = h["sphere_channels"]
    num_edges = 6

    emb = torch.randn(num_edges, (lmax + 1) ** 2, channels, dtype=dtype)
    x_edge = torch.randn(num_edges, h["edge_channels"], dtype=dtype)

    ez_x = EZEmb(0, [lmax], channels, device, dtype)
    off_x = OffEmb(0, [lmax], channels, device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    # After rotate in real pipeline mmax may be reduced; SO2Block expects m-primary of full lmax coeffs
    ez_x.set_lmax_mmax([lmax], [mmax])
    off_x.set_lmax_mmax([lmax], [mmax])

    ez_map = EZMap([lmax], [mmax], device)
    off_map = OffMap([lmax], [mmax], device)

    # SO2Block expects embeddings already rotated with reduced m layout size matching mappingReduced
    # Use full lmax coefficients with mapping for mmax — same as after rotate with mmax=lmax
    ez_x.set_lmax_mmax([lmax], [lmax])
    off_x.set_lmax_mmax([lmax], [lmax])
    # Rebuild mapping for reduced m after "fake rotate" that keeps all coeffs: use mapping with mmax
    # Official MessageBlock rotates then SO2 with mappingReduced(lmax,mmax). After rotate,
    # embedding length is res_size for that mmax. Simulate by rotating with mmax=lmax then
    # using CoefficientMapping(lmax,mmax) only if sizes match.
    # Simplest parity path: set mmax=lmax for this unit test.
    ez2, off2, h2 = build_so2_pair({"mmax": lmax})
    ez_map2 = EZMap([lmax], [lmax], device)
    off_map2 = OffMap([lmax], [lmax], device)
    ez_x2 = EZEmb(0, [lmax], channels, device, dtype)
    off_x2 = OffEmb(0, [lmax], channels, device, dtype)
    ez_x2.set_embedding(emb.clone())
    off_x2.set_embedding(emb.clone())
    ez2(ez_x2, x_edge, ez_map2)
    off2(off_x2, x_edge, off_map2)
    assert_close(ez_x2.embedding, off_x2.embedding)


def test_so3_grid_roundtrip_matches_upstream():
    torch.manual_seed(2)
    device = torch.device("cpu")
    dtype = torch.float64
    lmax = PARITY_HPARAMS["lmax"]
    channels = PARITY_HPARAMS["sphere_channels"]
    ez_grids, off_grids = build_so3_grids(lmax)

    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from so3 import SO3_Embedding as OffEmb
    from so3 import CoefficientMapping as OffMap
    from enerzyme.models.so3 import CoefficientMapping as EZMap

    num = 4
    emb = torch.randn(num, (lmax + 1) ** 2, channels, dtype=dtype)
    # Use only coeffs with m<=mmax for grid_act path
    mmax = lmax
    mapping = EZMap([lmax], [mmax], device)
    # grid act on reduced embedding: take coefficient_idx subset size
    idx = mapping.coefficient_idx(lmax, mmax)
    emb_red = emb[:, idx, :].contiguous()

    ez_x = EZEmb(0, [lmax], channels, device, dtype)
    off_x = OffEmb(0, [lmax], channels, device, dtype)
    ez_x.set_embedding(emb_red.clone())
    off_x.set_embedding(emb_red.clone())
    ez_x.set_lmax_mmax([lmax], [mmax])
    off_x.set_lmax_mmax([lmax], [mmax])

    act = SiLU()
    ez_x._grid_act(ez_grids, act, mapping)
    off_map = OffMap([lmax], [mmax], device)
    off_x._grid_act(off_grids, act, off_map)
    assert_close(ez_x.embedding, off_x.embedding, atol=1e-5, rtol=1e-5)
