"""Helpers for EquiformerV2 numerical parity against vendored upstream fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
from numpy.testing import assert_allclose
from torch import Tensor

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "equiformer_v2_upstream"
# Prefer EquiformerV2 fixture over escn_upstream when both are on sys.path.
_escn = str(Path(__file__).resolve().parent / "fixtures" / "escn_upstream")
if _escn in sys.path:
    sys.path.remove(_escn)
if str(FIXTURE_ROOT) in sys.path:
    sys.path.remove(str(FIXTURE_ROOT))
sys.path.insert(0, str(FIXTURE_ROOT))

PARITY_HPARAMS = {
    "lmax": 2,
    "mmax": 2,
    "sphere_channels": 8,
    "attn_hidden_channels": 8,
    "num_heads": 2,
    "attn_alpha_channels": 4,
    "attn_value_channels": 4,
    "ffn_hidden_channels": 16,
    "edge_channels": 8,
    "num_rbf": 8,
    "max_Za": 16,
}


def assert_close(a: Tensor, b: Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
    )


def build_complete_graph(num_nodes: int) -> Tensor:
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.stack(
        [torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)],
        dim=0,
    )


def deterministic_edge_rot_mat(edge_distance_vec: Tensor) -> Tensor:
    from enerzyme.models.so3 import init_edge_rot_mat

    return init_edge_rot_mat(edge_distance_vec)


def copy_state_dict(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy overlapping parameters (Enerzyme mapping/grids may not be nn buffers)."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    missing_in_src = set(dst_sd) - set(src_sd)
    if missing_in_src:
        raise AssertionError(f"dst has keys missing from src: {missing_in_src}")
    with torch.no_grad():
        for k, dst_t in dst_sd.items():
            src_t = src_sd[k]
            if dst_t.shape != src_t.shape:
                raise AssertionError(
                    f"shape mismatch for {k}: dst {tuple(dst_t.shape)} vs src {tuple(src_t.shape)}"
                )
            dst_t.copy_(src_t)


def build_so2_convolution_pair(extra_m0: int | None = None):
    """Enerzyme vs upstream SO2_Convolution with shared mapping."""
    from enerzyme.models.so3 import SO2_Convolution as EZConv
    from enerzyme.models.so3 import CoefficientMapping as EZMap
    from so2_ops import SO2_Convolution as OffConv
    from eqv2_so3 import CoefficientMappingModule as OffMap

    h = PARITY_HPARAMS
    device = torch.device("cpu")
    lmax, mmax = h["lmax"], h["mmax"]
    ez_map = EZMap([lmax], [mmax], device)
    off_map = OffMap([lmax], [mmax])
    edge_channels_list = [h["num_rbf"], h["edge_channels"], h["edge_channels"]]
    kwargs = dict(
        sphere_channels=h["sphere_channels"],
        m_output_channels=h["sphere_channels"],
        lmax_list=[lmax],
        mmax_list=[mmax],
        internal_weights=False,
        edge_channels_list=edge_channels_list,
        extra_m0_output_channels=extra_m0,
    )
    ez = EZConv(mappingReduced=ez_map, **kwargs)
    off = OffConv(mappingReduced=off_map, **kwargs)
    return ez, off, ez_map, off_map, h


def build_so3_grids_v2(lmax: int, resolution: int | None = None):
    """Component-normalized grids with mmax rescale (EquiformerV2 style)."""
    from enerzyme.models.so3 import SO3_Grid as EZ_Grid
    from eqv2_so3 import SO3_Grid as Off_Grid

    ez = torch.nn.ModuleList()
    off = torch.nn.ModuleList()
    for lval in range(lmax + 1):
        ez_m = torch.nn.ModuleList()
        off_m = torch.nn.ModuleList()
        for mval in range(lmax + 1):
            ez_m.append(
                EZ_Grid(
                    lval,
                    mval,
                    resolution=resolution,
                    normalization="component",
                    rescale_by_mmax=True,
                )
            )
            off_m.append(
                Off_Grid(
                    lval,
                    mval,
                    resolution=resolution,
                    normalization="component",
                )
            )
        ez.append(ez_m)
        off.append(off_m)
    return ez, off
