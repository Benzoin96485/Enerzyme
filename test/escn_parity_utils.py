"""Helpers for eSCN numerical parity against vendored fairchem v1 fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.testing import assert_allclose
from torch import Tensor
from torch.nn import SiLU

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "escn_upstream"
MOL_PATH = Path(__file__).resolve().parent / "fixtures" / "equiformer_parity_mol.npz"

# Allow `from so3 import ...` / `from escn_blocks import ...` relative to fixture dir.
if str(FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURE_ROOT))

PARITY_HPARAMS = {
    "lmax": 2,
    "mmax": 2,
    "sphere_channels": 8,
    "hidden_channels": 16,
    "edge_channels": 8,
    "num_rbf": 8,
    "max_Za": 16,
}


def load_parity_mol() -> Dict[str, Tensor]:
    data = np.load(MOL_PATH)
    return {
        "Za": torch.as_tensor(data["Za"], dtype=torch.long),
        "pos": torch.as_tensor(data["pos"], dtype=torch.float64),
        "r_max": float(data["r_max"]),
    }


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
    """Shared frame builder for parity (Enerzyme-style least-aligned helper)."""
    from enerzyme.models.so3 import init_edge_rot_mat

    return init_edge_rot_mat(edge_distance_vec)


def assert_close(a: Tensor, b: Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
    )


def build_so3_grids(lmax: int, resolution: int | None = None):
    """Parallel ModuleLists of SO3_Grid for official and Enerzyme."""
    from enerzyme.models.so3 import SO3_Grid as EZ_Grid
    from so3 import SO3_Grid as Off_Grid

    ez = torch.nn.ModuleList()
    off = torch.nn.ModuleList()
    for lval in range(lmax + 1):
        ez_m = torch.nn.ModuleList()
        off_m = torch.nn.ModuleList()
        for m in range(lmax + 1):
            ez_m.append(EZ_Grid(lval, m, resolution=resolution))
            off_m.append(Off_Grid(lval, m, resolution=resolution))
        ez.append(ez_m)
        off.append(off_m)
    return ez, off


def copy_state_dict(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    missing = [k for k in dst_sd if k not in src_sd]
    unexpected = [k for k in src_sd if k not in dst_sd]
    if missing or unexpected:
        raise KeyError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
    for k, v in src_sd.items():
        if dst_sd[k].shape != v.shape:
            raise RuntimeError(f"shape mismatch for {k}: {dst_sd[k].shape} vs {v.shape}")
    dst.load_state_dict(src_sd)


def build_so2_pair(hparams: Dict | None = None):
    from enerzyme.models.so3 import SO2Block as EZ_SO2
    from escn_blocks import SO2Block as Off_SO2

    h = {**PARITY_HPARAMS, **(hparams or {})}
    lmax_list = [h["lmax"]]
    mmax_list = [h["mmax"]]
    act = SiLU()
    kwargs = dict(
        sphere_channels=h["sphere_channels"],
        hidden_channels=h["hidden_channels"],
        edge_channels=h["edge_channels"],
        lmax_list=lmax_list,
        mmax_list=mmax_list,
        act=act,
    )
    off = Off_SO2(**kwargs).double()
    ez = EZ_SO2(**kwargs).double()
    copy_state_dict(ez, off)
    return ez, off, h


def build_message_pair(hparams: Dict | None = None, so3_grids=None):
    from enerzyme.models.escn.interaction import MessageBlock as EZ_Msg
    from escn_blocks import MessageBlock as Off_Msg

    h = {**PARITY_HPARAMS, **(hparams or {})}
    lmax_list = [h["lmax"]]
    mmax_list = [h["mmax"]]
    act = SiLU()
    if so3_grids is None:
        ez_grid, off_grid = build_so3_grids(h["lmax"])
    else:
        ez_grid, off_grid = so3_grids

    max_num_elements = h["max_Za"] + 1
    off = Off_Msg(
        h["sphere_channels"],
        h["hidden_channels"],
        h["edge_channels"],
        lmax_list,
        mmax_list,
        h["num_rbf"],
        max_num_elements,
        off_grid,
        act,
    ).double()
    ez = EZ_Msg(
        h["sphere_channels"],
        h["hidden_channels"],
        h["edge_channels"],
        lmax_list,
        mmax_list,
        h["num_rbf"],
        h["max_Za"],
        ez_grid,
        act,
    ).double()
    copy_state_dict(ez, off)
    return ez, off, h, ez_grid, off_grid


def build_layer_pair(hparams: Dict | None = None):
    from enerzyme.models.escn.interaction import LayerBlock as EZ_Layer
    from escn_blocks import LayerBlock as Off_Layer

    h = {**PARITY_HPARAMS, **(hparams or {})}
    lmax_list = [h["lmax"]]
    mmax_list = [h["mmax"]]
    act = SiLU()
    ez_grid, off_grid = build_so3_grids(h["lmax"])
    max_num_elements = h["max_Za"] + 1
    off = Off_Layer(
        h["sphere_channels"],
        h["hidden_channels"],
        h["edge_channels"],
        lmax_list,
        mmax_list,
        h["num_rbf"],
        max_num_elements,
        off_grid,
        act,
    ).double()
    ez = EZ_Layer(
        h["sphere_channels"],
        h["hidden_channels"],
        h["edge_channels"],
        lmax_list,
        mmax_list,
        h["num_rbf"],
        h["max_Za"],
        ez_grid,
        act,
    ).double()
    copy_state_dict(ez, off)
    return ez, off, h, ez_grid, off_grid
