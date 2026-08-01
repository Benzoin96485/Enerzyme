"""Helpers for EquiformerV3 numerical parity against vendored upstream fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose
from torch import Tensor

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "equiformer_v3_upstream"
# Prefer EquiformerV3 fixture over other so3 fixtures when both are on sys.path.
for rival in (
    Path(__file__).resolve().parent / "fixtures" / "escn_upstream",
    Path(__file__).resolve().parent / "fixtures" / "equiformer_v2_upstream",
):
    rival_s = str(rival)
    if rival_s in sys.path:
        sys.path.remove(rival_s)
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


def _is_s2_grid_buffer(key: str) -> bool:
    return key.endswith("to_grid_mat") or key.endswith("from_grid_mat")


def copy_state_dict(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy overlapping parameters; skip S² grid mats when layouts diverge."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    missing_in_src = {
        k for k in dst_sd if k not in src_sd and not _is_s2_grid_buffer(k)
    }
    if missing_in_src:
        raise AssertionError(f"dst has keys missing from src: {missing_in_src}")
    with torch.no_grad():
        for k, dst_t in dst_sd.items():
            if _is_s2_grid_buffer(k) or k not in src_sd:
                continue
            src_t = src_sd[k]
            if dst_t.shape != src_t.shape:
                raise AssertionError(
                    f"shape mismatch for {k}: dst {tuple(dst_t.shape)} vs src {tuple(src_t.shape)}"
                )
            dst_t.copy_(src_t)


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
