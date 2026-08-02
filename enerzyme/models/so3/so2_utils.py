"""SO(2) layout helpers for TECE / TACE SO(2) paths.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def so2_expand_index(
    mmax: int, lmax: int, start: int = 0
) -> Tuple[int, torch.Tensor]:
    expand_index = []
    offset = 0
    for m in range(start, mmax + 1):
        index = torch.arange(lmax + 1 - m)
        index = index + offset
        expand_index.append(index)
        if m > 0:
            expand_index.append(index)
        offset = offset + len(index)
    expand_index = torch.cat(expand_index, dim=0).long()
    return offset, expand_index


def so3_expand_index(mmax: int, lmax: int) -> Tuple[int, torch.Tensor]:
    assert mmax == lmax
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    start_idx = 0
    for l in range(lmax + 1):
        length = 2 * l + 1
        expand_index[start_idx : start_idx + length] = l
        start_idx = start_idx + length
    return lmax + 1, expand_index


def num_so2_components(lmax: int, mmax: int) -> int:
    total = lmax + 1
    for m in range(1, mmax + 1):
        total += 2 * (lmax + 1 - m)
    return total


def rotate_real_irrep(x: torch.Tensor, theta: float, m: int) -> torch.Tensor:
    # x: [B, 2, n, C]
    c = math.cos(m * theta)
    s = math.sin(m * theta)
    xr = x[:, 0]
    xi = x[:, 1]
    yr = c * xr - s * xi
    yi = s * xr + c * xi
    return torch.stack([yr, yi], dim=1)
