"""Lebedev quadrature grids for Euclidean Fast Attention.

Grid tables are vendored from Google e3x (``e3x/so3/_lebedev_grids.npz``,
Apache-2.0). Weights already satisfy ``sum(w) == 1``, so
``sum_j w_j f(u_j)`` approximates ``(1/4π) ∫_{S²} f(u) du`` as used by EFA.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

_LEBEDEV_NPZ = Path(__file__).resolve().parent / "lebedev_grids.npz"

# Max ERoPE frequency budget vs Lebedev order (EFA SI / reference rope.py).
LEBEDEV_FREQUENCY_LOOKUP: Dict[int, float] = {
    50: float(np.pi),
    86: float(2 * np.pi),
    110: float(2.5 * np.pi),
    146: float(3 * np.pi),
    194: float(4 * np.pi),
    230: float(4.5 * np.pi),
    266: float(5 * np.pi),
    302: float(5.5 * np.pi),
    350: float(6.5 * np.pi),
    434: float(7.5 * np.pi),
    590: float(9 * np.pi),
    770: float(11 * np.pi),
    974: float(12.5 * np.pi),
    6000: float(35 * np.pi),
}


@lru_cache(maxsize=1)
def _load_index() -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(_LEBEDEV_NPZ)
    return np.asarray(data["num"]), np.asarray(data["precision"])


def available_lebedev_nums() -> Tuple[int, ...]:
    nums, _ = _load_index()
    return tuple(int(n) for n in nums)


def lebedev_quadrature(num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Lebedev points ``(M, 3)`` and weights ``(M,)`` for order ``num``.

    Selects the highest-precision grid with at most ``num`` points (e3x
    convention). Raises if ``num`` is below the smallest available grid.
    """
    nums, _ = _load_index()
    if num < int(nums.min()):
        raise ValueError(
            f"Lebedev num={num} is below the smallest available grid "
            f"({int(nums.min())}). Available: {available_lebedev_nums()}"
        )
    # Closest available with nums[i] <= num (prefer exact match).
    eligible = np.where(nums <= num)[0]
    i = int(eligible[np.argmax(nums[eligible])])
    if int(nums[i]) != num:
        raise ValueError(
            f"Lebedev num={num} is not available. Closest ≤num is "
            f"{int(nums[i])}. Available: {available_lebedev_nums()}"
        )
    data = np.load(_LEBEDEV_NPZ)
    points = np.asarray(data[f"r{i}"], dtype=np.float64)
    weights = np.asarray(data[f"w{i}"], dtype=np.float64)
    return points, weights


def lebedev_tensors(
    num: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Lebedev grid as torch tensors."""
    points, weights = lebedev_quadrature(num)
    grid_u = torch.as_tensor(points, device=device, dtype=dtype)
    grid_w = torch.as_tensor(weights, device=device, dtype=dtype)
    return grid_u, grid_w


def recommend_max_frequency(lebedev_num: int) -> float:
    """Return the SI-recommended ``b_max`` for a Lebedev order, if tabulated."""
    if lebedev_num not in LEBEDEV_FREQUENCY_LOOKUP:
        raise KeyError(
            f"No recommended max_frequency for lebedev_num={lebedev_num}. "
            f"Known orders: {sorted(LEBEDEV_FREQUENCY_LOOKUP)}"
        )
    return LEBEDEV_FREQUENCY_LOOKUP[lebedev_num]
