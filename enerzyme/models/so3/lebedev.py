"""Lebedev quadrature tables and S² grid projectors.

Point/weight tables are vendored from Google e3x
(``e3x/so3/_lebedev_grids.npz``, Apache-2.0). Weights satisfy ``sum(w) == 1``.
Used by EFA (points only) and DPA4 EMFA FFN (``S2LebedevProjector``).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .spherical_harmonics import spherical_harmonics

LEBEDEV_GRIDS_FILE = Path(__file__).with_name("data") / "lebedev_grids.npz"

# Static map matching the packaged e3x table (precision → point count).
LEBEDEV_PRECISION_TO_NPOINTS: Dict[int, int] = {
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
    59: 1202,
    65: 1454,
    71: 1730,
    77: 2030,
    83: 2354,
    89: 2702,
    95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810,
}

# EFA SI-recommended max RoPE frequency vs Lebedev point count.
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
    if not LEBEDEV_GRIDS_FILE.exists():
        raise FileNotFoundError(
            f"Lebedev quadrature data file is missing: {LEBEDEV_GRIDS_FILE}"
        )
    data = np.load(LEBEDEV_GRIDS_FILE)
    return np.asarray(data["num"]), np.asarray(data["precision"])


def available_lebedev_nums() -> Tuple[int, ...]:
    nums, _ = _load_index()
    return tuple(int(n) for n in nums)


def available_lebedev_precisions() -> Tuple[int, ...]:
    _, precs = _load_index()
    return tuple(int(p) for p in precs)


def lebedev_quadrature(num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Lebedev points ``(M, 3)`` and weights ``(M,)`` for point count ``num``.

    Requires an exact packaged grid size (e3x / EFA convention).
    """
    nums, _ = _load_index()
    if num < int(nums.min()):
        raise ValueError(
            f"Lebedev num={num} is below the smallest available grid "
            f"({int(nums.min())}). Available: {available_lebedev_nums()}"
        )
    eligible = np.where(nums <= num)[0]
    i = int(eligible[np.argmax(nums[eligible])])
    if int(nums[i]) != num:
        raise ValueError(
            f"Lebedev num={num} is not available. Closest ≤num is "
            f"{int(nums[i])}. Available: {available_lebedev_nums()}"
        )
    data = np.load(LEBEDEV_GRIDS_FILE)
    points = np.asarray(data[f"r{i}"], dtype=np.float64)
    weights = np.asarray(data[f"w{i}"], dtype=np.float64)
    return points, weights


def lebedev_tensors(
    num: int,
    *,
    device: Optional[torch.device] = None,
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


def resolve_lebedev_precision(lmax: int) -> int:
    """Smallest packaged Lebedev precision with algebraic order ``>= 3 * lmax``."""
    required = 3 * int(lmax)
    for precision in sorted(LEBEDEV_PRECISION_TO_NPOINTS):
        if precision >= required:
            return int(precision)
    raise ValueError(
        f"No packaged Lebedev rule with precision >= {required} for lmax={lmax}; "
        f"available: {sorted(LEBEDEV_PRECISION_TO_NPOINTS)}"
    )


def load_lebedev_rule(precision: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load Cartesian unit points ``(A, 3)`` and weights ``(A,)`` by precision."""
    if not isinstance(precision, (int, np.integer)) or isinstance(precision, bool):
        raise TypeError(
            f"`precision` must be an integer, got {type(precision).__name__}"
        )
    nums, precs = _load_index()
    matches = np.where(precs == int(precision))[0]
    if matches.size == 0:
        raise ValueError(
            f"Lebedev rule with precision {precision} is not packaged; "
            f"available precisions: {sorted(LEBEDEV_PRECISION_TO_NPOINTS)}"
        )
    i = int(matches[0])
    data = np.load(LEBEDEV_GRIDS_FILE)
    points = np.asarray(data[f"r{i}"], dtype=np.float64)
    weights = np.asarray(data[f"w{i}"], dtype=np.float64)
    assert int(nums[i]) == LEBEDEV_PRECISION_TO_NPOINTS[int(precision)]
    return points, weights


class S2LebedevProjector(nn.Module):
    """Project packed SO(3) coefficients to/from a Lebedev S² grid.

    Uses e3nn-layout real spherical harmonics (component normalization) so the
    synthesis/analysis pair is consistent for DPA4-style grid FFNs.
    """

    def __init__(self, lmax: int, precision: Optional[int] = None) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        prec = int(precision) if precision is not None else resolve_lebedev_precision(
            self.lmax
        )
        points, weights = load_lebedev_rule(prec)
        degrees = list(range(self.lmax + 1))
        pts = torch.as_tensor(points, dtype=torch.float64)
        with torch.no_grad():
            harmonics = spherical_harmonics(
                pts,
                degrees,
                layout="e3nn",
                normalization="component",
                normalize_input=True,
            ).cpu().numpy().astype(np.float64)

        scale = math.sqrt(float(self.lmax + 1))
        degree_factors = np.array(
            [
                float(2 * degree + 1)
                for degree in range(self.lmax + 1)
                for _ in range(2 * degree + 1)
            ],
            dtype=np.float64,
        )
        to_grid = harmonics / scale
        from_grid = (
            harmonics * (weights[:, None] * scale * degree_factors[None, :])
        ).T

        self.register_buffer("to_grid_mat", torch.tensor(to_grid, dtype=torch.float32))
        self.register_buffer(
            "from_grid_mat", torch.tensor(from_grid, dtype=torch.float32)
        )
        self.grid_size = int(to_grid.shape[0])
        self.precision = prec

    def to_grid(self, x: Tensor) -> Tensor:
        """``(N, D, C)`` → ``(N, G, C)``."""
        return torch.matmul(self.to_grid_mat.to(dtype=x.dtype), x)

    def from_grid(self, grid: Tensor) -> Tensor:
        """``(N, G, C)`` → ``(N, D, C)``."""
        return torch.matmul(self.from_grid_mat.to(dtype=grid.dtype), grid)
