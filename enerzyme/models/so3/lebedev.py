"""Lebedev quadrature and S² grid projectors for equivariant FFNs.

Lebedev rules are vendored from deepmd-kit (LGPL-3.0-or-later), originally from
John Burkardt's sphere Lebedev dataset. Used by DPA4 EMFA FFN (arXiv:2606.02419)
and available for other SO(3) grid paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .spherical_harmonics import spherical_harmonics

LEBEDEV_RULES_FILE = Path(__file__).with_name("data") / "lebedev_rules.npz"

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
    """Load Cartesian unit points ``(A, 3)`` and weights ``(A,)`` (sum to 1)."""
    if not isinstance(precision, (int, np.integer)) or isinstance(precision, bool):
        raise TypeError(
            f"`precision` must be an integer, got {type(precision).__name__}"
        )
    if not LEBEDEV_RULES_FILE.exists():
        raise FileNotFoundError(
            f"Lebedev quadrature data file is missing: {LEBEDEV_RULES_FILE}"
        )
    rule_key = f"{int(precision):03d}"
    with np.load(LEBEDEV_RULES_FILE) as rules:
        point_key = f"points_{rule_key}"
        weight_key = f"weights_{rule_key}"
        if point_key not in rules or weight_key not in rules:
            raise ValueError(
                f"Lebedev rule with precision {precision} is not packaged; "
                f"available precisions: {sorted(LEBEDEV_PRECISION_TO_NPOINTS)}"
            )
        points = np.asarray(rules[point_key], dtype=np.float64)
        weights = np.asarray(rules[weight_key], dtype=np.float64)
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
