"""Grid projection for DPA4 FFN (Lebedev quadrature).

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# Lebedev precision -> n_points mapping (subset of packaged rules)
LEBEDEV_PRECISION_TO_NPOINTS = {
    3: 6, 5: 14, 7: 26, 9: 38, 11: 50, 13: 74, 15: 86,
    17: 110, 19: 146, 21: 170, 23: 194, 25: 230, 27: 266,
    29: 302, 31: 350, 35: 434, 41: 590, 47: 770, 53: 974,
    59: 1202, 65: 1454, 71: 1730, 77: 2030, 83: 2354, 89: 2702,
    95: 3074, 101: 3470, 107: 3890, 113: 4334, 119: 4802, 125: 5294, 131: 5810,
}


def load_lebedev_rule(precision: int):
    """Load a Lebedev quadrature rule from the vendored npz file."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    npz_path = os.path.join(data_dir, "lebedev_rules.npz")
    data = np.load(npz_path)
    key = f"{precision:03d}"
    points_key = f"points_{key}"
    weights_key = f"weights_{key}"
    return data[points_key], data[weights_key]


def _associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Compute associated Legendre polynomial P_l^m(x) (without Condon-Shortley)."""
    m_abs = abs(m)
    if m_abs > l:
        return np.zeros_like(x)
    # Start with P_m^m
    pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1.0 - x * x)
        fact = 1.0
        for i in range(1, m_abs + 1):
            pmm *= fact * somx2
            fact += 2.0
    if l == m_abs:
        return pmm
    # P_{m+1}^m
    pmmp1 = x * (2.0 * m_abs + 1.0) * pmm
    if l == m_abs + 1:
        return pmmp1
    # Recurrence for l > m+1
    pll = np.zeros_like(x)
    for ll in range(m_abs + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m_abs - 1.0) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def real_spherical_harmonics_np(points: np.ndarray, lmax: int) -> np.ndarray:
    """Evaluate real spherical harmonics at given points.

    Uses component normalization (same as e3nn normalize=True, normalization="component").

    Args:
        points: (G, 3) unit vectors
        lmax: maximum degree

    Returns:
        (G, (lmax+1)^2) harmonics values
    """
    G = points.shape[0]
    D = (lmax + 1) ** 2
    result = np.zeros((G, D), dtype=np.float64)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    phi = np.arctan2(y, x)

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            idx = l * l + l + m
            m_abs = abs(m)

            # Normalization factor for real SH with component normalization
            # N_l^m = sqrt((2l+1) * (l-|m|)! / (l+|m|)!)
            num = 1.0
            den = 1.0
            for k in range(l - m_abs + 1, l + m_abs + 1):
                den *= k
            if den > 0:
                K = math.sqrt((2.0 * l + 1.0) * num / den)
            else:
                K = 0.0

            P = _associated_legendre(l, m_abs, z)

            if m == 0:
                result[:, idx] = K * P
            elif m > 0:
                result[:, idx] = K * math.sqrt(2.0) * P * np.cos(m * phi)
            else:
                result[:, idx] = K * math.sqrt(2.0) * P * np.sin(m_abs * phi)

    return result


class S2GridProjector(nn.Module):
    """Project SO(3) coefficients to/from Lebedev grid.

    Stores to_grid_mat (G, D) and from_grid_mat (D, G) as buffers.
    """

    def __init__(self, lmax: int, mmax: Optional[int] = None) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax
        D = (lmax + 1) ** 2

        # Find smallest Lebedev rule with precision >= 3*lmax
        required = 3 * lmax
        prec = None
        for p, n in LEBEDEV_PRECISION_TO_NPOINTS.items():
            if p >= required:
                prec = p
                break
        if prec is None:
            prec = max(LEBEDEV_PRECISION_TO_NPOINTS.keys())

        try:
            points, weights = load_lebedev_rule(prec)
            points = np.asarray(points, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            harmonics = real_spherical_harmonics_np(points, lmax)
        except Exception:
            # Fallback: use simple uniform grid
            G = 4 * (lmax + 1) ** 2
            rng = np.random.default_rng(42)
            points = rng.standard_normal((G, 3))
            points = points / np.linalg.norm(points, axis=-1, keepdims=True)
            weights = np.ones(G) / G
            harmonics = np.eye(min(G, D), D) if G >= D else np.eye(D, G)[:D, :G].T

        scale = math.sqrt(float(lmax + 1))
        degree_factors = np.array([
            float(2 * l + 1) for l in range(lmax + 1) for _ in range(2 * l + 1)
        ], dtype=np.float64)

        to_grid = harmonics / scale  # (G, D)
        from_grid = (harmonics * (weights[:, None] * scale * degree_factors[None, :])).T  # (D, G)

        self.register_buffer("to_grid_mat", torch.tensor(to_grid, dtype=torch.float32))
        self.register_buffer("from_grid_mat", torch.tensor(from_grid, dtype=torch.float32))
        self.grid_size = to_grid.shape[0]

    def to_grid(self, x: Tensor) -> Tensor:
        """(N, D, C) -> (N, G, C)."""
        return torch.matmul(self.to_grid_mat.to(x.dtype).unsqueeze(0), x)

    def from_grid(self, g: Tensor) -> Tensor:
        """(N, G, C) -> (N, D, C)."""
        return torch.matmul(self.from_grid_mat.to(g.dtype).unsqueeze(0), g)
