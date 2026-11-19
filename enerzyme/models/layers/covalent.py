"""Covalent radii and the Geodite short-range distance weight.

The weight is the attenuation ``y(r)`` in Reschützegger et al.,
J. Chem. Theory Comput. 2026 (Eq. 2). It multiplies a radial basis so the
learned features vanish at contact while a separate ZBL term carries the
wall. ``p`` and ``q`` are shared across species and constrained to be
greater than 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def covalent_radii_table(max_Za: int, dtype: torch.dtype | None = None) -> Tensor:
    """ASE covalent radii in Å, indexed by atomic number, length ``max_Za + 1``."""
    if int(max_Za) < 1:
        raise ValueError(f"max_Za must be >= 1, got {max_Za}")
    from ase.data import covalent_radii

    if dtype is None:
        dtype = torch.get_default_dtype()
    table = torch.tensor(covalent_radii, dtype=dtype)
    width = int(max_Za) + 1
    if table.numel() < width:
        pad = table[-1].repeat(width - table.numel())
        table = torch.cat([table, pad])
    return table[:width].contiguous()


def constrained_attenuation_exponents(p_raw: Tensor, q_raw: Tensor) -> tuple[Tensor, Tensor]:
    """Map unconstrained parameters to ``p, q > 1``.

    ``softplus(·, beta=0.5)`` matches the Geodite-MP initialization, where
    the stored ``p`` starts near 1.
    """
    p = F.softplus(p_raw, beta=0.5) + 1
    q = F.softplus(q_raw, beta=0.5) + 1
    return p, q


def covalent_distance_weight(
    distance: Tensor,
    radius_sum: Tensor,
    p: Tensor,
    q: Tensor,
) -> Tensor:
    """``y(r) = a t^q / (1 + t^{q-p} + a t^q)`` with ``t = r / (r_i + r_j)``.

    ``a = -2 (p + q - 2pq) / (p² + p + q² + q)``. For ``p, q > 1`` this is
    bounded in ``(0, 1)``, zero at contact, and one at long range.
    """
    t = distance / radius_sum.clamp_min(1e-6)
    # A tiny floor keeps a negative exponent from overflowing at r = 0.
    t = t.clamp(min=0).clamp_min(1e-12)
    a = -2 * (p + q - 2 * q * p) / (p.square() + p + q.square() + q)
    t_q = t.pow(q)
    return (a * t_q) / (1.0 + t.pow(q - p) + a * t_q)
