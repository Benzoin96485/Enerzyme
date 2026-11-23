"""Equivariant orbital message used by EquiREANN.

Wu, Xia, Zhang, and Jiang, J. Phys. Chem. A 2024, 128, 11061, Eq. 5, and
``GetDensity.forward`` in https://github.com/bjiangch/EquiREANN ``reann/src/density.py``.
The geometric orbital is kept. Each iteration adds the neighbor atom's
aggregated orbital, multiplied by the same squared-cosine window that already
sits inside the geometric channels.
"""

from __future__ import annotations

import torch
from torch import Tensor


def add_neighbor_orbital(
    pair: Tensor,
    psi: Tensor,
    idx_j: Tensor,
    window: Tensor,
) -> Tensor:
    """Return ``pair_ij + fc(r_ij) * ψ_j``.

    ``pair`` and ``psi`` use the Cartesian channel order from
    :func:`enerzyme.models.reann.density.cartesian_angular`. ``idx_j`` is the
    neighbor that owns ``ψ``. The window has one value per edge.
    """
    if pair.ndim != 3 or psi.ndim != 3:
        raise ValueError(
            f"pair and psi must be rank-3 orbitals, got {tuple(pair.shape)} and {tuple(psi.shape)}"
        )
    if psi.shape[1:] != pair.shape[1:]:
        raise ValueError(
            f"psi channels {tuple(psi.shape[1:])} do not match pair {tuple(pair.shape[1:])}"
        )
    if window.reshape(-1).shape[0] != pair.shape[0]:
        raise ValueError(
            f"window has length {window.reshape(-1).shape[0]} but pair has {pair.shape[0]} edges"
        )
    if idx_j.shape[0] != pair.shape[0]:
        raise ValueError(
            f"idx_j has length {idx_j.shape[0]} but pair has {pair.shape[0]} edges"
        )
    scale = window.reshape(-1).to(dtype=pair.dtype, device=pair.device).unsqueeze(-1).unsqueeze(-1)
    return pair + psi.index_select(0, idx_j.long()) * scale
