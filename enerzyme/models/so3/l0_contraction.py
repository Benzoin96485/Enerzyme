"""L0 (scalar) contractions of spherical harmonic coordinates (SPHC).

Adapted from So3krates-torch
(https://github.com/TCPUniLU/So3krates-torch, MIT license), matching mlff
``make_l0_contraction_fn``. Uses bundled ``cgmatrix.npz`` Clebsch–Gordan diagonals.
"""

from __future__ import annotations

import itertools as it
import os
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

_indx_fn = lambda x: int((x + 1) ** 2) if x >= 0 else 0

_CGMATRIX_PATH = os.path.join(os.path.dirname(__file__), "cgmatrix.npz")


def load_cgmatrix() -> np.ndarray:
    """Load the precomputed Clebsch–Gordan tensor from ``cgmatrix.npz``."""
    return np.load(_CGMATRIX_PATH)["cg"]


def init_clebsch_gordan_matrix(
    degrees: Sequence[int], l_out_max: int = 0
) -> np.ndarray:
    """Slice the CG tensor for the requested input degrees and ``l_out_max``."""
    l_in_max = max(degrees)
    l_in_min = min(degrees)
    offset_corr = _indx_fn(l_in_min - 1)
    cg_full = load_cgmatrix()
    return cg_full[
        offset_corr : _indx_fn(l_out_max),
        offset_corr : _indx_fn(l_in_max),
        offset_corr : _indx_fn(l_in_max),
    ]


class L0Contraction(Module):
    """Map SPHC ``[B, m_tot]`` to one scalar invariant per degree ``[B, |L|]``.

    For each degree segment, computes a CG-weighted sum of squared coefficients
    (paper / mlff L0 contraction).
    """

    def __init__(
        self,
        degrees: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        degrees = list(degrees)
        self.degrees: List[int] = degrees
        self.num_segments = len(degrees)

        cg_matrix = init_clebsch_gordan_matrix(
            degrees=list({0, *degrees}), l_out_max=0
        )
        cg_diag = np.diagonal(cg_matrix, axis1=1, axis2=2)[0]

        cg_rep = []
        degrees_np = np.array(degrees)
        unique_degrees, counts = np.unique(degrees_np, return_counts=True)
        for d, r in zip(unique_degrees, counts):
            block = cg_diag[_indx_fn(d - 1) : _indx_fn(d)]
            tiled = np.tile(block, r)
            cg_rep.append(tiled)
        cg_rep = np.concatenate(cg_rep)
        self.register_buffer("cg_rep", torch.tensor(cg_rep, dtype=dtype))

        segment_ids = list(
            it.chain(*[[n] * (2 * degrees[n] + 1) for n in range(len(degrees))])
        )
        self.register_buffer(
            "segment_ids", torch.tensor(segment_ids, dtype=torch.long)
        )

        m_tot = len(segment_ids)
        S = torch.zeros(m_tot, self.num_segments, dtype=dtype)
        S[torch.arange(m_tot), self.segment_ids] = 1.0
        self.register_buffer("segment_sum_matrix", S)

    def forward(self, sphc: Tensor) -> Tensor:
        """
        Parameters
        ----------
        sphc:
            Shape ``(B, m_tot)``.

        Returns
        -------
        Tensor
            Shape ``(B, len(degrees))``.
        """
        weighted = sphc * sphc * self.cg_rep
        return weighted @ self.segment_sum_matrix.to(dtype=weighted.dtype)
