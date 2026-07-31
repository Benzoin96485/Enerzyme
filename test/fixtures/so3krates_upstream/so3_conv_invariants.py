"""Vendored So3krates-torch L0 contraction (import path adapted for fixtures).

Source: https://github.com/TCPUniLU/So3krates-torch (MIT)
"""

from __future__ import annotations

import itertools as it
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module

indx_fn = lambda x: int((x + 1) ** 2) if x >= 0 else 0

_CG = Path(__file__).resolve().parent / "cgmatrix.npz"


def load_cgmatrix():
    return np.load(_CG)["cg"]


def init_clebsch_gordan_matrix(degrees, l_out_max=0):
    l_in_max = max(degrees)
    l_in_min = min(degrees)
    offset_corr = indx_fn(l_in_min - 1)
    cg_full = load_cgmatrix()
    return cg_full[
        offset_corr : indx_fn(l_out_max),
        offset_corr : indx_fn(l_in_max),
        offset_corr : indx_fn(l_in_max),
    ]


class L0Contraction(Module):
    def __init__(self, degrees, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.degrees = degrees
        self.num_segments = len(degrees)

        cg_matrix = init_clebsch_gordan_matrix(
            degrees=list({0, *degrees}), l_out_max=0
        )
        cg_diag = np.diagonal(cg_matrix, axis1=1, axis2=2)[0]

        cg_rep = []
        degrees_np = np.array(degrees)
        unique_degrees, counts = np.unique(degrees_np, return_counts=True)
        for d, r in zip(unique_degrees, counts):
            block = cg_diag[indx_fn(d - 1) : indx_fn(d)]
            tiled = np.tile(block, r)
            cg_rep.append(tiled)

        cg_rep = np.concatenate(cg_rep)
        self.register_buffer(
            "cg_rep", torch.tensor(cg_rep, dtype=dtype, device=device)
        )

        segment_ids = list(
            it.chain(
                *[[n] * (2 * degrees[n] + 1) for n in range(len(degrees))]
            )
        )
        self.register_buffer(
            "segment_ids",
            torch.tensor(segment_ids, dtype=torch.long, device=device),
        )

        m_tot = len(segment_ids)
        S = torch.zeros(m_tot, self.num_segments, dtype=dtype, device=device)
        S[torch.arange(m_tot, device=device), self.segment_ids] = 1.0
        self.register_buffer("segment_sum_matrix", S)

    def forward(self, sphc: torch.Tensor) -> torch.Tensor:
        weighted = sphc * sphc * self.cg_rep
        return weighted @ self.segment_sum_matrix.to(weighted.dtype)
