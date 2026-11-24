"""Cartesian edge-direction bases for atomic moment and density expansions."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def cartesian_dyads(vectors: Tensor, max_rank: int, eps: float = 1e-12) -> Dict[int, Tensor]:
    """Polyadic products of the unit vector, rank 0 through ``max_rank``.

    A zero vector contributes a zero dyad of rank 1 and above. Rank 0 is 1,
    including on an empty edge list.
    """
    if max_rank < 0:
        raise ValueError(f"max_rank must be >= 0, got {max_rank}")
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    hat = torch.where(norm > eps, vectors / norm.clamp_min(eps), torch.zeros_like(vectors))
    dyads = {0: vectors.new_ones(vectors.shape[0])}
    if max_rank == 0:
        return dyads
    dyads[1] = hat
    current = hat
    for rank in range(2, max_rank + 1):
        current = current.unsqueeze(-1) * hat.reshape(
            *hat.shape[:-1], *([1] * (rank - 1)), 3
        )
        dyads[rank] = current
    return dyads
