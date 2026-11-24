"""Cartesian edge-direction bases for atomic moment and density expansions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn


def cartesian_dyads(vectors: Tensor, max_rank: int, eps: float = 1e-12) -> Dict[int, Tensor]:
    """Polyadic products of the unit vector, rank 0 through ``max_rank``.

    A zero vector contributes a zero dyad of rank 1 and above. Rank 0 is 1,
    including on an empty edge list.
    """
    if max_rank < 0:
        raise ValueError(f"max_rank must be >= 0, got {max_rank}")
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    hat = torch.where(norm > eps, vectors / norm.clamp_min(eps), torch.zeros_like(vectors))
    return cartesian_powers(hat, max_rank)


def cartesian_powers(vectors: Tensor, max_rank: int, weight: Tensor | None = None) -> Dict[int, Tensor]:
    """Ordered tensor powers of each vector, with an optional weight shared by all degrees."""
    count = vectors.shape[0]
    powers = {0: vectors.new_ones(count) if weight is None else weight.reshape(count)}
    if not max_rank:
        return powers
    current = vectors if weight is None else powers[0][:, None] * vectors
    powers[1] = current
    for rank in range(2, max_rank + 1):
        current = current.unsqueeze(-1) * vectors.reshape(count, *([1] * (rank - 1)), 3)
        powers[rank] = current
    return powers


Index = Tuple[int, int, int]


def multi_indices(lmax: int) -> List[Index]:
    """Nonnegative ``(lx, ly, lz)`` with ``lx + ly + lz <= lmax``, grouped by ``l``."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    indices: List[Index] = []
    for total in range(lmax + 1):
        for lx in range(total + 1):
            for ly in range(total - lx + 1):
                indices.append((lx, ly, total - lx - ly))
    return indices


def cartesian_monomials(unit: Tensor, powers: Tensor, max_power: int | None = None) -> Tensor:
    """Evaluate ``x^{lx} y^{ly} z^{lz}`` for every multi-index.

    ``unit`` has shape ``[E, 3]`` and ``powers`` has shape ``[A, 3]``.
    Multiplication keeps the sign of each Cartesian component. A zero-length
    edge (the zero vector) contributes 1 to ``l = 0`` and 0 above it.
    """
    if powers.ndim != 2 or powers.shape[-1] != 3:
        raise ValueError(f"powers must be [A, 3], got {tuple(powers.shape)}")
    n_angular = powers.shape[0]
    if unit.numel() == 0:
        return unit.new_zeros((0, n_angular))
    if unit.ndim != 2 or unit.shape[-1] != 3:
        raise ValueError(f"unit vectors must be [E, 3], got {tuple(unit.shape)}")
    powers = powers.to(device=unit.device)
    result = unit.new_ones(unit.shape[0], n_angular)
    if n_angular == 0:
        return result
    if max_power is None:
        max_power = int(powers.max().item())
    for axis, component in enumerate(unit.unbind(dim=-1)):
        axis_powers = [torch.ones_like(component)]
        for _ in range(max_power):
            axis_powers.append(axis_powers[-1] * component)
        values = torch.stack(axis_powers, dim=-1)
        result = result * values.index_select(1, powers[:, axis])
    return result


class CartesianAngularBasis(nn.Module):
    """Monomials through ``lmax`` and the shell layout of that list."""

    def __init__(self, lmax: int) -> None:
        super().__init__()
        indices = multi_indices(lmax)
        self.lmax = int(lmax)
        self.n_angular = len(indices)
        self.register_buffer("powers", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "l_index",
            torch.tensor([sum(item) for item in indices], dtype=torch.long),
        )
        bounds = []
        start = 0
        for total in range(self.lmax + 1):
            count = (total + 1) * (total + 2) // 2
            bounds.append((start, start + count))
            start += count
        self.register_buffer("shells", torch.tensor(bounds, dtype=torch.long))

    def forward(self, unit: Tensor) -> Tensor:
        return cartesian_monomials(unit, self.powers, self.lmax)
