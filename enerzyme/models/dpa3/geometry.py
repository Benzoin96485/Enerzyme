"""Angular pairs for the DPA3 line graph of order 2.

An angle is an ordered pair of edges that share a receiver, including the
pair of an edge with itself. That diagonal is part of the DPA3 cosine
matrix. GemNet triplets drop this pair and are a different graph.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class AnglePairs(NamedTuple):
    """Edges ``ij`` and ``ik`` of each angle at ``center``."""

    edge_ij: Tensor
    edge_ik: Tensor
    center: Tensor


def angle_pairs(
    idx_i: Tensor,
    dist: Tensor,
    num_nodes: int,
    angular_cutoff: float,
) -> AnglePairs:
    """Ordered pairs of edges shorter than ``angular_cutoff`` at one receiver.

    ``idx_i`` is the receiver of each directed edge. The returned indices
    point into that edge list. A receiver with one angular neighbor yields
    the single diagonal pair.
    """
    empty = idx_i.new_empty(0)
    blank = AnglePairs(empty, empty, empty)
    if idx_i.numel() == 0:
        return blank
    if dist.shape[0] != idx_i.shape[0]:
        raise ValueError("dist and idx_i must describe the same edges")
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive when the graph has edges")

    angular = torch.where(dist < angular_cutoff)[0]
    n_angular = int(angular.numel())
    if n_angular == 0:
        return blank

    receivers = idx_i[angular]
    counts = torch.bincount(receivers, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, dtype=torch.long, device=idx_i.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    perm = torch.argsort(receivers, stable=True)

    degree = counts[receivers]
    total = int(degree.sum().item())
    edge_pos = torch.repeat_interleave(torch.arange(n_angular, device=idx_i.device), degree)
    prefix = torch.cumsum(degree, dim=0) - degree
    slot = torch.arange(total, device=idx_i.device) - torch.repeat_interleave(prefix, degree)
    other_pos = perm[offsets[receivers[edge_pos]] + slot]

    edge_ij = angular[edge_pos]
    edge_ik = angular[other_pos]
    return AnglePairs(edge_ij, edge_ik, idx_i[edge_ij])
