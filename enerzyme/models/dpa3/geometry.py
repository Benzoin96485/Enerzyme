"""Angular pairs for the DPA3 line graph of order 2.

An angle is an ordered pair of edges that share a receiver, including the
pair of an edge with itself. That diagonal is part of the DPA3 cosine
matrix. GemNet triplets drop this pair and are a different graph.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from ..mp.graph import receiver_pairs


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
    if idx_i.numel() and dist.shape[0] != idx_i.shape[0]:
        raise ValueError("dist and idx_i must describe the same edges")
    if idx_i.numel() and num_nodes < 1:
        raise ValueError("num_nodes must be positive when the graph has edges")
    angular = torch.where(dist < angular_cutoff)[0]
    pairs = receiver_pairs(idx_i, num_nodes, angular, include_self=True)
    return AnglePairs(pairs.first, pairs.second, idx_i[pairs.first])
