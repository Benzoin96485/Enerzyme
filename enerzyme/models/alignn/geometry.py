"""Directed line graph of an atomistic bond graph.

ALIGNN's line graph is the DGL line graph, not the receiver-triplet graph
used by M3GNet, CHGNet, and DPA3. Enerzyme stores ``idx_i`` as the
destination and ``idx_j`` as the source, with ``vij = R[source] - R[destination]``.
A line-graph edge runs from bond ``e1`` to bond ``e2`` when the destination
of ``e1`` is the source of ``e2``: a path ``a -> b -> c``. The angle sits at
``b``.

``backtracking=True`` keeps the pair of a bond with its reverse, which is
what ``dgl.line_graph`` does by default and what the released ALIGNN model
uses. The cosine of that pair is ``+1``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..mp.graph import path_pairs


def directed_line_graph(
    idx_i: Tensor,
    idx_j: Tensor,
    num_nodes: int,
    *,
    backtracking: bool = True,
    vij: Optional[Tensor] = None,
    atol: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Head-to-tail pairs ``(src_edge, dst_edge)`` in the atom-edge list.

    ``src_edge`` arrives at the shared atom. ``dst_edge`` leaves it.
    ``backtracking=False`` drops a pair only when ``dst_edge`` is the
    geometric reverse of ``src_edge`` (opposite ``vij``), so a distinct
    periodic image of the same atoms stays in the graph.
    """
    empty = idx_i.new_empty(0)
    if idx_i.numel() == 0:
        return empty, empty
    if idx_i.shape[0] != idx_j.shape[0]:
        raise ValueError(
            "idx_i and idx_j must index the same directed edges, "
            f"got {tuple(idx_i.shape)} and {tuple(idx_j.shape)}"
        )
    if int(num_nodes) < 1:
        raise ValueError(f"num_nodes must be positive when the graph has edges, got {num_nodes}")
    centers = idx_i.to(dtype=torch.long)
    sources = idx_j.to(dtype=torch.long)
    if int(centers.min()) < 0 or int(sources.min()) < 0:
        raise ValueError("edge indices must be non-negative")
    if int(centers.max()) >= int(num_nodes) or int(sources.max()) >= int(num_nodes):
        raise ValueError(
            "edge indices must be smaller than num_nodes "
            f"({num_nodes}), got max {int(centers.max())}, {int(sources.max())}"
        )

    pairs = path_pairs(centers, sources, int(num_nodes), order="atom")
    src_edge, dst_edge = pairs.first, pairs.second
    if backtracking:
        return src_edge, dst_edge
    if vij is None:
        raise ValueError(
            "backtracking=False needs vij so a distinct periodic image is not "
            "treated as the reverse bond"
        )
    if vij.ndim != 2 or vij.shape[0] != centers.shape[0] or vij.shape[-1] != 3:
        raise ValueError(
            f"vij must have one row per directed edge, got {tuple(vij.shape)}"
        )
    returns = centers[dst_edge] == sources[src_edge]
    gap = (vij[src_edge] + vij[dst_edge]).abs().amax(dim=-1)
    drop = returns & (gap <= float(atol))
    return src_edge[~drop], dst_edge[~drop]


def line_graph_cosine(vij: Tensor, src_edge: Tensor, dst_edge: Tensor) -> Tensor:
    """Cosine of the angle at the shared atom, in ``[-1, 1]``.

    ALIGNN stores ``r = R[destination] - R[source] = -vij`` and evaluates
    ``(-r_src) · r_dst``. That is ``-(vij_src · vij_dst)`` after normalizing.
    A bond and its reverse give cosine ``+1``.
    """
    if src_edge.numel() == 0:
        return vij.new_zeros(0)
    if src_edge.shape[0] != dst_edge.shape[0]:
        raise ValueError(
            "line-graph endpoints must have the same length, "
            f"got {src_edge.shape[0]} and {dst_edge.shape[0]}"
        )
    left = vij[src_edge]
    right = vij[dst_edge]
    dot = -(left * right).sum(dim=-1)
    denom = left.norm(dim=-1).clamp_min(1e-12) * right.norm(dim=-1).clamp_min(1e-12)
    return (dot / denom).clamp(-1.0, 1.0)
