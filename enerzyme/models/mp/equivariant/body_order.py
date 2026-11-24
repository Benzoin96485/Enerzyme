"""Body-order contractions of equivariant moments into lower-rank tensors."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor


def contract_feature_dyad(
    feature: Tensor, dyad: Tensor, feature_rank: int, dyad_rank: int
) -> Tensor:
    """Fully contract ``feature`` into the leading axes of ``dyad``.

    ``feature`` is ``(C, E, 3^v1)`` and ``dyad`` is ``(E, 3^v2)``. The result
    has rank ``v2 - v1``. Symmetric inputs make the choice of axes irrelevant.
    """
    if feature_rank > dyad_rank:
        raise ValueError(f"feature rank {feature_rank} exceeds dyad rank {dyad_rank}")
    out_rank = dyad_rank - feature_rank
    channels, edges = feature.shape[:2]
    if feature_rank == 0:
        if dyad_rank == 0:
            return feature * dyad.unsqueeze(0)
        return feature.reshape(channels, edges, *([1] * dyad_rank)) * dyad.unsqueeze(0)
    flat_feature = feature.reshape(channels, edges, 3 ** feature_rank)
    flat_dyad = dyad.reshape(edges, 3 ** feature_rank, 3 ** out_rank if out_rank else 1)
    mixed = torch.einsum("cej,ejk->cek", flat_feature, flat_dyad)
    if out_rank == 0:
        return mixed.squeeze(-1)
    return mixed.reshape(channels, edges, *([3] * out_rank))


def contract_moments(moments: Dict[int, Tensor], ranks: Sequence[int]) -> Tensor:
    """Contract every factor but one copy of the highest rank into that copy."""
    if len(ranks) == 1:
        return moments[int(ranks[0])]
    sink = max(range(len(ranks)), key=lambda index: ranks[index])
    sink_rank = int(ranks[sink])
    others = [int(ranks[index]) for index in range(len(ranks)) if index != sink]
    acc_rank = sum(others)
    out_rank = sink_rank - acc_rank
    if out_rank < 0:
        raise RuntimeError(f"hyper-moment ranks {tuple(ranks)} do not contract")
    sink_tensor = moments[sink_rank]
    channels, nodes = sink_tensor.shape[:2]
    spatial_out = 3 ** out_rank if out_rank else 1
    reduced = sink_tensor.reshape(channels, nodes, 3 ** acc_rank, spatial_out)
    consumed = 0
    for rank in others:
        width = 3 ** rank
        remaining = acc_rank - consumed
        reduced = reduced.reshape(
            channels, nodes, width, 3 ** (remaining - rank), spatial_out
        )
        other = moments[rank].reshape(channels, nodes, width)
        reduced = torch.einsum("cnabk,cna->cnbk", reduced, other)
        consumed += rank
    reduced = reduced.reshape(channels, nodes, spatial_out)
    if out_rank == 0:
        return reduced.squeeze(-1)
    return reduced.reshape(channels, nodes, *([3] * out_rank))
