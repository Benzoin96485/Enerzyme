"""Three-body index for M3GNet.

Enerzyme stores ``idx_i`` as the receiver and ``idx_j`` as the sender, with
``vij = R[sender] - R[receiver]``. A triplet is two such edges that share the
receiver. The angle is at that atom. Distinct periodic images of one neighbor
stay in the triplet list: they are different bonds even when ``idx_j`` matches.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import Tensor


_COS_EPS = 1e-7


class ThreeBodyPairs(NamedTuple):
    """``idx_ij`` receives the update. ``idx_ik`` supplies neighbor ``k``."""

    idx_ij: Tensor
    idx_ik: Tensor


def _receiver_csr(receivers: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor, Tensor]:
    counts = torch.bincount(receivers, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, dtype=torch.long, device=receivers.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    perm = torch.argsort(receivers, stable=True)
    return counts, offsets, perm


def receiver_triplets(
    idx_i: Tensor,
    dist: Tensor,
    num_nodes: int,
    threebody_cutoff: float,
) -> ThreeBodyPairs:
    """Ordered pairs of edges inside ``threebody_cutoff`` that share a receiver.

    Bonds with ``dist > threebody_cutoff`` are left out, matching matgl's
    line-graph prune. The returned indices address the original edge list.
    """
    empty = idx_i.new_empty(0)
    if idx_i.numel() == 0:
        return ThreeBodyPairs(empty, empty)
    if dist.shape[0] != idx_i.shape[0]:
        raise ValueError(
            f"dist rows {dist.shape[0]} != edge rows {idx_i.shape[0]}"
        )
    kept = torch.where(dist <= float(threebody_cutoff))[0]
    if kept.numel() == 0:
        return ThreeBodyPairs(empty, empty)

    device = idx_i.device
    centers = idx_i[kept]
    counts, offsets, perm = _receiver_csr(centers, num_nodes)
    n_kept = int(kept.numel())
    local = torch.empty(n_kept, dtype=torch.long, device=device)
    local[perm] = torch.arange(n_kept, device=device) - offsets[centers[perm]]
    n_other = counts[centers] - 1
    total = int(n_other.sum().item())
    if total == 0:
        return ThreeBodyPairs(empty, empty)

    src_local = torch.repeat_interleave(torch.arange(n_kept, device=device), n_other)
    prefix = torch.cumsum(n_other, dim=0) - n_other
    slot = torch.arange(total, device=device) - torch.repeat_interleave(prefix, n_other)
    self_local = local[src_local]
    other_local = slot + (slot >= self_local).to(slot.dtype)
    dst_local = perm[offsets[centers[src_local]] + other_local]
    return ThreeBodyPairs(kept[src_local], kept[dst_local])


def bond_angle(vij: Tensor, idx_ij: Tensor, idx_ik: Tensor) -> Tensor:
    """Angle at the receiver between ``vij[ij]`` and ``vij[ik]``, in ``(0, π)``.

    Cosine is clamped away from ±1, the same tolerance matgl uses before
    ``acos``.
    """
    if idx_ij.numel() == 0:
        return vij.new_zeros(0)
    left = vij[idx_ij]
    right = vij[idx_ik]
    norm_left = left.norm(dim=-1).clamp(min=1e-12)
    norm_right = right.norm(dim=-1).clamp(min=1e-12)
    cosine = (left * right).sum(dim=-1) / (norm_left * norm_right)
    cosine = cosine.clamp(min=-1.0 + _COS_EPS, max=1.0 - _COS_EPS)
    return torch.acos(cosine)
