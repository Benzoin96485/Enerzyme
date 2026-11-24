"""
Bi-kNN + radius graph construction (fairchem AllScAIP layout: padded per-source neighbors).
Ported from fairchem `radius_graph_v2.py` (MIT), open-boundary and batched-molecule paths.
"""
from __future__ import annotations

import warnings
from itertools import accumulate
from typing import NamedTuple, Sequence

import torch

from ..mp.ops import safe_norm
from .radius_utils import (
    envelope_fn,
    hard_rank,
    soft_rank,
    soft_rank_low_mem,
)


class NeighborSlots(NamedTuple):
    src: torch.Tensor
    dst: torch.Tensor
    image: torch.Tensor
    src_rank: torch.Tensor
    dst_rank: torch.Tensor


class PreparedBiKNN(NamedTuple):
    sizes: tuple[int, ...]
    images: tuple[torch.Tensor, ...]
    slots: tuple[NeighborSlots, ...]
    pad_size: int


def _radius_geometry(pos, cell, image_id, cutoff, k, soft, sigmoid_scale, lse_scale,
                     use_low_mem, delta, compute_dist_pairwise):
    N = pos.size(0)
    M = image_id.size(0)
    src_pos = pos[:, None, :] + torch.mm(image_id, cell)[None, :, :]
    disp = src_pos[None, :, :, :] - pos[:, None, None, :]
    dist = safe_norm(disp, dim=-1)
    dist_pairwise = dist.min(dim=2)[0] if compute_dist_pairwise else None

    dist_T = dist.transpose(0, 1).contiguous()
    if soft:
        if use_low_mem:
            src_ranks = soft_rank_low_mem(dist.view(N, N * M), k, sigmoid_scale, delta).view(
                N, N, M
            )
            dst_ranks = (
                soft_rank_low_mem(dist_T.view(N, N * M), k, sigmoid_scale, delta)
                .view(N, N, M)
                .transpose(0, 1)
            )
        else:
            src_ranks = soft_rank(dist.view(N, N * M), sigmoid_scale).view(N, N, M)
            dst_ranks = (
                soft_rank(dist_T.view(N, N * M), sigmoid_scale).view(N, N, M).transpose(0, 1)
            )
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = lse_scale * torch.logsumexp(env / lse_scale, dim=0)
    else:
        src_ranks = hard_rank(dist.view(N, N * M)).view(N, N, M)
        dst_ranks = hard_rank(dist_T.view(N, N * M)).view(N, N, M).transpose(0, 1)
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = torch.amax(env, dim=0)
    env = env.masked_fill(dist == 0.0, 0.0)

    return disp, env, dist_pairwise


def _neighbor_slots(env):
    N, _, M = env.shape
    device = env.device
    index = torch.arange(N, device=device)[:, None]
    ranks = torch.arange(M * N, device=device, dtype=torch.long)[None, :]
    index1_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    src_argsort = torch.argsort(env.view(N, N * M), dim=1)
    index1_rank[index, src_argsort // M, src_argsort % M] = ranks
    index2_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    dst_argsort = torch.argsort(env.transpose(0, 1).reshape(N, N * M), dim=1)
    index2_rank[dst_argsort // M, index, dst_argsort % M] = ranks
    mask = env < 1.0
    index1, index2, index3 = torch.where(mask)
    index1_rank = index1_rank[index1, index2, index3]
    index2_rank = index2_rank[index1, index2, index3]
    return NeighborSlots(index1, index2, index3, index1_rank, index2_rank)


def build_radius_graph(
    pos: torch.Tensor,
    cell: torch.Tensor,
    image_id: torch.Tensor,
    cutoff: float,
    start_index: int,
    device: torch.device,
    k: int = 30,
    soft: bool = False,
    sigmoid_scale: float = 0.2,
    lse_scale: float = 0.1,
    use_low_mem: bool = False,
    delta: int = 20,
    compute_dist_pairwise: bool = True,
    slots: NeighborSlots | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """One system: return edge list, ranks, displacements, envelope, optional pairwise distances."""
    disp, env, dist_pairwise = _radius_geometry(
        pos, cell, image_id, cutoff, k, soft, sigmoid_scale, lse_scale, use_low_mem, delta, compute_dist_pairwise,
    )
    index1, index2, index3, index1_rank, index2_rank = slots if slots is not None else _neighbor_slots(env)
    disp = disp[index1, index2, index3]
    env = env[index1, index2, index3]
    index1 = index1 + start_index
    index2 = index2 + start_index
    return index1, index2, index1_rank, index2_rank, disp, env, dist_pairwise


def batched_radius_graph(
    pos_list: Sequence[torch.Tensor],
    cell_list: Sequence[torch.Tensor],
    image_id_list: Sequence[torch.Tensor],
    num_atoms: int,
    max_atoms: int | None,
    start_indices: Sequence[int] | torch.Tensor,
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    knn_use_low_mem: bool,
    knn_pad_size: int | None,
    cutoff: float,
    device: torch.device,
    compute_dist_pairwise: bool = True,
    prepared: PreparedBiKNN | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    n_g = len(pos_list)
    if isinstance(start_indices, torch.Tensor):
        t = start_indices.flatten().tolist()
        if len(t) == n_g + 1:
            starts = t[:-1]
        elif len(t) == n_g:
            starts = t
        else:
            raise ValueError(
                f"start_indices length {len(t)} incompatible with {n_g} graphs "
                f"(expected {n_g} starts or ptr of length {n_g + 1})."
            )
    else:
        starts = list(start_indices)
        if len(starts) == n_g + 1:
            starts = starts[:-1]
        elif len(starts) != n_g:
            raise ValueError(f"start_indices length mismatch: got {len(starts)}, need {n_g} or {n_g + 1}.")

    results = [
        build_radius_graph(
            pos,
            cell,
            image_id,
            cutoff,
            int(start_idx),
            device,
            knn_k,
            knn_soft,
            knn_sigmoid_scale,
            knn_lse_scale,
            knn_use_low_mem,
            compute_dist_pairwise=compute_dist_pairwise,
            slots=None if prepared is None else prepared.slots[g],
        )
        for g, (pos, cell, image_id, start_idx) in enumerate(zip(pos_list, cell_list, image_id_list, starts))
    ]
    index1 = torch.cat([r[0] for r in results])
    index2 = torch.cat([r[1] for r in results])
    index1_rank = torch.cat([r[2] for r in results])
    index2_rank = torch.cat([r[3] for r in results])
    disp = torch.cat([r[4] for r in results])
    env = torch.cat([r[5] for r in results])
    dist_blocks = [r[6] for r in results]

    if compute_dist_pairwise:
        dist_pairwise = torch.block_diag(*[d for d in dist_blocks if d is not None])
    else:
        dist_pairwise = None

    pad_size = knn_pad_size
    if prepared is not None:
        pad_size = prepared.pad_size
    elif pad_size is None:
        pad_size = int(max(index1_rank.max().item(), index2_rank.max().item())) + 1
    elif (index1_rank >= pad_size).any() or (index2_rank >= pad_size).any():
        warnings.warn(
            "knn_pad_size smaller than neighbor rank; trimming excess neighbors.",
            stacklevel=2,
        )
        keep = (index1_rank < pad_size) & (index2_rank < pad_size)
        index1 = index1[keep]
        index2 = index2[keep]
        index1_rank = index1_rank[keep]
        index2_rank = index2_rank[keep]
        disp = disp[keep]
        env = env[keep]

    if max_atoms is None:
        max_atoms = num_atoms

    padded_index = (
        torch.arange(max_atoms, device=device).view(-1, 1).expand(max_atoms, pad_size)
    )
    padded_rank = (
        torch.arange(pad_size, device=device).view(1, -1).expand(max_atoms, pad_size)
    )
    padded_disp = torch.zeros((max_atoms, pad_size, 3), device=device, dtype=disp.dtype)
    src_env = torch.full((max_atoms, pad_size), torch.inf, device=device, dtype=env.dtype)
    dst_env = torch.full((max_atoms, pad_size), torch.inf, device=device, dtype=env.dtype)
    edge_index = torch.stack([padded_index, padded_index], dim=0)
    src_index = torch.stack([padded_index, padded_rank], dim=0)
    dst_index = torch.stack([padded_index, padded_rank], dim=0)

    padded_disp[index1, index1_rank] = disp
    src_env[index1, index1_rank] = env
    dst_env[index2, index2_rank] = env
    edge_index[0, index1, index1_rank] = index1
    edge_index[1, index1, index1_rank] = index2
    src_index[0, index1, index1_rank] = index2
    src_index[1, index1, index1_rank] = index2_rank
    dst_index[0, index2, index2_rank] = index1
    dst_index[1, index2, index2_rank] = index1_rank

    if num_atoms < max_atoms:
        src_env[num_atoms:] = 0
        dst_env[num_atoms:] = 0

    return (
        dist_pairwise,
        padded_disp,
        src_env,
        dst_env,
        src_index,
        dst_index,
        edge_index,
    )


def _batch_images(Ra, batch_seg, cutoff, cell, pbc):
    device = Ra.device
    dtype = Ra.dtype
    num_graphs = int(batch_seg.max().item()) + 1
    natoms = torch.bincount(batch_seg.long(), minlength=num_graphs)
    if not pbc:
        identity = torch.zeros((1, 3), device=device, dtype=dtype)
        image_id_list = [identity] * num_graphs
    else:
        if cell is None:
            raise ValueError("pbc=True requires cell of shape (num_graphs, 3, 3)")
        image_id_list = []
        for g in range(num_graphs):
            c = cell[g]
            cross_a2a3 = torch.cross(c[1], c[2], dim=-1)
            cell_vol = torch.sum(c[0] * cross_a2a3, dim=-1, keepdim=True)
            inv_min_dist_a1 = safe_norm(cross_a2a3 / cell_vol, dim=-1)
            rep_a1 = int(torch.ceil(cutoff * inv_min_dist_a1).item())
            cross_a3a1 = torch.cross(c[2], c[0], dim=-1)
            inv_min_dist_a2 = safe_norm(cross_a3a1 / cell_vol, dim=-1)
            rep_a2 = int(torch.ceil(cutoff * inv_min_dist_a2).item())
            cross_a1a2 = torch.cross(c[0], c[1], dim=-1)
            inv_min_dist_a3 = safe_norm(cross_a1a2 / cell_vol, dim=-1)
            rep_a3 = int(torch.ceil(cutoff * inv_min_dist_a3).item())
            grid = torch.cartesian_prod(
                torch.arange(-rep_a1, rep_a1 + 1, device=device, dtype=dtype),
                torch.arange(-rep_a2, rep_a2 + 1, device=device, dtype=dtype),
                torch.arange(-rep_a3, rep_a3 + 1, device=device, dtype=dtype),
            )
            image_id_list.append(grid)

    return tuple(natoms.tolist()), tuple(image_id_list)


def _cells(Ra, cell, pbc, count):
    if pbc:
        return list(cell.unbind(0))
    return [torch.eye(3, device=Ra.device, dtype=Ra.dtype)] * count


def prepare_biknn(Ra, batch_seg, cell, hp):
    """Select discrete images, neighbor ranks and padding for this minibatch."""
    if hp.preprocess_on_cpu and Ra.device.type == "cuda":
        Ra, batch_seg = Ra.cpu(), batch_seg.cpu()
        if cell is not None:
            cell = cell.cpu()
    if Ra.dtype in (torch.float16, torch.bfloat16):
        Ra = Ra.float()
    sizes, images = _batch_images(Ra, batch_seg, hp.max_radius, cell, hp.pbc)
    cells = _cells(Ra, cell, hp.pbc, len(sizes))
    slots = []
    for pos, lattice, offsets in zip(Ra.split(sizes), cells, images):
        _, env, _ = _radius_geometry(
            pos, lattice, offsets, hp.max_radius, hp.knn_k, hp.knn_soft,
            hp.knn_sigmoid_scale, hp.knn_lse_scale, hp.knn_use_low_mem, 20, False,
        )
        slots.append(_neighbor_slots(env))
    pad_size = hp.knn_pad_size
    if pad_size is None:
        pad_size = int(torch.cat([rank for s in slots for rank in (s.src_rank, s.dst_rank)]).max()) + 1
    else:
        trimmed = []
        for table in slots:
            keep = (table.src_rank < pad_size) & (table.dst_rank < pad_size)
            if not keep.all():
                warnings.warn("knn_pad_size smaller than neighbor rank; trimming excess neighbors.", stacklevel=2)
            trimmed.append(NeighborSlots(*(value[keep] for value in table)))
        slots = trimmed
    return PreparedBiKNN(sizes, tuple(image.long() for image in images), tuple(slots), pad_size)


def biknn_radius_graph_from_batch(
    Ra: torch.Tensor,
    batch_seg: torch.Tensor,
    cutoff: float,
    knn_k: int,
    knn_soft: bool = False,
    knn_sigmoid_scale: float = 0.2,
    knn_lse_scale: float = 0.1,
    knn_use_low_mem: bool = False,
    knn_pad_size: int | None = None,
    max_atoms: int | None = None,
    compute_dist_pairwise: bool = True,
    cell: torch.Tensor | None = None,
    pbc: bool = False,
    prepared: PreparedBiKNN | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build fairchem-layout padded neighbor tensors from Enerzyme ``Ra`` / ``batch_seg``.

    When ``pbc`` is False (default), uses a single image (0,0,0) per system; ``cell`` is ignored.
    When ``pbc`` is True, ``cell`` must be ``(num_graphs, 3, 3)`` and image lists are built
    from cutoff and cell geometry (fairchem-style).
    """
    if prepared is None:
        sizes, image_id_list = _batch_images(Ra, batch_seg, cutoff, cell, pbc)
    else:
        sizes = prepared.sizes
        image_id_list = tuple(image.to(Ra) for image in prepared.images)
    pos_list = Ra.split(sizes)
    cell_list = _cells(Ra, cell, pbc, len(sizes))
    starts = tuple(accumulate(sizes, initial=0))[:-1]
    num_atoms_total = Ra.shape[0]
    return batched_radius_graph(
        pos_list,
        cell_list,
        image_id_list,
        num_atoms_total,
        max_atoms,
        starts,
        knn_k,
        knn_soft,
        knn_sigmoid_scale,
        knn_lse_scale,
        knn_use_low_mem,
        knn_pad_size,
        cutoff,
        Ra.device,
        compute_dist_pairwise=compute_dist_pairwise,
        prepared=prepared,
    )
