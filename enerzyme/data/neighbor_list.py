from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor


def full_neighbor_list(N):
    idx = np.indices((N, N))
    idx_i = np.concatenate(idx[0][:, :N-1])
    idx_j = []
    for i in range(N):
        idx_int = idx[1][i]
        idx_int = idx_int[idx_int != i]
        idx_j.append(idx_int)
    idx_j = np.concatenate(idx_j)
    return idx_i, idx_j


def full_neighbor_list_torch(N: int, device: Optional[torch.device]=None) -> Tuple[Tensor, Tensor]:
    '''
    Same pairs and ordering as `full_neighbor_list`, built directly on `device`.
    '''
    idx_i = torch.arange(N, device=device).repeat_interleave(N - 1)
    r = torch.arange(N - 1, device=device).repeat(N)
    idx_j = r + (r >= idx_i)
    return idx_i, idx_j


@torch.no_grad()
def radius_neighbor_list(
    Ra: Tensor, cutoff: float, batch_seg: Optional[Tensor]=None, max_block_elements: int=1 << 24
) -> Tuple[Tensor, Tensor]:
    '''
    All ordered pairs (i, j), i != j, with |R_i - R_j| < cutoff (and the same batch
    segment if `batch_seg` is given), ordered by i and then j like the rows of
    `full_neighbor_list`. The O(N^2) distance test is done blockwise on the device of `Ra`.
    '''
    N = Ra.shape[0]
    device = Ra.device
    if N < 2:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    block = max(1, min(N, max_block_elements // N))
    cutoff2 = cutoff * cutoff
    R = Ra.detach()
    idx_i, idx_j = [], []
    for start in range(0, N, block):
        end = min(N, start + block)
        d2 = (R[start:end].unsqueeze(1) - R.unsqueeze(0)).square_().sum(-1)
        mask = d2 < cutoff2
        rows = torch.arange(end - start, device=device)
        mask[rows, rows + start] = False
        if batch_seg is not None:
            mask &= batch_seg[start:end].unsqueeze(1) == batch_seg.unsqueeze(0)
        nz = mask.nonzero()
        idx_i.append(nz[:, 0] + start)
        idx_j.append(nz[:, 1])
    return torch.cat(idx_i), torch.cat(idx_j)


class VerletNeighborList:
    '''
    Caches a `radius_neighbor_list` built with `cutoff + skin` and rebuilds it only when
    some atom moved more than skin / 2 since the last build, so the cached list always
    contains every pair within `cutoff`.
    '''
    def __init__(self, cutoff: float, skin: float=0.0, margin: float=1e-3) -> None:
        self.cutoff = cutoff
        self.skin = skin
        self.margin = margin
        self.R_ref: Optional[Tensor] = None
        self.batch_ref: Optional[Tensor] = None
        self.pairs: Optional[Tuple[Tensor, Tensor]] = None
        self.num_builds = 0

    def _needs_rebuild(self, Ra: Tensor, batch_seg: Optional[Tensor]) -> bool:
        if self.pairs is None or self.skin <= 0:
            return True
        if self.R_ref.shape != Ra.shape or self.R_ref.device != Ra.device:
            return True
        if (self.batch_ref is None) != (batch_seg is None):
            return True
        if batch_seg is not None and not torch.equal(self.batch_ref, batch_seg):
            return True
        max_disp2 = (Ra.detach() - self.R_ref).square().sum(-1).max()
        return bool(max_disp2 > (0.5 * self.skin) ** 2)

    def __call__(self, Ra: Tensor, batch_seg: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        if self._needs_rebuild(Ra, batch_seg):
            self.pairs = radius_neighbor_list(Ra, self.cutoff + self.skin + self.margin, batch_seg)
            self.R_ref = Ra.detach().clone()
            self.batch_ref = None if batch_seg is None else batch_seg.clone()
            self.num_builds += 1
        return self.pairs
