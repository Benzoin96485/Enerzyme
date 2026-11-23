"""Self pairs and unit directions shared by TorchMD-Net-style cores.

TorchMD-Net's neighbor list includes a self pair (``loop=True``) and stores
the displacement from the attention receiver toward the sender. Enerzyme's
shared neighbor list omits the self pair, so each Core inserts ``r = 0``
through the same radial basis the real edges used.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module


def unit_direction(vij: Tensor, dist: Tensor) -> Tensor:
    """``vij / ||vij||``. A zero distance stays the zero vector.

    Self pairs and any exact contact keep a zero direction, so the quotient
    does not send a gradient through a zero-length edge.
    """
    length = dist.reshape(-1)
    positive = length > 0
    hat = vij / length.clamp_min(1e-12).unsqueeze(-1)
    return torch.where(positive.unsqueeze(-1), hat, torch.zeros_like(vij))


def insert_missing_self_pairs(
    n_atoms: int,
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    dist: Tensor,
    rbf: Tensor,
    window: Tensor,
    *,
    enabled: bool,
    radial_basis: Optional[Module],
    num_rbf: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Append one ``r = 0`` pair for every atom that does not already have one.

    ``radial_basis`` must implement ``get_rbf``. The appended window is 1,
    which is the cosine cutoff at the origin. Atoms that already carry a
    self pair are left unchanged.
    """
    if not enabled or n_atoms == 0:
        return idx_i, idx_j, vij, dist, rbf, window
    present = torch.zeros(n_atoms, dtype=torch.bool, device=idx_i.device)
    if idx_i.numel():
        self_edge = idx_i == idx_j
        if self_edge.any():
            present[idx_i[self_edge]] = True
    missing = (~present).nonzero(as_tuple=False).flatten()
    if missing.numel() == 0:
        return idx_i, idx_j, vij, dist, rbf, window
    if radial_basis is None or not callable(getattr(radial_basis, "get_rbf", None)):
        raise RuntimeError(
            "include_self needs the pre-core radial basis "
            "(a layer that writes rbf and implements get_rbf). "
            "Call build(), or pass a self pair for every atom."
        )
    zeros = dist.new_zeros(missing.numel())
    self_rbf = radial_basis.get_rbf(zeros).to(dtype=rbf.dtype)
    if self_rbf.shape[-1] != int(num_rbf):
        raise ValueError(
            f"self-pair rbf width {self_rbf.shape[-1]} != num_rbf {num_rbf}"
        )
    self_window = window.new_ones(missing.numel())
    return (
        torch.cat((idx_i, missing)),
        torch.cat((idx_j, missing)),
        torch.cat((vij, vij.new_zeros(missing.numel(), 3))),
        torch.cat((dist, zeros)),
        torch.cat((rbf, self_rbf), dim=0),
        torch.cat((window, self_window)),
    )
