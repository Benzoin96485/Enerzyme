"""Radial bases for GRACE.

Foundation GRACE models map a Chebyshev expansion through a two-hidden-layer
MLP and a smooth cutoff (Lysogorskiy, Bochkarev, Drautz, 2025, Supplementary
Information A). The Finnis–Sinclair student uses a linear map on spherical
Bessel functions, the same ``l = 0`` basis as :class:`~enerzyme.models.layers.rbf.BesselRBFLayer`
with ``flavor="default"``. The cutoff is applied to the radial functions
themselves so they vanish at ``cutoff`` independently of the MLP bias.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor
from torch.nn import Linear, Module, Sequential, SiLU, Tanh


def powered_cosine_window(distance: Tensor, cutoff: float, order: int) -> Tensor:
    """``0.5 * (1 + cos(pi (r/rc)^p))`` on ``[0, rc)``, else 0.

    The derivative vanishes at ``r = rc`` because ``sin(pi) = 0``, so a hard
    mask at the cutoff does not kink the force.
    """
    if order < 1:
        raise ValueError(f"cutoff order must be >= 1, got {order}")
    ratio = (distance / cutoff).clamp(min=0)
    powered = ratio if order == 1 else ratio.pow(int(order))
    window = 0.5 * (1.0 + torch.cos(math.pi * powered))
    return torch.where(distance < cutoff, window, torch.zeros_like(distance))


def chebyshev_basis(distance: Tensor, n_basis: int, cutoff: float) -> Tensor:
    """Chebyshev polynomials ``T_0 ... T_{n-1}`` of ``2r/rc - 1``. Shape ``[E, n]``."""
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1, got {n_basis}")
    argument = 2.0 * (distance / cutoff) - 1.0
    columns: List[Tensor] = [torch.ones_like(argument)]
    if n_basis > 1:
        columns.append(argument)
    for _ in range(2, n_basis):
        columns.append(2.0 * argument * columns[-1] - columns[-2])
    return torch.stack(columns, dim=-1)


def spherical_bessel_basis(distance: Tensor, n_basis: int, cutoff: float) -> Tensor:
    """``sqrt(2/rc) sin(n π r / rc) / r`` for ``n = 1 ... n_basis``.

    Same radial functions as ``BesselRBF(flavor="default")``. The ``r → 0``
    limit ``sqrt(2/rc) * n π / rc`` is filled in explicitly.
    """
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1, got {n_basis}")
    frequencies = torch.arange(
        1, n_basis + 1, device=distance.device, dtype=distance.dtype
    )
    prefactor = math.sqrt(2.0 / cutoff)
    argument = distance.unsqueeze(-1) * (math.pi / cutoff) * frequencies
    safe = distance.unsqueeze(-1).clamp_min(1e-12)
    values = prefactor * torch.sin(argument) / safe
    small = distance < 1e-8
    if bool(small.any()):
        limit = prefactor * (math.pi / cutoff) * frequencies
        values = torch.where(small.unsqueeze(-1), limit.expand_as(values), values)
    return values


def _activation(name: str) -> Module:
    key = name.lower()
    if key == "silu":
        return SiLU()
    if key == "tanh":
        return Tanh()
    raise ValueError(f"Unknown radial activation {name!r}; expected 'silu' or 'tanh'")


class RadialEmbedding(Module):
    """Map a small radial basis to ``R_{nl}(r)`` with shape ``[E, n_rad, lmax+1]``.

    ``hidden <= 0`` is the linear Finnis–Sinclair map. A positive width builds
    the foundation-model MLP: basis → hidden → hidden → ``n_rad * (lmax+1)``.
    """

    def __init__(
        self,
        n_basis: int,
        n_rad: int,
        lmax: int,
        cutoff: float,
        cutoff_order: int,
        basis: str = "chebyshev",
        hidden: int = 64,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if n_rad < 1:
            raise ValueError(f"n_rad must be >= 1, got {n_rad}")
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        key = basis.lower()
        if key not in {"chebyshev", "sbessel"}:
            raise ValueError(
                f"Unknown radial basis {basis!r}; expected 'chebyshev' or 'sbessel'"
            )
        self.n_basis = int(n_basis)
        self.n_rad = int(n_rad)
        self.lmax = int(lmax)
        self.cutoff = float(cutoff)
        self.cutoff_order = int(cutoff_order)
        self.basis = key
        self.hidden = int(hidden)
        out = self.n_rad * (self.lmax + 1)
        if self.hidden > 0:
            act = _activation(activation)
            self.network: Module = Sequential(
                Linear(self.n_basis, self.hidden),
                act,
                Linear(self.hidden, self.hidden),
                _activation(activation),
                Linear(self.hidden, out),
            )
        else:
            self.network = Linear(self.n_basis, out)

    def forward(self, distance: Tensor) -> Tensor:
        if distance.numel() == 0:
            return distance.new_zeros((0, self.n_rad, self.lmax + 1))
        if self.basis == "chebyshev":
            raw = chebyshev_basis(distance, self.n_basis, self.cutoff)
        else:
            raw = spherical_bessel_basis(distance, self.n_basis, self.cutoff)
        window = powered_cosine_window(distance, self.cutoff, self.cutoff_order)
        embedded = self.network(raw).view(-1, self.n_rad, self.lmax + 1)
        return embedded * window.unsqueeze(-1).unsqueeze(-1)
