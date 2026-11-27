"""Channel-aligned Clebsch–Gordan products and the GRACE recursion.

GRACE builds many-body basis functions by a sparse product
(Bochkarev, Lysogorskiy, Drautz, Phys. Rev. X 14, 021036, 2024): a linear
mix reduces the radial/embedding channels, then each output channel is the
Clebsch–Gordan product of that same channel on the two operands (e3nn
connection ``uuu``). The published recursion is

* order 2: ``A ⊗ A``
* order 3: ``(A ⊗ A) ⊗ A``
* order 4: ``(A ⊗ A) ⊗ (A ⊗ A)``

with a per-order angular cutoff. This is an original PyTorch implementation
of that contraction. It does not import or vendor TensorPotential.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Parameter

from ..mp.equivariant.product_ladder import DegreeMix, OrderSpec, ProductLadder, SparseCGProduct


class SpeciesChannelLinear(Module):
    """Species-dependent linear map, shared across ``m``.

    A bias is equivariant only for ``l = 0``. Pass ``bias=False`` for vectors.
    """

    def __init__(
        self, n_species: int, channels_in: int, channels_out: int, bias: bool = False
    ) -> None:
        super().__init__()
        if channels_in < 1 or channels_out < 1:
            raise ValueError(
                f"channel widths must be positive, got {channels_in} -> {channels_out}"
            )
        self.n_species = int(n_species)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.weight = Parameter(
            torch.empty(self.n_species, self.channels_out, self.channels_in)
        )
        self.bias = Parameter(torch.zeros(self.n_species, self.channels_out)) if bias else None
        torch.nn.init.xavier_uniform_(self.weight)
        self.weight.data.mul_(0.1)

    def forward(self, features: Tensor, species: Tensor) -> Tensor:
        vector = features.ndim == 3
        if not vector:
            if features.ndim != 2:
                raise ValueError(
                    f"species linear expected [N, C] or [N, C, M], got {tuple(features.shape)}"
                )
            features = features.unsqueeze(-1)
        weight = self.weight[species]
        mixed = torch.einsum("noi,nim->nom", weight, features)
        if self.bias is not None:
            mixed = mixed + self.bias[species].unsqueeze(-1)
        return mixed if vector else mixed.squeeze(-1)


class EquivariantIndicator(Module):
    """Linear message from every product order, species-dependent on the center.

    The second map is shared across species and sets the channel count expected
    by the next layer's one-particle basis. Degrees above ``lmax`` are dropped.
    """

    def __init__(
        self,
        order_channels: Sequence[int],
        order_lmax: Sequence[int],
        lmax: int,
        hidden: int,
        channels_out: int,
        n_species: int,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.order_lmax = [int(value) for value in order_lmax]
        self.order_channels = [int(value) for value in order_channels]
        self.hidden = int(hidden)
        self.channels_out = int(channels_out)
        self.species = ModuleList()
        self.project = ModuleList()
        for degree in range(self.lmax + 1):
            width = sum(
                channels
                for channels, limit in zip(self.order_channels, self.order_lmax)
                if limit >= degree
            )
            if width < 1:
                raise ValueError(
                    f"indicator l={degree} has no source channels; "
                    f"order Lmax={self.order_lmax}"
                )
            self.species.append(
                SpeciesChannelLinear(
                    n_species, width, self.hidden, bias=(degree == 0)
                )
            )
            self.project.append(Linear(self.hidden, self.channels_out, bias=False))

    def forward(
        self, orders: Sequence[Sequence[Tensor]], species: Tensor
    ) -> List[Tensor]:
        messages: List[Tensor] = []
        for degree, (species_mix, project) in enumerate(zip(self.species, self.project)):
            parts = [
                order[degree]
                for order, limit in zip(orders, self.order_lmax)
                if limit >= degree
            ]
            combined = torch.cat(parts, dim=1)
            hidden = species_mix(combined, species)
            messages.append(project(hidden.transpose(1, 2)).transpose(1, 2))
        return messages


class DensityProjection(Module):
    """Species-dependent map from every scalar basis channel to densities."""

    def __init__(
        self, order_channels: Sequence[int], n_densities: int, n_species: int
    ) -> None:
        super().__init__()
        self.order_channels = [int(value) for value in order_channels]
        width = sum(self.order_channels)
        self.linear = SpeciesChannelLinear(
            n_species, width, int(n_densities), bias=True
        )

    def forward(
        self, layers: Sequence[Sequence[Sequence[Tensor]]], species: Tensor
    ) -> Tensor:
        columns: List[Tensor] = []
        for orders in layers:
            for block in orders:
                # l = 0 has a single m component.
                columns.append(block[0].squeeze(-1))
        return self.linear(torch.cat(columns, dim=-1), species)
