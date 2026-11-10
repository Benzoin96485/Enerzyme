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

from .features import (
    cg_scale,
    clip_lmax,
    natural_parity,
    rms_norm,
    truncate_features,
)


class OrderSpec:
    """One product order: output channels, output ``Lmax``, optional input clip."""

    def __init__(
        self, channels: int, lmax: int, lmax_in: Optional[int] = None
    ) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        if lmax_in is not None and lmax_in < 0:
            raise ValueError(f"lmax_in must be >= 0, got {lmax_in}")
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.lmax_in = None if lmax_in is None else int(lmax_in)

    def __repr__(self) -> str:
        return (
            f"OrderSpec(channels={self.channels}, lmax={self.lmax}, "
            f"lmax_in={self.lmax_in})"
        )


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


class DegreeMix(Module):
    """Independent linear mix of each degree, optional per-atom RMS."""

    def __init__(
        self,
        lmax: int,
        channels_in: int,
        channels_out: int,
        *,
        normalize: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.normalize = bool(normalize)
        self.epsilon = float(epsilon)
        self.linears = ModuleList(
            [
                Linear(self.channels_in, self.channels_out, bias=False)
                for _ in range(self.lmax + 1)
            ]
        )

    def forward(self, blocks: Sequence[Tensor]) -> List[Tensor]:
        if len(blocks) < self.lmax + 1:
            raise ValueError(
                f"DegreeMix lmax {self.lmax} received only {len(blocks) - 1}"
            )
        mixed: List[Tensor] = []
        for degree, linear in enumerate(self.linears):
            transformed = linear(blocks[degree].transpose(1, 2)).transpose(1, 2)
            if self.normalize:
                transformed = rms_norm(transformed, self.epsilon)
            mixed.append(transformed)
        return mixed


class SparseCGProduct(Module):
    """Channel-aligned Clebsch–Gordan product of two equivariant features.

    Both operands use ``channels`` copies of the natural-parity irreps
    ``0e + 1o + ...``. Output degrees run through ``Lmax``. Paths that violate
    the triangle rule are omitted. A learnable scalar multiplies each
    ``(l1, l2, L)`` path; paths that land on the same ``L`` are normalized by
    ``1/sqrt(n_paths)`` so the initial scale stays O(1).
    """

    def __init__(
        self, channels: int, lmax_left: int, lmax_right: int, Lmax: int
    ) -> None:
        super().__init__()
        from e3nn.o3 import wigner_3j

        self.channels = int(channels)
        self.lmax_left = int(lmax_left)
        self.lmax_right = int(lmax_right)
        self.Lmax = int(Lmax)
        paths: List[tuple] = []
        counts = [0 for _ in range(self.Lmax + 1)]
        for l1 in range(self.lmax_left + 1):
            for l2 in range(self.lmax_right + 1):
                low = abs(l1 - l2)
                high = min(l1 + l2, self.Lmax)
                for out in range(low, high + 1):
                    if natural_parity(l1) * natural_parity(l2) != natural_parity(out):
                        continue
                    paths.append((l1, l2, out))
                    counts[out] += 1
        if not paths:
            raise ValueError(
                f"no coupling paths for lmax=({lmax_left}, {lmax_right}) -> {Lmax}"
            )
        self.paths = paths
        norms = []
        for l1, l2, out in paths:
            norms.append(cg_scale(out) / math_sqrt(counts[out]))
        self.register_buffer("path_norm", torch.tensor(norms, dtype=torch.float32))
        self.path_weight = Parameter(torch.ones(len(paths)))
        for index, (l1, l2, out) in enumerate(paths):
            coefficients = wigner_3j(l1, l2, out)
            expected = (2 * l1 + 1, 2 * l2 + 1, 2 * out + 1)
            if tuple(coefficients.shape) != expected:
                raise RuntimeError(
                    f"wigner_3j({l1}, {l2}, {out}) shape {tuple(coefficients.shape)} "
                    f"!= {expected}"
                )
            self.register_buffer(f"cg_{index}", coefficients)

    def forward(self, left: Sequence[Tensor], right: Sequence[Tensor]) -> List[Tensor]:
        if len(left) < self.lmax_left + 1 or len(right) < self.lmax_right + 1:
            raise ValueError("CG product received fewer degrees than it was built for")
        n_atoms = left[0].shape[0]
        channels = left[0].shape[1]
        if channels != self.channels or right[0].shape[1] != self.channels:
            raise ValueError(
                f"CG product channels {channels}, {right[0].shape[1]} != {self.channels}"
            )
        accumulated: List[Optional[Tensor]] = [None] * (self.Lmax + 1)
        weight = self.path_weight * self.path_norm.to(
            dtype=self.path_weight.dtype, device=self.path_weight.device
        )
        for index, (l1, l2, out) in enumerate(self.paths):
            coefficients = getattr(self, f"cg_{index}").to(
                dtype=left[l1].dtype, device=left[l1].device
            )
            coupled = torch.einsum(
                "nci,ncj,ijm->ncm", left[l1], right[l2], coefficients
            )
            coupled = coupled * weight[index].to(dtype=coupled.dtype)
            if accumulated[out] is None:
                accumulated[out] = coupled
            else:
                accumulated[out] = accumulated[out] + coupled
        output: List[Tensor] = []
        for degree, block in enumerate(accumulated):
            if block is None:
                output.append(
                    left[0].new_zeros((n_atoms, channels, 2 * degree + 1))
                )
            else:
                output.append(block)
        return output


def math_sqrt(value: int) -> float:
    return value ** 0.5


class ProductLadder(Module):
    """Recursive GRACE products up to order 4."""

    def __init__(self, orders: Sequence[OrderSpec], *, epsilon: float = 1e-6) -> None:
        super().__init__()
        if not 1 <= len(orders) <= 4:
            raise ValueError(
                f"GRACE recursion supports 1 to 4 product orders, got {len(orders)}"
            )
        self.orders = list(orders)
        self.epsilon = float(epsilon)
        if len(orders) >= 2:
            left = clip_lmax(orders[0].lmax, orders[1].lmax_in)
            self.mix_pair = DegreeMix(
                left, orders[0].channels, orders[1].channels, epsilon=epsilon
            )
            self.prod_pair = SparseCGProduct(
                orders[1].channels, left, left, orders[1].lmax
            )
        if len(orders) >= 3:
            left = clip_lmax(orders[1].lmax, orders[2].lmax_in)
            right = clip_lmax(orders[0].lmax, orders[2].lmax_in)
            self.mix_triple_left = DegreeMix(
                left, orders[1].channels, orders[2].channels, epsilon=epsilon
            )
            self.mix_triple_right = DegreeMix(
                right, orders[0].channels, orders[2].channels, epsilon=epsilon
            )
            self.prod_triple = SparseCGProduct(
                orders[2].channels, left, right, orders[2].lmax
            )
        if len(orders) >= 4:
            left = clip_lmax(orders[1].lmax, orders[3].lmax_in)
            self.mix_quad = DegreeMix(
                left, orders[1].channels, orders[3].channels, epsilon=epsilon
            )
            self.prod_quad = SparseCGProduct(
                orders[3].channels, left, left, orders[3].lmax
            )

    def forward(self, atomic: Sequence[Tensor]) -> List[List[Tensor]]:
        built: List[List[Tensor]] = [list(atomic)]
        if len(self.orders) == 1:
            return built
        pair_in = self.mix_pair(truncate_features(atomic, self.mix_pair.lmax))
        pair = self.prod_pair(pair_in, pair_in)
        built.append(pair)
        if len(self.orders) >= 3:
            left = self.mix_triple_left(truncate_features(pair, self.mix_triple_left.lmax))
            right = self.mix_triple_right(
                truncate_features(atomic, self.mix_triple_right.lmax)
            )
            built.append(self.prod_triple(left, right))
        if len(self.orders) >= 4:
            quad_in = self.mix_quad(truncate_features(pair, self.mix_quad.lmax))
            built.append(self.prod_quad(quad_in, quad_in))
        return built


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
