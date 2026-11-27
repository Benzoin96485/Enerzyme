"""Polynomial invariants of DPA4C degree-wise moments.

Degrees one and two pass through a bias-free residual channel map. The
readout then emits exact channel Grams, the Cartesian bispectrum, and the
projected quartic ``|Q v|^2``. There is no learned nonlinearity here.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..mp.equivariant.cg import gaunt
from ..mp.equivariant.body_order import bispectrum_contraction, vector_tensor_invariants
from .geometry import (
    degree_offsets,
    derive_bispectrum_ranks,
    derive_degree_channels,
    enumerate_degree_triples,
    probe_entries,
)


def _orthonormal_columns(width: int, rank: int) -> Tensor:
    gaussian = torch.randn(width, rank, dtype=torch.float64)
    orthogonal, triangular = torch.linalg.qr(gaussian, mode="reduced")
    sign = torch.where(
        torch.diagonal(triangular) < 0,
        torch.tensor(-1.0, dtype=orthogonal.dtype),
        torch.tensor(1.0, dtype=orthogonal.dtype),
    )
    return (orthogonal * sign.unsqueeze(0)).to(dtype=torch.get_default_dtype())


class InvariantReadout(nn.Module):
    """Contract flat moments ``(N, S)`` into scalar blocks, excluding masses."""

    def __init__(self, channels: int, lmax: int) -> None:
        super().__init__()
        self.degree_channels = derive_degree_channels(channels, lmax)
        self.ranks = derive_bispectrum_ranks(self.degree_channels)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.offsets = degree_offsets(self.degree_channels)
        self.triples = enumerate_degree_triples(lmax)

        self.alignment = nn.ModuleList()
        for width in self.degree_channels[1:3]:
            layer = nn.Linear(width, width, bias=False)
            nn.init.zeros_(layer.weight)
            self.alignment.append(layer)

        gram_index = []
        gram_offdiag = []
        gram_offsets = [0]
        for width in self.degree_channels[1:]:
            row, column = torch.triu_indices(width, width)
            gram_index.append(row * width + column)
            gram_offdiag.append(row != column)
            gram_offsets.append(gram_offsets[-1] + int(row.shape[0]))
        self.register_buffer("gram_index", torch.cat(gram_index).long())
        self.register_buffer("gram_offdiag", torch.cat(gram_offdiag))
        self.gram_offsets = tuple(gram_offsets)

        self.probes = nn.ModuleList()
        for width, rank in zip(self.degree_channels[1:], self.ranks):
            if rank == width:
                self.probes.append(nn.Identity())
                continue
            layer = nn.Linear(width, rank, bias=False)
            basis = _orthonormal_columns(width, rank)
            with torch.no_grad():
                layer.weight.copy_(basis.transpose(0, 1).to(dtype=layer.weight.dtype))
            self.probes.append(layer)

        probe_index_parts = []
        probe_scale_parts = []
        probe_offsets = [0]
        coupling_parts = []
        coupling_offsets = [0]
        for degrees in self.triples:
            coupling = gaunt(*degrees, dtype=torch.float64, convention="dpa4c")
            coupling_parts.append(coupling.flatten())
            coupling_offsets.append(coupling_offsets[-1] + coupling.numel())
            index, scale = probe_entries(degrees, self.ranks)
            probe_index_parts.append(torch.as_tensor(index, dtype=torch.long))
            probe_scale_parts.append(torch.as_tensor(scale, dtype=torch.float64))
            probe_offsets.append(probe_offsets[-1] + int(index.shape[0]))
        self.probe_offsets = tuple(probe_offsets)
        self.coupling_offsets = tuple(coupling_offsets)
        self.bispectrum_dim = probe_offsets[-1]
        self.quartic_dim = self.ranks[0] * self.ranks[1]
        self.register_buffer(
            "probe_index",
            torch.cat(probe_index_parts) if probe_index_parts else torch.zeros(0, dtype=torch.long),
        )
        self.register_buffer(
            "probe_scale",
            torch.cat(probe_scale_parts) if probe_scale_parts else torch.zeros(0),
        )
        self.register_buffer(
            "coupling",
            torch.cat(coupling_parts) if coupling_parts else torch.zeros(0),
        )

    @property
    def dim_out(self) -> int:
        """Scalar moments, Grams, bispectrum, and quartic."""
        return self.channels + int(self.gram_index.shape[0]) + self.bispectrum_dim + self.quartic_dim

    def forward(self, moments: Tensor) -> Tensor:
        blocks = self._split(moments)
        aligned = list(blocks)
        for degree, layer in enumerate(self.alignment, start=1):
            mixed = layer(blocks[degree])
            aligned[degree] = blocks[degree] + mixed
        projected = [layer(block) for layer, block in zip(self.probes, aligned[1:])]
        bispectrum, quartic = self._bispectrum(projected)
        pieces = [blocks[0][:, 0, :], *self._grams(aligned), *bispectrum, quartic]
        return torch.cat(pieces, dim=-1)

    def _split(self, moments: Tensor) -> List[Tensor]:
        blocks = []
        nodes = moments.shape[0]
        for degree, width in enumerate(self.degree_channels):
            start, stop = self.offsets[degree], self.offsets[degree + 1]
            blocks.append(moments[:, start:stop].reshape(nodes, 2 * degree + 1, width))
        return blocks

    def _grams(self, blocks: Sequence[Tensor]) -> List[Tensor]:
        index = self.gram_index.to(device=blocks[0].device)
        offdiag = self.gram_offdiag.to(device=blocks[0].device)
        scale = torch.ones_like(offdiag, dtype=blocks[0].dtype).masked_fill_(offdiag, math.sqrt(2.0))
        parts = []
        for degree, block in enumerate(blocks[1:], start=1):
            gram = torch.matmul(block.transpose(-1, -2), block)
            flat = gram.reshape(block.shape[0], self.degree_channels[degree] ** 2)
            start, stop = self.gram_offsets[degree - 1 : degree + 1]
            parts.append(flat.index_select(1, index[start:stop]) * scale[start:stop])
        return parts

    def _bispectrum(self, projected: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
        cubic_112, quartic = vector_tensor_invariants(projected[0], projected[1])
        index = self.probe_index.to(device=projected[0].device)
        scale = self.probe_scale.to(dtype=projected[0].dtype, device=projected[0].device)
        coupling = self.coupling.to(dtype=projected[0].dtype, device=projected[0].device)
        parts = []
        for triple_index, degrees in enumerate(self.triples):
            degree_1, degree_2, degree_3 = degrees
            if degrees == (1, 1, 2):
                full = cubic_112
            else:
                start, stop = self.coupling_offsets[triple_index : triple_index + 2]
                tensor = coupling[start:stop].reshape(
                    2 * degree_1 + 1,
                    2 * degree_2 + 1,
                    2 * degree_3 + 1,
                )
                full = bispectrum_contraction(
                    tensor,
                    projected[degree_1 - 1],
                    projected[degree_2 - 1],
                    projected[degree_3 - 1],
                )
            probe_start, probe_stop = self.probe_offsets[triple_index : triple_index + 2]
            parts.append(
                full.index_select(1, index[probe_start:probe_stop])
                * scale[probe_start:probe_stop]
            )
        return parts, quartic
