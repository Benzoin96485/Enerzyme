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

from .geometry import (
    cached_couplings,
    degree_offsets,
    derive_bispectrum_ranks,
    derive_degree_channels,
    enumerate_degree_triples,
    packed_l2_to_stf,
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
        couplings = cached_couplings(lmax)
        for degrees, coupling in zip(self.triples, couplings):
            coupling_parts.append(torch.as_tensor(coupling.reshape(-1), dtype=torch.float64))
            coupling_offsets.append(coupling_offsets[-1] + int(coupling.size))
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
        parts = []
        for degree, block in enumerate(blocks[1:], start=1):
            gram = torch.matmul(block.transpose(-1, -2), block)
            flat = gram.reshape(block.shape[0], self.degree_channels[degree] ** 2)
            start, stop = self.gram_offsets[degree - 1 : degree + 1]
            scale = torch.ones(stop - start, dtype=block.dtype, device=block.device)
            scale[offdiag[start:stop]] = math.sqrt(2.0)
            parts.append(flat.index_select(1, index[start:stop]) * scale)
        return parts

    def _bispectrum(self, projected: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
        cubic_112, quartic = self._vector_tensor(projected[0], projected[1])
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
                full = self._contract(
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

    def _vector_tensor(self, vector: Tensor, packed: Tensor) -> Tuple[Tensor, Tensor]:
        nodes, _, rank_vector = vector.shape
        rank_tensor = packed.shape[-1]
        vectors = vector.transpose(1, 2)
        matrices = packed_l2_to_stf(packed.transpose(1, 2))
        contracted = torch.einsum("neij,nbi->nebj", matrices, vectors)
        cubic = torch.einsum("nai,nebi->nabe", vectors, contracted)
        cubic = cubic.reshape(nodes, rank_vector * rank_vector * rank_tensor)
        cubic = cubic * (-1.0 / math.sqrt(5.0))
        quartic = torch.sum(contracted * contracted, dim=-1).reshape(nodes, rank_tensor * rank_vector)
        return cubic, quartic

    @staticmethod
    def _contract(
        coupling: Tensor,
        value_1: Tensor,
        value_2: Tensor,
        value_3: Tensor,
    ) -> Tensor:
        nodes = value_1.shape[0]
        rank_1 = value_1.shape[-1]
        rank_2 = value_2.shape[-1]
        rank_3 = value_3.shape[-1]
        full = torch.einsum("abc,naA,nbB,ncC->nABC", coupling, value_1, value_2, value_3)
        return full.reshape(nodes, rank_1 * rank_2 * rank_3)
