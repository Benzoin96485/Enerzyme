"""Atomic moments, hyper moments, and one CAMP message-passing layer.

Tensors are stored as ``(channels, atoms or edges, 3, ..., 3)``. A channel mix
is a matrix on the leading axis. A path mix is a per-channel weighted sum of
that leading path axis. Neighbor sums divide by ``sqrt(avg_num_neighbors)``.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from .radial import SpeciesPairChebyshev, resolve_hidden_layers, silu_mlp
from .rules import atomic_paths, hyper_moment_rules
from ..mp.ops import scatter_sum


def cartesian_dyads(vectors: Tensor, max_rank: int, eps: float = 1e-12) -> Dict[int, Tensor]:
    """Polyadic products of the unit vector, rank 0 through ``max_rank``.

    A zero vector contributes a zero dyad of rank 1 and above. Rank 0 is 1,
    including on an empty edge list.
    """
    if max_rank < 0:
        raise ValueError(f"max_rank must be >= 0, got {max_rank}")
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    hat = torch.where(norm > eps, vectors / norm.clamp_min(eps), torch.zeros_like(vectors))
    dyads = {0: vectors.new_ones(vectors.shape[0])}
    if max_rank == 0:
        return dyads
    dyads[1] = hat
    current = hat
    for rank in range(2, max_rank + 1):
        current = current.unsqueeze(-1) * hat.reshape(
            *hat.shape[:-1], *([1] * (rank - 1)), 3
        )
        dyads[rank] = current
    return dyads


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



class ChannelMix(nn.Module):
    """Linear map on the channel axis of ``(C, ...)``."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_channels, n_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features: Tensor) -> Tensor:
        return torch.tensordot(self.weight, features, dims=1)


class PathMix(nn.Module):
    """Per-channel weighted sum of a leading path axis ``(P, C, ...)``."""

    def __init__(self, n_paths: int, n_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_paths, n_channels))
        bound = n_paths ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, features: Tensor) -> Tensor:
        return torch.einsum("pc,pc...->c...", self.weight, features)


class AtomicMoment(nn.Module):
    """Eqs. (5) and (6): neighbor sum of ``R(r) h_j ⊙ D(r_hat)``, then path mix."""

    def __init__(
        self,
        n_channels: int,
        n_species: int,
        max_v: int,
        max_feature_rank: int,
        avg_num_neighbors: float,
        degree: int,
        cutoff: float,
        envelope_p: int,
        radial_hidden: Sequence[int],
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.max_v = int(max_v)
        self.max_feature_rank = int(max_feature_rank)
        self.norm = float(avg_num_neighbors) ** 0.5
        self.radial = SpeciesPairChebyshev(
            self.n_channels, n_species, degree, cutoff, envelope_p
        )
        self.paths = {
            rank: atomic_paths(self.max_v, self.max_feature_rank, rank)
            for rank in range(self.max_v + 1)
        }
        self.radial_mlp = nn.ModuleDict()
        for rank, paths in self.paths.items():
            for feature_rank, dyad_rank in paths:
                self.radial_mlp[f"{rank}_{feature_rank}_{dyad_rank}"] = silu_mlp(
                    self.n_channels, radial_hidden, self.n_channels
                )
        self.path_mix = nn.ModuleDict(
            {
                str(rank): PathMix(len(paths), self.n_channels)
                for rank, paths in self.paths.items()
                if len(paths) > 1
            }
        )
        self.channel_mix = nn.ModuleDict(
            {str(rank): ChannelMix(self.n_channels) for rank in self.paths}
        )

    def forward(
        self,
        distance: Tensor,
        vectors: Tensor,
        zi: Tensor,
        zj: Tensor,
        receivers: Tensor,
        senders: Tensor,
        features: Dict[int, Tensor],
        n_atoms: int,
    ) -> Dict[int, Tensor]:
        radial = self.radial(distance, zi, zj)
        dyads = cartesian_dyads(vectors, self.max_v)
        moments: Dict[int, Tensor] = {}
        for rank, paths in self.paths.items():
            pieces = []
            for feature_rank, dyad_rank in paths:
                weight = self.radial_mlp[f"{rank}_{feature_rank}_{dyad_rank}"](radial)
                weight = weight.transpose(0, 1)
                feature = features[feature_rank].index_select(1, senders)
                contracted = contract_feature_dyad(
                    feature, dyads[dyad_rank], feature_rank, dyad_rank
                )
                view = weight.reshape(weight.shape + (1,) * rank)
                pieces.append(scatter_sum(view * contracted, receivers, n_atoms, dim=1) / self.norm)
            combined = pieces[0] if len(pieces) == 1 else self.path_mix[str(rank)](torch.stack(pieces))
            moments[rank] = self.channel_mix[str(rank)](combined)
        return moments


class HyperMoment(nn.Module):
    """Eqs. (7) and (8): self-contraction of atomic moments, then a path mix."""

    def __init__(self, n_channels: int, max_v: int, max_out_rank: int) -> None:
        super().__init__()
        if max_out_rank > max_v:
            raise ValueError(f"max_out_rank {max_out_rank} exceeds max_v {max_v}")
        self.max_v = int(max_v)
        self.max_out_rank = int(max_out_rank)
        self.rules = {
            rank: hyper_moment_rules(self.max_v, rank)
            for rank in range(self.max_out_rank + 1)
        }
        self.path_mix = nn.ModuleDict(
            {
                str(rank): PathMix(len(rules), n_channels)
                for rank, rules in self.rules.items()
                if len(rules) > 1
            }
        )

    def forward(self, moments: Dict[int, Tensor]) -> Dict[int, Tensor]:
        output: Dict[int, Tensor] = {}
        for rank, rules in self.rules.items():
            pieces = [contract_moments(moments, rule) for rule in rules]
            output[rank] = (
                pieces[0] if len(pieces) == 1 else self.path_mix[str(rank)](torch.stack(pieces))
            )
        return output


class CAMPLayer(nn.Module):
    """One message-passing step: channel mix, atomic moment, hyper moment, residual.

    The first layer receives scalars only and has no residual. The last layer
    emits scalars only. Intermediate layers keep ranks ``0 .. max_v``.
    """

    def __init__(
        self,
        n_channels: int,
        n_species: int,
        max_v: int,
        max_in_rank: int,
        max_out_rank: int,
        residual: bool,
        avg_num_neighbors: float,
        degree: int,
        cutoff: float,
        envelope_p: int,
        radial_hidden: Sequence[int],
    ) -> None:
        super().__init__()
        self.max_in_rank = int(max_in_rank)
        self.max_out_rank = int(max_out_rank)
        self.residual = bool(residual)
        self.pre_mix = nn.ModuleDict(
            {str(rank): ChannelMix(n_channels) for rank in range(self.max_in_rank + 1)}
        )
        self.atomic = AtomicMoment(
            n_channels,
            n_species,
            max_v,
            self.max_in_rank,
            avg_num_neighbors,
            degree,
            cutoff,
            envelope_p,
            radial_hidden,
        )
        self.hyper = HyperMoment(n_channels, max_v, self.max_out_rank)
        self.post_mix = nn.ModuleDict(
            {str(rank): ChannelMix(n_channels) for rank in range(self.max_out_rank + 1)}
        )
        self.residual_mix = nn.ModuleDict(
            {
                str(rank): ChannelMix(n_channels)
                for rank in range(min(self.max_in_rank, self.max_out_rank) + 1)
            }
            if self.residual
            else {}
        )

    def forward(
        self,
        distance: Tensor,
        vectors: Tensor,
        zi: Tensor,
        zj: Tensor,
        receivers: Tensor,
        senders: Tensor,
        features: Dict[int, Tensor],
        n_atoms: int,
    ) -> Dict[int, Tensor]:
        mixed = {rank: self.pre_mix[str(rank)](value) for rank, value in features.items()}
        moments = self.atomic(
            distance, vectors, zi, zj, receivers, senders, mixed, n_atoms
        )
        updated = {
            rank: self.post_mix[str(rank)](value)
            for rank, value in self.hyper(moments).items()
        }
        if self.residual:
            limit = min(self.max_in_rank, self.max_out_rank)
            for rank in range(limit + 1):
                updated[rank] = updated[rank] + self.residual_mix[str(rank)](mixed[rank])
        return updated
