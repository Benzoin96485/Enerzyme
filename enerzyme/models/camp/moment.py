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
from ..mp.basis.cartesian import cartesian_dyads
from ..mp.equivariant.body_order import contract_feature_dyad, contract_moments
from ..mp.core import Embed
from ..mp.graph import GraphContext
from ..mp.block import Block, BlockState


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
        dyads: Dict[int, Tensor] | None = None,
    ) -> Dict[int, Tensor]:
        radial = self.radial(distance, zi, zj)
        if dyads is None:
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


class CAMPEmbed(Embed):
    """Prepare the chemical species pairs and dyads shared by every CAMP layer."""

    input_fields = ("atom_embedding", "Za")

    def __init__(self, width: int, lookup: Tensor, max_rank: int) -> None:
        super().__init__()
        self.dim_embedding = width
        self.max_Za = lookup.numel() - 1
        self.max_rank = max_rank
        self.register_buffer("index_of_z", lookup)

    def _species_index(self, atomic_numbers: Tensor) -> Tensor:
        species = atomic_numbers.long()
        if species.numel() and (
            int(species.min()) < 0 or int(species.max()) > self.max_Za
        ):
            raise ValueError(
                f"atomic numbers must lie in 0..max_Za ({self.max_Za}), "
                f"got {int(species.min())}..{int(species.max())}"
            )
        index = self.index_of_z[species.clamp(min=0)]
        if torch.any(index < 0):
            missing = species[index < 0].unique().tolist()
            raise ValueError(
                f"atomic numbers {missing} are outside the CAMP species table"
            )
        return index

    def forward(self, inputs, ctx: GraphContext) -> BlockState:
        atom_embedding = inputs["atom_embedding"]
        Za = inputs["Za"]
        Dij_sr, vij_sr = ctx.dist, ctx.vec
        receivers, senders = ctx.receiver, ctx.sender
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must be [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if Za.shape[0] != atom_embedding.shape[0]:
            raise ValueError(
                f"Za and atom_embedding must share the atom count, "
                f"got {Za.shape[0]} and {atom_embedding.shape[0]}"
            )
        n_atoms = int(atom_embedding.shape[0])
        n_edges = int(receivers.shape[0])
        if senders.shape[0] != n_edges or Dij_sr.shape[0] != n_edges or vij_sr.shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, vij_sr, receivers, and senders must share the edge count, "
                f"got {tuple(Dij_sr.shape)}, {tuple(vij_sr.shape)}, "
                f"{tuple(receivers.shape)}, {tuple(senders.shape)}"
            )
        if n_edges and vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must be [E, 3], got {tuple(vij_sr.shape)}")
        species = self._species_index(Za)
        dtype = atom_embedding.dtype
        if n_edges == 0:
            distance = atom_embedding.new_zeros(0)
            vectors = atom_embedding.new_zeros(0, 3)
            receivers = receivers.new_zeros(0)
            senders = senders.new_zeros(0)
            zi = species.new_zeros(0)
            zj = species.new_zeros(0)
        else:
            if (
                int(receivers.min()) < 0
                or int(senders.min()) < 0
                or int(receivers.max()) >= n_atoms
                or int(senders.max()) >= n_atoms
            ):
                raise ValueError("neighbor indices exceed the atom count")
            distance = Dij_sr.reshape(-1).to(dtype=dtype)
            vectors = vij_sr.to(dtype=dtype)
            zi = species.index_select(0, receivers)
            zj = species.index_select(0, senders)
        dyads = cartesian_dyads(vectors, self.max_rank)
        features = {0: atom_embedding.transpose(0, 1).contiguous()}
        return BlockState(scalar=atom_embedding, equi=features, edge=(distance, vectors, zi, zj, dyads))


class CAMPLayer(Block):
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

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        distance, vectors, zi, zj, dyads = state.edge
        features = state.equi
        mixed = {rank: self.pre_mix[str(rank)](value) for rank, value in features.items()}
        moments = self.atomic(
            distance, vectors, zi, zj, ctx.receiver, ctx.sender, mixed, ctx.num_nodes, dyads
        )
        updated = {
            rank: self.post_mix[str(rank)](value)
            for rank, value in self.hyper(moments).items()
        }
        if self.residual:
            limit = min(self.max_in_rank, self.max_out_rank)
            for rank in range(limit + 1):
                updated[rank] = updated[rank] + self.residual_mix[str(rank)](mixed[rank])
        return BlockState(scalar=updated[0].transpose(0, 1), equi=updated, edge=state.edge)
