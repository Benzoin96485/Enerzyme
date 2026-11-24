"""DPA3 Core: repflow message passing on a line graph of order 2.

Zhang, Peng, Cai et al., arXiv:2506.01686. The Core maps an atom embedding
and the short-range graph to an invariant atom latent. Species embeddings,
the elemental energy bias, dataset codes, and the energy head stay outside.

The released DeePMD-kit descriptor is this order-2 stack (atoms, bonds,
angles). Order-3 dihedrals appear only in one scaling-law experiment and
are not part of this Core. DeePMD-kit is LGPL-3.0 and is not imported;
its checkpoints do not load.
"""

from __future__ import annotations

from ..mp.block import BlockState
from ..mp.core import Embed

import math
from typing import List

import torch
from torch import Tensor, nn

from ..mp.basis.envelope import exponential_switch, polynomial_transition
from ..layers import BaseAtomEmbedding, RangeSeparationLayer
from .geometry import AnglePairs


class DPA3Embed(Embed):
    def __init__(
        self,
        n_dim,
        a_dim,
        e_rcut,
        e_rcut_smth,
        a_rcut,
        a_rcut_smth,
        update_angle,
        edge_init_use_dist,
        switch,
        switch_decay,
        env_protection,
        stat_std,
        act,
        edge_embed,
        angle_embed,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.a_dim = a_dim
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.update_angle = update_angle
        self.edge_init_use_dist = edge_init_use_dist
        self.switch = switch
        self.switch_decay = switch_decay
        self.env_protection = env_protection
        self.stat_std = stat_std
        self.act = act
        self.edge_embed = edge_embed
        self.angle_embed = angle_embed

    def _switch(self, distance: Tensor, cutoff: float, cuton: float) -> Tensor:
        if self.switch == "polynomial":
            return polynomial_transition(distance, cutoff, cuton)
        return exponential_switch(distance, cutoff, cuton, self.switch_decay)

    def _embed_angles(self, pairs: AnglePairs, vij: Tensor, dist: Tensor) -> Tensor:
        assert self.angle_embed is not None
        if pairs.edge_ij.numel() == 0:
            cosine = dist.new_zeros(0, 1)
        else:
            left = vij[pairs.edge_ij]
            right = vij[pairs.edge_ik]
            span = (dist[pairs.edge_ij] * dist[pairs.edge_ik]).clamp(min=1e-12)
            cosine = (left * right).sum(dim=-1) / span
            cosine = (cosine.clamp(-1.0, 1.0) * (1.0 - 1e-6)).unsqueeze(-1)
            cosine = cosine / math.sqrt(math.pi)
        return self.angle_embed(cosine)

    def _check_edges(
        self,
        dist: Tensor,
        vij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        num_nodes: int,
    ) -> None:
        aligned = (
            dist.ndim == 1
            and vij.ndim == 2
            and vij.shape[-1] == 3
            and idx_i.ndim == 1
            and idx_j.ndim == 1
            and dist.shape[0] == vij.shape[0] == idx_i.shape[0] == idx_j.shape[0]
        )
        if not aligned:
            raise ValueError(
                "Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same edges "
                f"(got {tuple(dist.shape)}, {tuple(vij.shape)}, {tuple(idx_i.shape)}, {tuple(idx_j.shape)})"
            )
        if torch.compiler.is_compiling():
            torch._assert_async(((idx_i >= 0) & (idx_j >= 0)).all(), "edge indices must be non-negative")
            torch._assert_async(((idx_i < num_nodes) & (idx_j < num_nodes)).all(),
                                "edge index is outside the atom embedding")
            return
        if idx_i.numel() == 0:
            return
        if int(idx_i.min()) < 0 or int(idx_j.min()) < 0:
            raise ValueError("edge indices must be non-negative")
        if int(idx_i.max()) >= num_nodes or int(idx_j.max()) >= num_nodes:
            raise ValueError("edge index is outside the atom embedding")

    def forward(self, inputs, ctx):
        num_nodes = int(inputs["atom_embedding"].shape[0])
        if inputs["atom_embedding"].ndim != 2 or inputs["atom_embedding"].shape[-1] != self.n_dim:
            raise ValueError(
                f"atom_embedding must be (N, {self.n_dim}), got {tuple(inputs["atom_embedding"].shape)}"
            )
        self._check_edges(ctx.dist, ctx.vec, ctx.receiver, ctx.sender, num_nodes)

        edges = ctx.within(self.e_rcut)
        dist, vij = edges.dist, edges.vec
        idx_i, idx_j = edges.receiver, edges.sender
        edge_switch = self._switch(dist, self.e_rcut, self.e_rcut_smth)
        angle_switch = self._switch(dist, self.a_rcut, self.a_rcut_smth)
        denom = (dist + self.env_protection).clamp(min=1e-12)
        radial = (edge_switch / denom / self.stat_std).unsqueeze(-1)
        h = edge_switch.unsqueeze(-1) * vij / denom.square().unsqueeze(-1) / self.stat_std
        if self.edge_init_use_dist:
            edge = self.edge_embed(dist.unsqueeze(-1))
        else:
            edge = self.act(self.edge_embed(radial))

        if self.update_angle:
            first, second = edges.triplets("receiver", cutoff=self.a_rcut, include_self=True, strict=True)
            pairs = AnglePairs(first, second, idx_i[first])
            angle = self._embed_angles(pairs, vij, dist)
        else:
            pairs = AnglePairs(idx_i.new_empty(0), idx_i.new_empty(0), idx_i.new_empty(0))
            angle = edge.new_zeros(0, self.a_dim)
        angular_edge = dist < self.a_rcut
        node = self.act(inputs["atom_embedding"])
        ctx.cache["repflow"] = (
            h,
            edge_switch,
            angle_switch,
            idx_i,
            idx_j,
            pairs.edge_ij,
            pairs.edge_ik,
            pairs.center,
            angular_edge,
        )
        return BlockState(scalar=node, equi=angle, edge=edge)


def build_stack(core, built_layers: List[nn.Module]) -> None:
    core.adopt_radius_stack(built_layers)
    for layer in built_layers:
        if isinstance(layer, BaseAtomEmbedding) and int(layer.dim_embedding) != core.n_dim:
            raise ValueError(f"atom embedding width {layer.dim_embedding} must equal DPA3 n_dim {core.n_dim}")
        if isinstance(layer, RangeSeparationLayer):
            if float(layer.cutoff_sr) + 1e-5 < core.e_rcut:
                raise ValueError(
                    f"RangeSeparation cutoff_sr ({layer.cutoff_sr}) must be "
                    f">= DPA3 e_rcut ({core.e_rcut})"
                )


def repflow_step(block, state, ctx):
    node, edge, angle = block(state.scalar, state.edge, state.equi, *ctx.cache["repflow"])
    return BlockState(scalar=node, equi=angle, edge=edge)
