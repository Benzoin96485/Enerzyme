"""SpinConv Core (Shuaibi et al., arXiv:2106.09575).

The Core maps a species embedding and a Gaussian bond basis to a scalar
atomic state. Edge messages are initialized from the distance block, then
updated by spin convolutions on the source-centered sphere. The species
table and the Gaussian centers stay outside. The last linear of the energy
embedding block is :class:`~enerzyme.models.layers.readout.SimpleReadout`.
Conservative forces are ``Fa = -∇E`` from the shared ``Force`` layer.

The released force-centric head paints three axis-aligned spheres and
averages random rotations. It is a direct force model and is not part of
this potential.

``idx_i_sr`` is the destination and ``idx_j_sr`` is the source.
``vij_sr = R[source] - R[destination]``.
"""

from __future__ import annotations

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.head import Head

from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from ..mp.ops import scatter_sum
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.rbf import GaussianRBFLayer, gaussian_expansion
from .geometry import symmetric_neighbor_mask, sphere_projection


class SpinConvEmbed(Embed):
    def __init__(
        self,
        dim_embedding,
        num_rbf,
        max_num_neighbors,
        act,
        distance_map,
        init_fc1,
        init_fc2,
        init_mix,
        pair_scale,
        gaussian_center,
        gaussian_coeff,
        cutoff,
        n_lat,
        n_long,
    ):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.n_lat = n_lat
        self.n_long = n_long
        self.act = act
        self.distance_map = distance_map
        self.init_fc1 = init_fc1
        self.init_fc2 = init_fc2
        self.init_mix = init_mix
        self.pair_scale = pair_scale
        self.register_buffer("gaussian_center", gaussian_center)
        self.register_buffer("gaussian_coeff", gaussian_coeff)

    def _check_inputs(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        Za: Tensor,
    ) -> None:
        num_nodes = int(atom_embedding.shape[0])
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must have shape (N, {self.dim_embedding}), "
                f"got {tuple(atom_embedding.shape)}"
            )
        if Za.shape != (num_nodes,):
            raise ValueError(f"Za must have shape ({num_nodes},), got {tuple(Za.shape)}")
        edges = int(idx_i_sr.shape[0]) if idx_i_sr.ndim == 1 else -1
        aligned = (
            idx_i_sr.ndim == 1
            and idx_j_sr.shape == idx_i_sr.shape
            and Dij_sr.shape == idx_i_sr.shape
            and vij_sr.shape == (edges, 3)
            and rbf.shape == (edges, self.num_rbf)
        )
        if not aligned:
            raise ValueError(
                "rbf, Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same "
                f"edges (rbf width {self.num_rbf}); got rbf {tuple(rbf.shape)}, "
                f"Dij_sr {tuple(Dij_sr.shape)}, vij_sr {tuple(vij_sr.shape)}, "
                f"idx_i_sr {tuple(idx_i_sr.shape)}, idx_j_sr {tuple(idx_j_sr.shape)}"
            )
        if edges and (
            int(idx_i_sr.min()) < 0
            or int(idx_j_sr.min()) < 0
            or int(idx_i_sr.max()) >= num_nodes
            or int(idx_j_sr.max()) >= num_nodes
        ):
            raise ValueError("edge indices are outside the atom range")

    def forward(self, inputs, ctx):
        self._check_inputs(
            inputs["atom_embedding"], ctx.rbf, ctx.dist, ctx.vec, ctx.receiver, ctx.sender, inputs["Za"]
        )
        num_nodes = int(inputs["atom_embedding"].shape[0])
        idx_i = ctx.receiver.long()
        idx_j = ctx.sender.long()
        distance = ctx.dist
        vij = ctx.vec
        basis = ctx.rbf
        edges = ctx
        if idx_i.numel():
            keep = symmetric_neighbor_mask(idx_i, idx_j, distance, num_nodes, self.max_num_neighbors)
            if not bool(keep.all()):
                edges = ctx.subset(keep)
                idx_i, idx_j = edges.receiver, edges.sender
                distance, vij, basis = edges.dist, edges.vec, edges.rbf

        if self.pair_scale is not None:
            scaled = self.pair_scale(distance, inputs["Za"][idx_j], inputs["Za"][idx_i])
            basis = gaussian_expansion(scaled, self.gaussian_center, self.gaussian_coeff)
        distance_embed = self.distance_map(basis)
        hidden = self.act(self.init_fc2(self.act(self.init_fc1(distance_embed))))
        messages = self.init_mix(hidden, inputs["atom_embedding"], idx_j, idx_i)
        projection = sphere_projection(
            vij, idx_i, idx_j, distance, self.cutoff, num_nodes, self.n_lat, self.n_long,
            pairs=edges.triplets("path"),
        )
        ctx.cache["spin_graph"] = (
            distance_embed,
            inputs["atom_embedding"],
            idx_i,
            idx_j,
            vij,
            distance,
            num_nodes,
            projection,
        )
        return BlockState(scalar=inputs["atom_embedding"], edge=messages)


def build_stack(core, built_layers: List[Module]) -> None:
    core.adopt_radius_stack(built_layers)
    from ..layers.readout import HierachicalNSEReadout, HierachicalReadout

    if any(isinstance(layer, (HierachicalReadout, HierachicalNSEReadout)) for layer in built_layers):
        raise ValueError(
            "SpinConv atom_feature has shape (N, mid_hidden_channels). "
            "Use SimpleReadout. HierachicalReadout expects (N, F, blocks)."
        )
    gaussians: list[GaussianRBFLayer] = []
    for layer in built_layers:
        if isinstance(layer, BaseAtomEmbedding):
            core.atom_embedding = layer
        elif isinstance(layer, GaussianRBFLayer) and getattr(layer, "flavor", None) == "spinconv":
            gaussians.append(layer)
    if len(gaussians) != 1:
        raise ValueError(
            "SpinConvCore requires exactly one GaussianRBF layer with flavor "
            f"'spinconv', found {len(gaussians)}"
        )
    radial = gaussians[0]
    if radial.apply_cutoff_fn:
        raise ValueError(
            "SpinConv GaussianRBF must set apply_cutoff_fn: false. "
            "The published expansion has no envelope."
        )
    if int(radial.num_rbf) != core.num_rbf:
        raise ValueError(f"GaussianRBF num_rbf ({radial.num_rbf}) != Core num_rbf ({core.num_rbf})")
    if abs(float(radial.cutoff_sr) - core.cutoff_sr) > 1e-4:
        raise ValueError(f"GaussianRBF cutoff_sr ({radial.cutoff_sr}) != Core cutoff_sr ({core.cutoff_sr})")
    scalar = getattr(radial, "basis_width_scalar", None)
    if scalar is None or abs(float(scalar) - core.basis_width_scalar) > 1e-6:
        raise ValueError(
            "GaussianRBF basis_width_scalar "
            f"({scalar}) must equal Core basis_width_scalar ({core.basis_width_scalar})"
        )
    if not torch.allclose(radial.center, core.embed.gaussian_center):
        raise ValueError("GaussianRBF centers do not match the SpinConv distance basis")
    core.radial_basis_function = radial
    embedding = getattr(core, "atom_embedding", None)
    if embedding is None:
        raise ValueError("SpinConvCore requires a RandomAtomEmbedding layer")
    if int(embedding.dim_embedding) != core.dim_embedding:
        raise ValueError(
            f"atom embedding width {embedding.dim_embedding} != dim_embedding {core.dim_embedding}"
        )
    if int(embedding.max_Za) != core.max_Za:
        raise ValueError(f"atom embedding max_Za {embedding.max_Za} != Core max_Za {core.max_Za}")


class SpinMessage(Block):
    def __init__(self, message, residual):
        super().__init__()
        self.message = message
        self.residual = residual

    def forward(self, state, ctx):
        messages = self.message(state.edge, *ctx.cache["spin_graph"])
        if self.residual:
            messages = messages + state.edge
        return BlockState(scalar=state.scalar, edge=messages)


class SpinHead(Head):
    def __init__(self, mixture, channels, divisor):
        super().__init__()
        self.mixture = mixture
        self.channels = channels
        self.divisor = divisor

    def forward(self, state, ctx, taps, inputs):
        receiver = ctx.cache["spin_graph"][2]
        if ctx.num_nodes == 0:
            aggregate = state.scalar.new_zeros((0, self.channels))
        else:
            aggregate = scatter_sum(state.edge, receiver, ctx.num_nodes) / self.divisor
        nodes = torch.arange(ctx.num_nodes, device=state.scalar.device)
        return {"atom_feature": self.mixture(aggregate, state.scalar, nodes, nodes)}
