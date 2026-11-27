"""ALIGNN Core (Choudhary and DeCost, npj Comput. Mater. 2021).

The Core maps a species embedding and a Gaussian bond basis to a scalar
atomic state. Bond-angle cosines are expanded inside the Core because they
live on the directed line graph, which is part of the interaction. The
species table, the distance Gaussian, the energy head, and ``Fa = -∇E``
stay outside.

``idx_i_sr`` is the destination and ``idx_j_sr`` is the source.
``vij_sr = R[source] - R[destination]``. The neighbor list is the graph
cutoff. The released force field does not put a smooth envelope on the
messages; ``multiply_envelope`` scales only the initial bond embedding by
:func:`enerzyme.models.mp.basis.envelope.polynomial_envelope`, the same optional
cutoff as ``ALIGNNAtomWise``.
"""

from __future__ import annotations

from ..mp.block import BlockState
from ..mp.core import Embed

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ..mp.basis.envelope import polynomial_envelope
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import BaseRBF, GaussianRBFLayer
from .geometry import line_graph_cosine


def _norm_name(value: Optional[str]) -> Optional[str]:
    if value in (None, "none", "null"):
        return None
    if value not in ("layer", "batch"):
        raise ValueError(f"norm must be 'layer', 'batch', or null, got {value!r}")
    return str(value)


class ALIGNNEmbed(Embed):
    def __init__(
        self,
        dim_embedding,
        num_rbf,
        cutoff_sr,
        alignn_layers,
        update_order,
        backtracking,
        multiply_envelope,
        envelope_exponent,
        atom_mlp,
        edge_mlp,
        angle_mlp,
        angle_centers,
        angle_coeff,
    ):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr
        self.alignn_layers = alignn_layers
        self.update_order = update_order
        self.backtracking = backtracking
        self.multiply_envelope = multiply_envelope
        self.envelope_exponent = envelope_exponent
        self.atom_mlp = atom_mlp
        self.edge_mlp = edge_mlp
        self.angle_mlp = angle_mlp
        self.register_buffer("angle_centers", angle_centers)
        self.register_buffer("angle_coeff", angle_coeff)

    def _embed(self, atom_embedding: Tensor, rbf: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
        atom = self.atom_mlp(atom_embedding)
        bond = self.edge_mlp(rbf)
        if self.multiply_envelope:
            envelope = polynomial_envelope(distance, self.cutoff_sr, self.envelope_exponent)
            bond = bond * envelope.unsqueeze(-1)
        return atom, bond

    def _angles(self, ctx):
        src_edge, dst_edge = ctx.triplets("line_graph", exclude_backtracking=not self.backtracking)
        cosine = line_graph_cosine(ctx.vec, src_edge, dst_edge)
        expanded = torch.exp(self.angle_coeff * (cosine.unsqueeze(-1) - self.angle_centers) ** 2)
        return src_edge, dst_edge, self.angle_mlp(expanded)

    def forward(self, inputs, ctx):
        if inputs["atom_embedding"].ndim != 2 or inputs["atom_embedding"].shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be "
                f"(N, {self.dim_embedding}), got {tuple(inputs["atom_embedding"].shape)}"
            )
        n_edges = int(ctx.receiver.shape[0])
        if ctx.sender.shape[0] != n_edges:
            raise ValueError(
                "idx_i_sr and idx_j_sr must share the edge count, " f"got {n_edges} and {ctx.sender.shape[0]}"
            )
        if ctx.dist.ndim != 1 or ctx.dist.shape[0] != n_edges:
            raise ValueError(f"Dij_sr must be ({n_edges},), got {tuple(ctx.dist.shape)}")
        if ctx.vec.ndim != 2 or ctx.vec.shape[0] != n_edges or ctx.vec.shape[-1] != 3:
            raise ValueError(f"vij_sr must be ({n_edges}, 3), got {tuple(ctx.vec.shape)}")
        if ctx.rbf.ndim != 2 or ctx.rbf.shape[0] != n_edges or ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf must be ({n_edges}, {self.num_rbf}), got {tuple(ctx.rbf.shape)}")
        n_atoms = int(inputs["atom_embedding"].shape[0])
        if torch.compiler.is_compiling():
            valid = (ctx.receiver >= 0) & (ctx.receiver < n_atoms) & (ctx.sender >= 0) & (ctx.sender < n_atoms)
            torch._assert_async(valid.all(), "edge indices are outside the atom range")
        elif n_edges and (
            int(ctx.receiver.max()) >= n_atoms
            or int(ctx.sender.max()) >= n_atoms
            or int(ctx.receiver.min()) < 0
            or int(ctx.sender.min()) < 0
        ):
            raise ValueError(f"edge indices must lie in 0..{n_atoms - 1}")
        atom, bond = self._embed(inputs["atom_embedding"], ctx.rbf, ctx.dist)
        angle = None
        if self.alignn_layers:
            source, target, angle = self._angles(ctx)
            ctx.cache["line_graph"] = source, target, self.update_order
        return BlockState(scalar=atom, equi=angle, edge=bond)


def build_stack(core, built_layers: list[Module]) -> None:
    core.adopt_radius_stack(built_layers)
    for candidate in built_layers:
        if isinstance(candidate, BaseAtomEmbedding):
            if int(candidate.dim_embedding) != core.dim_embedding:
                raise ValueError(
                    "Atom embedding width "
                    f"({candidate.dim_embedding}) must equal dim_embedding "
                    f"({core.dim_embedding})"
                )
            core.atom_embedding_layer = candidate
        if isinstance(candidate, RangeSeparationLayer):
            if abs(float(candidate.cutoff_sr) - core.cutoff_sr) > 1e-4:
                raise ValueError(
                    "RangeSeparation cutoff_sr "
                    f"({candidate.cutoff_sr}) must equal ALIGNN cutoff_sr "
                    f"({core.cutoff_sr})"
                )
    radial = [layer for layer in built_layers if isinstance(layer, BaseRBF)]
    gaussian = [
        layer
        for layer in radial
        if isinstance(layer, GaussianRBFLayer) and getattr(layer, "flavor", None) == "alignn"
    ]
    if len(radial) != 1 or len(gaussian) != 1:
        found = [f"{type(layer).__name__}:{getattr(layer, 'flavor', None)}" for layer in radial]
        raise ValueError(
            "ALIGNNCore requires exactly one GaussianRBF layer with flavor "
            f"'alignn' before the Core, found {found or 'none'}"
        )
    layer = gaussian[0]
    if int(layer.num_rbf) != core.num_rbf:
        raise ValueError(
            "GaussianRBF num_rbf " f"({layer.num_rbf}) must equal the Core num_rbf ({core.num_rbf})"
        )
    if abs(float(layer.cutoff_sr) - core.cutoff_sr) > 1e-4:
        raise ValueError(
            "GaussianRBF cutoff_sr " f"({layer.cutoff_sr}) must equal ALIGNN cutoff_sr ({core.cutoff_sr})"
        )
    core.radial_basis_function = layer


def alignn_step(layer, state, ctx):
    src, dst, order = ctx.cache["line_graph"]
    atom, bond, angle = layer(state.scalar, state.edge, state.equi, ctx.sender, ctx.receiver, src, dst, order)
    return BlockState(scalar=atom, equi=angle, edge=bond)


def gcn_step(layer, state, ctx):
    atom, bond = layer(state.scalar, state.edge, ctx.sender, ctx.receiver)
    return BlockState(scalar=atom, edge=bond)
