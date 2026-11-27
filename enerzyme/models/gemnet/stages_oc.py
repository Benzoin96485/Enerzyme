"""GemNet-OC Core — interaction hierarchy on the GemNet message stack.

Gasteiger, Shuaibi, Sriram, Günnemann, Ulissi, Zitnick, and Das, TMLR 2022.
The Core owns the capped graphs and the interaction blocks. The species
embedding and the Gaussian radial basis stay outside
(``GaussianRBF`` with ``flavor: gemnet-oc``). Each block emits an atom
latent for :class:`~enerzyme.models.layers.readout.HierachicalReadout`.
Forces of this stack are ``Fa = -∇E``.

The published OC20 model predicts forces directly from edge states and
restricts quadruplets with atom tags. Neither choice is part of this Core.
fairchem checkpoints do not load.
"""

from __future__ import annotations

from ..mp.block import BlockState
from ..mp.core import Embed
from ..mp.message.triplet_graph import check_gemnet_edges

from dataclasses import dataclass
from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.nn import Module

from ..layers import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import GaussianRBFLayer
from .basis import legendre_outer, y_lm0
from ..mp.graph import GraphContext
from ..mp.quadruplets import dihedral_geometry, quadruplet_indices
from .geometry import (
    canonical_edge_indices,
    mixed_graph_triplets,
    mixed_triplet_indices,
    symmetrize_edges,
)
from .layers import BasisDownProjection


def _cutoff(name: str, value: float) -> float:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


class OCIndices(NamedTuple):
    capped: Tensor
    canonical: Tensor
    atom_edge: Tensor | None
    atom_atom: Tensor | None
    quad_edges: Tensor | None
    triplets: tuple
    atom_to_edge: tuple | None
    edge_to_atom: tuple | None
    quadruplets: tuple | None


def _main_graph(capped, canonical):
    def build():
        i, j, d, v, swap = symmetrize_edges(
            capped.receiver, capped.sender, capped.dist, capped.vec, canonical)
        return GraphContext(i, j, capped.num_nodes, vec=v, dist=d), swap

    return capped.cached("gemnet_oc_main", build)


@dataclass(frozen=True)
class GemNetOCGraphRequest:
    """Neighbor caps, periodic image matches and integer interaction tables."""

    main: tuple
    atom_edge: tuple | None
    atom_atom: tuple | None
    quadruplet: tuple | None
    use_a2e: bool
    use_e2a: bool

    @property
    def force_cache_key(self):
        return ("gemnet_oc_indices", self.main, self.atom_edge, self.atom_atom, self.quadruplet,
                self.use_a2e, self.use_e2a)

    def prepare(self, ctx):
        return ctx.cached(self.force_cache_key, lambda: self._indices(ctx))

    def neighbor_view(self, ctx, kind, indices):
        """Reuse a role's neighbor list when its cutoff and cap agree."""
        return ctx.subset(indices, key=("gemnet_oc", kind, getattr(self, kind)))

    def _indices(self, ctx):
        num_nodes = ctx.num_nodes
        if torch.compiler.is_compiling():
            torch._assert_async(((ctx.receiver >= 0) & (ctx.receiver < num_nodes)
                                 & (ctx.sender >= 0) & (ctx.sender < num_nodes)).all(),
                                "edge index is outside atom_embedding")
        elif ctx.receiver.numel() and (
            int(ctx.receiver.min()) < 0
            or int(ctx.sender.min()) < 0
            or int(ctx.receiver.max()) >= num_nodes
            or int(ctx.sender.max()) >= num_nodes
        ):
            raise ValueError("edge index is outside atom_embedding")

        capped_ids = ctx.closest_indices(*self.main)
        capped = self.neighbor_view(ctx, "main", capped_ids)
        canonical = canonical_edge_indices(capped.receiver, capped.sender, capped.vec)
        main, _ = _main_graph(capped, canonical)
        atom_edge = ctx.closest_indices(*self.atom_edge) if self.atom_edge else None
        atom_atom = ctx.closest_indices(*self.atom_atom) if self.atom_atom else None
        quad_edges = ctx.closest_indices(*self.quadruplet) if self.quadruplet else None
        ae = self.neighbor_view(ctx, "atom_edge", atom_edge) if atom_edge is not None else None
        return OCIndices(
            capped_ids, canonical, atom_edge, atom_atom, quad_edges,
            mixed_triplet_indices(main, main),
            mixed_triplet_indices(main, ae) if self.use_a2e else None,
            mixed_triplet_indices(ae, main) if self.use_e2a else None,
            quadruplet_indices(main, self.neighbor_view(ctx, "quadruplet", quad_edges), images=True)
            if quad_edges is not None else None,
        )


class GemNetOCEmbed(Embed):
    def __init__(
        self,
        quad_interaction,
        atom_edge_interaction,
        edge_atom_interaction,
        atom_interaction,
        dim_embedding,
        num_rbf,
        cutoff,
        cutoff_qint,
        cutoff_aeaint,
        cutoff_aint,
        max_neighbors,
        max_neighbors_qint,
        max_neighbors_aeaint,
        max_neighbors_aint,
        num_spherical,
        emb_size_cbf,
        edge_emb,
        mlp_rbf_tint,
        mlp_rbf_h,
        mlp_rbf_out,
        mlp_cbf_tint,
        mlp_rbf_qint,
        mlp_cbf_qint,
        mlp_sbf_qint,
        mlp_rbf_aeint,
        mlp_cbf_aeint,
        mlp_rbf_eaint,
        mlp_cbf_eaint,
        mlp_rbf_aint,
    ):
        super().__init__()
        self.quad_interaction = quad_interaction
        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.cutoff_qint = cutoff_qint
        self.cutoff_aeaint = cutoff_aeaint
        self.cutoff_aint = cutoff_aint
        self.max_neighbors = max_neighbors
        self.max_neighbors_qint = max_neighbors_qint
        self.max_neighbors_aeaint = max_neighbors_aeaint
        self.max_neighbors_aint = max_neighbors_aint
        self.graph_request = GemNetOCGraphRequest(
            (cutoff, max_neighbors),
            (cutoff_aeaint, max_neighbors_aeaint) if atom_edge_interaction or edge_atom_interaction else None,
            (cutoff_aint, max_neighbors_aint) if atom_interaction else None,
            (cutoff_qint, max_neighbors_qint) if quad_interaction else None,
            atom_edge_interaction, edge_atom_interaction,
        )
        self.num_spherical = num_spherical
        self.emb_size_cbf = emb_size_cbf
        self.radial_basis = None
        self.edge_emb = edge_emb
        self.mlp_rbf_tint = mlp_rbf_tint
        self.mlp_rbf_h = mlp_rbf_h
        self.mlp_rbf_out = mlp_rbf_out
        self.mlp_cbf_tint = mlp_cbf_tint
        self.mlp_rbf_qint = mlp_rbf_qint
        self.mlp_cbf_qint = mlp_cbf_qint
        self.mlp_sbf_qint = mlp_sbf_qint
        self.mlp_rbf_aeint = mlp_rbf_aeint
        self.mlp_cbf_aeint = mlp_cbf_aeint
        self.mlp_rbf_eaint = mlp_rbf_eaint
        self.mlp_cbf_eaint = mlp_cbf_eaint
        self.mlp_rbf_aint = mlp_rbf_aint

    def _basis(self, distance: Tensor, cutoff: float) -> Tensor:
        radial = self.radial_basis
        if radial is None:
            raise RuntimeError("GemNetOCCore.build has not adopted a gemnet-oc GaussianRBF")
        return radial.expansion(distance, cutoff)

    def _down(self, layer: BasisDownProjection, raw: Tensor) -> Tensor:
        n_angular = int(layer.weight.shape[0])
        return layer(raw.unsqueeze(1).expand(-1, n_angular, -1).contiguous())

    def _circular(
        self,
        layer: BasisDownProjection,
        distance: Tensor,
        angle: Tensor,
        cutoff: float,
    ) -> Tensor:
        if angle.numel() == 0:
            return distance.new_zeros(0, int(layer.weight.shape[-1]))
        projected = self._down(layer, self._basis(distance, cutoff))
        return (projected * y_lm0(angle, self.num_spherical).unsqueeze(1)).sum(dim=-1)

    def _quadruplet_basis(self, main, interaction, rbf_main, indices) -> dict:
        assert self.mlp_rbf_qint is not None
        assert self.mlp_cbf_qint is not None
        assert self.mlp_sbf_qint is not None
        n_sph = self.num_spherical * self.num_spherical
        radial_w4 = self._down(self.mlp_sbf_qint, rbf_main)
        rbf_q = self.mlp_rbf_qint(rbf_main)
        quads = dihedral_geometry(main, interaction, indices)
        if quads.edge_ca.numel() == 0:
            cbf_q = rbf_main.new_zeros(0, self.emb_size_cbf)
            angular4 = rbf_main.new_zeros(0, n_sph)
        else:
            cbf_q = self._circular(
                self.mlp_cbf_qint,
                interaction.dist[quads.int_local],
                quads.angle_abd,
                self.cutoff_qint,
            )
            angular4 = legendre_outer(quads.angle_cab, quads.dihedral, self.num_spherical)
        return {
            "rbf_q": rbf_q,
            "cbf_q": cbf_q,
            "radial_w4": radial_w4,
            "angular4": angular4,
            "edge_db": quads.edge_db,
            "edge_ca": quads.edge_ca,
        }

    def forward(self, inputs, ctx):
        check_gemnet_edges(ctx, self.num_rbf)
        if inputs["atom_embedding"].ndim != 2 or inputs["atom_embedding"].shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be "
                f"(N, {self.dim_embedding}), got {tuple(inputs["atom_embedding"].shape)}"
            )
        request = self.graph_request
        indices = request.prepare(ctx)
        main, swap = _main_graph(request.neighbor_view(ctx, "main", indices.capped), indices.canonical)
        main_i, main_j, main_d = main.receiver, main.sender, main.dist
        if self.atom_edge_interaction or self.edge_atom_interaction:
            ae = request.neighbor_view(ctx, "atom_edge", indices.atom_edge)
        if self.atom_interaction:
            a2a = request.neighbor_view(ctx, "atom_atom", indices.atom_atom)

        rbf_main = self._basis(main_d, self.cutoff)
        rbf_e2e = self.mlp_rbf_tint(rbf_main)
        rbf_h = self.mlp_rbf_h(rbf_main)
        rbf_out = self.mlp_rbf_out(rbf_main)
        radial_w3 = self._down(self.mlp_cbf_tint, rbf_main)
        trips = mixed_graph_triplets(main, main, indices.triplets)
        angular3 = y_lm0(trips.angle, self.num_spherical)

        quad_kwargs: dict = {}
        if self.quad_interaction:
            assert self.mlp_rbf_qint is not None
            assert self.mlp_cbf_qint is not None
            assert self.mlp_sbf_qint is not None
            interaction = request.neighbor_view(ctx, "quadruplet", indices.quad_edges)
            quad_kwargs = self._quadruplet_basis(main, interaction, rbf_main, indices.quadruplets)
        ae_kwargs: dict = {}
        if self.atom_edge_interaction:
            assert self.mlp_rbf_aeint is not None and self.mlp_cbf_aeint is not None
            a2e = mixed_graph_triplets(main, ae, indices.atom_to_edge)
            ae_kwargs.update(
                rbf_a2e=self.mlp_rbf_aeint(self._basis(ae.dist, self.cutoff_aeaint)),
                radial_a2e=self._down(self.mlp_cbf_aeint, rbf_main),
                angular_a2e=y_lm0(a2e.angle, self.num_spherical),
                a2e_sender=ae.sender,
                a2e_other=a2e.idx_other,
                a2e_edge=a2e.idx_edge,
            )
        if self.edge_atom_interaction:
            assert self.mlp_rbf_eaint is not None and self.mlp_cbf_eaint is not None
            e2a = mixed_graph_triplets(ae, main, indices.edge_to_atom)
            raw_ae = self._basis(ae.dist, self.cutoff_aeaint)
            ae_kwargs.update(
                rbf_e2a=self.mlp_rbf_eaint(rbf_main),
                radial_e2a=self._down(self.mlp_cbf_eaint, raw_ae),
                angular_e2a=y_lm0(e2a.angle, self.num_spherical),
                e2a_other=e2a.idx_other,
                e2a_edge=e2a.idx_edge,
                e2a_receiver=ae.receiver,
            )
        if self.atom_interaction:
            assert self.mlp_rbf_aint is not None
            ae_kwargs.update(
                rbf_a2a=self.mlp_rbf_aint(self._basis(a2a.dist, self.cutoff_aint)),
                a2a_i=a2a.receiver,
                a2a_j=a2a.sender,
            )

        h = inputs["atom_embedding"]
        m = self.edge_emb(h, rbf_main, main_i, main_j)
        args = (rbf_e2e, rbf_h, radial_w3, angular3, trips.idx_edge, trips.idx_other, swap, main_i, main_j)
        ctx.cache["gemnet_interaction"] = args, {**quad_kwargs, **ae_kwargs}
        ctx.cache["gemnet_readout"] = rbf_out, main_i
        return BlockState(scalar=h, edge=m)


def build_stack(core, built_layers: List[Module]) -> None:
    core.adopt_radius_stack(built_layers)
    if core.output_mode == "last":
        from ..layers.readout import HierachicalNSEReadout, HierachicalReadout

        hierarchical = any(
            isinstance(layer, (HierachicalReadout, HierachicalNSEReadout)) for layer in built_layers
        )
        if hierarchical:
            raise ValueError(
                "GemNet-OC output_mode='last' returns atom_feature of shape "
                "(N, F) for SimpleReadout. HierachicalReadout reads "
                "(N, F, num_blocks+1). Use output_mode='feature'."
            )
    found_atom = False
    gaussians: List[GaussianRBFLayer] = []
    for layer in built_layers:
        if isinstance(layer, BaseAtomEmbedding):
            if int(layer.dim_embedding) != core.dim_embedding:
                raise ValueError(
                    "Atom embedding width "
                    f"({layer.dim_embedding}) must equal dim_embedding "
                    f"({core.dim_embedding})"
                )
            found_atom = True
        elif isinstance(layer, RangeSeparationLayer):
            if abs(float(layer.cutoff_sr) - core.cutoff_sr) > 1e-4:
                raise ValueError(
                    "RangeSeparation cutoff_sr "
                    f"({layer.cutoff_sr}) must equal GemNet-OC cutoff_sr "
                    f"({core.cutoff_sr})"
                )
        elif isinstance(layer, GaussianRBFLayer) and getattr(layer, "flavor", None) == "gemnet-oc":
            gaussians.append(layer)
    if not found_atom:
        raise ValueError("GemNetOCCore requires an atom embedding before the Core")
    if len(gaussians) != 1:
        raise ValueError(
            "GemNetOCCore requires exactly one GaussianRBF with flavor "
            f"'gemnet-oc', found {len(gaussians)}"
        )
    radial = gaussians[0]
    if int(radial.num_rbf) != core.num_rbf:
        raise ValueError(f"GaussianRBF num_rbf ({radial.num_rbf}) must equal num_rbf " f"({core.num_rbf})")
    if abs(float(radial.cutoff_sr) - core.cutoff_sr) > 1e-4:
        raise ValueError(
            "GaussianRBF cutoff_sr "
            f"({radial.cutoff_sr}) must equal GemNet-OC cutoff_sr "
            f"({core.cutoff_sr}). Shorter interaction cutoffs reuse this "
            "expansion."
        )
    if int(getattr(radial, "envelope_exponent", -1)) != core.envelope_exponent:
        raise ValueError(
            "GaussianRBF envelope_exponent "
            f"({getattr(radial, 'envelope_exponent', None)}) must equal "
            f"the Core envelope_exponent ({core.envelope_exponent})"
        )
    if radial.apply_cutoff_fn:
        raise ValueError("GaussianRBF flavor 'gemnet-oc' must set apply_cutoff_fn: false")
    # The layer already lives in pre_sequence. A second registration would
    # parent it twice.
    object.__setattr__(core.embed, "radial_basis", radial)
