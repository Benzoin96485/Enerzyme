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

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ..layers import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import GaussianRBFLayer
from .basis import legendre_outer, y_lm0
from .geometry import (
    cap_incoming_mask,
    mixed_triplets,
    quadruplet_graph,
    symmetrize_edges,
)
from .layers import BasisDownProjection


def _cutoff(name: str, value: float) -> float:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


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

    def _select(
        self,
        idx_i: Tensor,
        idx_j: Tensor,
        dist: Tensor,
        vij: Tensor,
        num_nodes: int,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask = cap_incoming_mask(idx_i, dist, num_nodes, max_neighbors, cutoff)
        return idx_i[mask], idx_j[mask], dist[mask], vij[mask]

    def _quadruplet_basis(
        self,
        main_i: Tensor,
        main_j: Tensor,
        main_d: Tensor,
        main_v: Tensor,
        rbf_main: Tensor,
        q_i: Tensor,
        q_j: Tensor,
        q_d: Tensor,
        q_v: Tensor,
        num_nodes: int,
    ) -> dict:
        assert self.mlp_rbf_qint is not None
        assert self.mlp_cbf_qint is not None
        assert self.mlp_sbf_qint is not None
        n_main = int(main_i.shape[0])
        n_sph = self.num_spherical * self.num_spherical
        radial_w4 = self._down(self.mlp_sbf_qint, rbf_main)
        rbf_q = self.mlp_rbf_qint(rbf_main)
        empty_e = main_i.new_empty(0)
        if n_main == 0 or q_i.numel() == 0:
            return {
                "rbf_q": rbf_q,
                "cbf_q": rbf_main.new_zeros(0, self.emb_size_cbf),
                "radial_w4": radial_w4,
                "angular4": rbf_main.new_zeros(0, n_sph),
                "edge_db": empty_e,
                "edge_ca": empty_e,
            }
        offset = n_main
        quads = quadruplet_graph(
            torch.cat([main_i, q_i]),
            torch.cat([main_j, q_j]),
            torch.cat([main_d, q_d]),
            torch.cat([main_v, q_v]),
            torch.arange(n_main, device=main_i.device),
            torch.arange(offset, offset + int(q_i.shape[0]), device=main_i.device),
            num_nodes,
            images=True,
        )
        if quads.edge_ca.numel() == 0:
            cbf_q = rbf_main.new_zeros(0, self.emb_size_cbf)
            angular4 = rbf_main.new_zeros(0, n_sph)
        else:
            cbf_q = self._circular(
                self.mlp_cbf_qint,
                q_d[quads.int_local],
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
        num_nodes = int(inputs["atom_embedding"].shape[0])
        if ctx.receiver.numel() and (
            int(ctx.receiver.min()) < 0
            or int(ctx.sender.min()) < 0
            or int(ctx.receiver.max()) >= num_nodes
            or int(ctx.sender.max()) >= num_nodes
        ):
            raise ValueError("edge index is outside atom_embedding")

        main_i, main_j, main_d, main_v = self._select(
            ctx.receiver,
            ctx.sender,
            ctx.dist,
            ctx.vec,
            num_nodes,
            self.cutoff,
            self.max_neighbors,
        )
        main_i, main_j, main_d, main_v, swap = symmetrize_edges(main_i, main_j, main_d, main_v)
        use_ae = self.atom_edge_interaction or self.edge_atom_interaction
        if use_ae:
            ae_i, ae_j, ae_d, ae_v = self._select(
                ctx.receiver,
                ctx.sender,
                ctx.dist,
                ctx.vec,
                num_nodes,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint,
            )
        else:
            ae_i = ae_j = ctx.receiver.new_empty(0)
            ae_d = ctx.dist.new_empty(0)
            ae_v = ctx.vec.new_empty(0, 3)
        if self.atom_interaction:
            a2a_i, a2a_j, a2a_d, _a2a_v = self._select(
                ctx.receiver,
                ctx.sender,
                ctx.dist,
                ctx.vec,
                num_nodes,
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
        else:
            a2a_i = a2a_j = ctx.receiver.new_empty(0)
            a2a_d = ctx.dist.new_empty(0)

        rbf_main = self._basis(main_d, self.cutoff)
        rbf_e2e = self.mlp_rbf_tint(rbf_main)
        rbf_h = self.mlp_rbf_h(rbf_main)
        rbf_out = self.mlp_rbf_out(rbf_main)
        radial_w3 = self._down(self.mlp_cbf_tint, rbf_main)
        trips = mixed_triplets(main_i, main_j, main_v, main_i, main_j, main_v, num_nodes)
        angular3 = y_lm0(trips.angle, self.num_spherical)

        quad_kwargs: dict = {}
        if self.quad_interaction:
            assert self.mlp_rbf_qint is not None
            assert self.mlp_cbf_qint is not None
            assert self.mlp_sbf_qint is not None
            q_i, q_j, q_d, q_v = self._select(
                ctx.receiver,
                ctx.sender,
                ctx.dist,
                ctx.vec,
                num_nodes,
                self.cutoff_qint,
                self.max_neighbors_qint,
            )
            quad_kwargs = self._quadruplet_basis(
                main_i,
                main_j,
                main_d,
                main_v,
                rbf_main,
                q_i,
                q_j,
                q_d,
                q_v,
                num_nodes,
            )
        ae_kwargs: dict = {}
        if self.atom_edge_interaction:
            assert self.mlp_rbf_aeint is not None and self.mlp_cbf_aeint is not None
            a2e = mixed_triplets(main_i, main_j, main_v, ae_i, ae_j, ae_v, num_nodes)
            ae_kwargs.update(
                rbf_a2e=self.mlp_rbf_aeint(self._basis(ae_d, self.cutoff_aeaint)),
                radial_a2e=self._down(self.mlp_cbf_aeint, rbf_main),
                angular_a2e=y_lm0(a2e.angle, self.num_spherical),
                a2e_sender=ae_j,
                a2e_other=a2e.idx_other,
                a2e_edge=a2e.idx_edge,
            )
        if self.edge_atom_interaction:
            assert self.mlp_rbf_eaint is not None and self.mlp_cbf_eaint is not None
            e2a = mixed_triplets(ae_i, ae_j, ae_v, main_i, main_j, main_v, num_nodes)
            raw_ae = self._basis(ae_d, self.cutoff_aeaint)
            ae_kwargs.update(
                rbf_e2a=self.mlp_rbf_eaint(rbf_main),
                radial_e2a=self._down(self.mlp_cbf_eaint, raw_ae),
                angular_e2a=y_lm0(e2a.angle, self.num_spherical),
                e2a_other=e2a.idx_other,
                e2a_edge=e2a.idx_edge,
                e2a_receiver=ae_i,
            )
        if self.atom_interaction:
            assert self.mlp_rbf_aint is not None
            ae_kwargs.update(
                rbf_a2a=self.mlp_rbf_aint(self._basis(a2a_d, self.cutoff_aint)),
                a2a_i=a2a_i,
                a2a_j=a2a_j,
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
