"""GemNet Core — directional message passing with triplets and dihedrals.

Gasteiger, Becker, and Günnemann, NeurIPS 2021. ``triplets_only=True`` is
GemNet-T (angles at the receiver). ``triplets_only=False`` is GemNet-Q
(those triplets plus dihedral quadruplets). The reference repository is
Hippocratic-licensed and is not imported; this module is an original
implementation of the published block structure.

The Core owns edge states, Fourier–Bessel angular bases, and the interaction
stack. Species embeddings and the radial Bessel basis are pre-core layers
(``RandomAtomEmbedding``, ``BesselRBF`` with ``flavor: gemnet``). Each block
emits an atom latent for :class:`~enerzyme.models.layers.readout.HierachicalReadout`;
the bias-free linear map from that latent to energy stays outside the Core.
Forces of the conservative model are ``Fa = -∇E``.
"""

from __future__ import annotations

from ..mp.block import BlockState
from ..mp.core import Embed
from ..mp.message.triplet_graph import check_gemnet_edges

from typing import Dict, List

from torch import Tensor
from torch.nn import Module

from ..layers import BaseAtomEmbedding, BaseRBF
from .basis import joint_zonal, real_spherical_harmonics, y_lm0
from ..mp.quadruplets import angle_between


class GemNetEmbed(Embed):
    def __init__(
        self,
        triplets_only,
        num_rbf,
        emb_cutoff,
        int_cutoff,
        num_spherical,
        edge_emb,
        triplet_radial,
        mlp_rbf3,
        mlp_cbf3,
        mlp_rbf_h,
        mlp_rbf_out,
        quad_cbf_radial,
        quad_sbf_radial,
        mlp_rbf4,
        mlp_cbf4,
        mlp_sbf4,
    ):
        super().__init__()
        self.triplets_only = triplets_only
        self.num_rbf = num_rbf
        self.emb_cutoff = emb_cutoff
        self.int_cutoff = int_cutoff
        self.num_spherical = num_spherical
        self.edge_emb = edge_emb
        self.triplet_radial = triplet_radial
        self.mlp_rbf3 = mlp_rbf3
        self.mlp_cbf3 = mlp_cbf3
        self.mlp_rbf_h = mlp_rbf_h
        self.mlp_rbf_out = mlp_rbf_out
        self.quad_cbf_radial = quad_cbf_radial
        self.quad_sbf_radial = quad_sbf_radial
        self.mlp_rbf4 = mlp_rbf4
        self.mlp_cbf4 = mlp_cbf4
        self.mlp_sbf4 = mlp_sbf4

    def _quad_basis(self, embedding, interaction, rbf_emb) -> Dict[str, Tensor]:
        assert self.quad_cbf_radial is not None
        assert self.quad_sbf_radial is not None
        assert self.mlp_rbf4 is not None
        assert self.mlp_cbf4 is not None
        assert self.mlp_sbf4 is not None
        quads = embedding.quadruplets(interaction)
        sbf_radial = self.quad_sbf_radial(embedding.dist)
        if quads.edge_ca.numel() == 0:
            cbf = rbf_emb.new_empty(0, self.mlp_cbf4.linear.out_features)
            angular = rbf_emb.new_empty(0, self.quad_sbf_radial.angular_dim)
        else:
            radial_ab = self.quad_cbf_radial(interaction.dist)[quads.int_local]
            joint = joint_zonal(radial_ab, y_lm0(quads.angle_abd, self.num_spherical))
            cbf = self.mlp_cbf4(joint)
            angular = real_spherical_harmonics(quads.angle_cab, quads.dihedral, self.num_spherical)
        return {
            "rbf4": self.mlp_rbf4(rbf_emb),
            "cbf4": cbf,
            "radial_w4": self.mlp_sbf4(sbf_radial),
            "angular4": angular,
            "edge_db": quads.edge_db,
            "edge_ca": quads.edge_ca,
        }

    def forward(self, inputs, ctx):
        check_gemnet_edges(ctx, self.num_rbf)
        embedding = ctx.within(self.emb_cutoff)
        idx_i, idx_j = embedding.receiver, embedding.sender
        rbf_emb = embedding.rbf
        swap = embedding.reverse_edge()
        edge, other = embedding.triplets("receiver", exclude_same_sender=True)
        angles = angle_between(embedding.vec[edge], embedding.vec[other])
        angular3 = y_lm0(angles, self.num_spherical)
        radial_w3 = self.mlp_cbf3(self.triplet_radial(embedding.dist))
        rbf3 = self.mlp_rbf3(rbf_emb)
        rbf_h = self.mlp_rbf_h(rbf_emb)
        rbf_out = self.mlp_rbf_out(rbf_emb)

        h = inputs["atom_embedding"]
        m = self.edge_emb(h, rbf_emb, idx_i, idx_j)
        quad: Dict[str, Tensor] = {}
        if not self.triplets_only:
            quad = self._quad_basis(embedding, ctx.within(self.int_cutoff), rbf_emb)
        args = (rbf3, rbf_h, radial_w3, angular3, edge, other, swap, idx_i, idx_j)
        ctx.cache["gemnet_interaction"] = args, quad
        ctx.cache["gemnet_readout"] = rbf_out, idx_i
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
                "GemNet output_mode='last' returns atom_feature of shape (N, F) "
                "for SimpleReadout. HierachicalReadout reads "
                "(N, F, num_blocks+1). Use output_mode='feature'."
            )
    for layer in built_layers:
        if isinstance(layer, BaseAtomEmbedding):
            core.atom_embedding = layer
        elif isinstance(layer, BaseRBF):
            flavor = getattr(layer, "flavor", None)
            if flavor != "gemnet":
                raise ValueError("GemNetCore requires BesselRBF with flavor 'gemnet', " f"found {flavor!r}")
            basis_cutoff = float(getattr(layer, "basis_cutoff", layer.cutoff_sr))
            if abs(basis_cutoff - core.emb_cutoff) > 1e-4:
                raise ValueError(
                    "BesselRBF basis_cutoff "
                    f"({basis_cutoff}) must equal GemNet emb_cutoff "
                    f"({core.emb_cutoff}). Set basis_cutoff when the "
                    "neighbor list is longer than the embedding cutoff."
                )
            core.radial_basis_function = layer
