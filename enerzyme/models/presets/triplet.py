"""Presets that exchange messages between adjacent directed edges."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from ..dimenet.basis import SphericalFourierBesselBasis
from ..dimenet.defaults import _SWISH
from ..dimenet.interaction import (
    DimeNetAtomReadout, DimeNetEmbeddingBlock, DimeNetInteractionBlock,
    DimeNetPPAtomReadout, DimeNetPPInteractionBlock,
)
from ..dimenet.triplets import triplet_angles
from ..mp.block import Block, BlockState
from ..mp.core import ComposableCore, Embed
from ..mp.graph import GraphRequires
from ..mp.head import Head
from ..mp.nn.activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE


class DimeNetEmbed(Embed):
    input_fields = ("atom_embedding",)

    def __init__(self, basis, embedding):
        super().__init__()
        self.basis = basis
        self.embedding = embedding

    def forward(self, inputs, ctx):
        incoming, outgoing = ctx.path_pairs(exclude_backtracking=True)
        angles = triplet_angles(ctx.vec, incoming, outgoing)
        angular = self.basis(ctx.dist, angles, incoming)
        edge = self.embedding(inputs["atom_embedding"], ctx.rbf, ctx.receiver, ctx.sender)
        return BlockState(edge=edge, aux={"triplets": (angular, incoming, outgoing)})


class TripletMessage(Block):
    """Update each directed edge from its incoming triplet messages."""

    def __init__(self, interaction):
        super().__init__()
        self.interaction = interaction

    def forward(self, state, ctx):
        edge = self.interaction(state.edge, ctx.rbf, *state.aux["triplets"])
        return BlockState(edge=edge, aux=state.aux)


class DimeNetHead(Head):
    def __init__(self, readouts, output_mode):
        super().__init__()
        self.readouts = nn.ModuleList(readouts)
        self.needs_taps = output_mode != "last"

    def forward(self, state, ctx, taps, inputs):
        if self.needs_taps:
            features = [readout(tap.edge, ctx.rbf, ctx.receiver, ctx.num_nodes)
                        for readout, tap in zip(self.readouts, taps)]
            return {"atom_feature": torch.stack(features, dim=-1)}
        return {"atom_feature": self.readouts[-1](state.edge, ctx.rbf, ctx.receiver, ctx.num_nodes)}


def _dimenet(dim_embedding, num_rbf, cutoff_sr, num_blocks, num_spherical, num_before_skip,
             num_after_skip, envelope_exponent, activation_fn, activation_params, output_mode,
             interaction, readout, width, interaction_params, architecture):
    if num_spherical < 1 or num_blocks < 1:
        raise ValueError("num_spherical and num_blocks must be >= 1")
    if activation_params is None:
        activation_params = dict(_SWISH)
    basis = SphericalFourierBesselBasis(num_spherical, num_rbf, cutoff_sr, envelope_exponent)
    embed = DimeNetEmbed(basis, DimeNetEmbeddingBlock(dim_embedding, num_rbf, activation_fn, activation_params))
    blocks = [TripletMessage(interaction(
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_sbf=basis.out_dim,
        num_before_skip=num_before_skip, num_after_skip=num_after_skip,
        activation_fn=activation_fn, activation_params=activation_params, **interaction_params,
    )) for _ in range(num_blocks)]
    head = DimeNetHead([readout() for _ in range(num_blocks + 1)], output_mode)
    return ComposableCore(
        embed, blocks, head, GraphRequires(optional_cutoff_values=False, path_triplets=True),
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture=architecture, output_mode=output_mode, num_output_blocks=num_blocks + 1,
        dim_feature_out=width, feature_irreps=f"{width}x0e", dim_embedding=dim_embedding,
        num_rbf=num_rbf, cutoff_sr=cutoff_sr, num_blocks=num_blocks, num_spherical=num_spherical,
        **interaction_params,
    )


def dimenet(dim_embedding: int, num_rbf: int, cutoff_sr: float, num_blocks: int = 6,
            num_bilinear: int = 8, num_spherical: int = 7, num_before_skip: int = 1,
            num_after_skip: int = 2, envelope_exponent: int = 5,
            activation_fn: ACTIVATION_KEY_TYPE = "swish",
            activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
            output_mode: Literal["feature", "last"] = "feature") -> ComposableCore:
    return _dimenet(
        dim_embedding, num_rbf, cutoff_sr, num_blocks, num_spherical, num_before_skip,
        num_after_skip, envelope_exponent, activation_fn, activation_params, output_mode,
        DimeNetInteractionBlock, lambda: DimeNetAtomReadout(num_rbf, dim_embedding),
        dim_embedding, {"num_bilinear": num_bilinear}, "dimenet",
    )


def dimenetpp(dim_embedding: int, num_rbf: int, cutoff_sr: float, num_blocks: int = 4,
              out_emb_size: int = 256, int_emb_size: int = 64, basis_emb_size: int = 8,
              num_spherical: int = 7, num_before_skip: int = 1, num_after_skip: int = 2,
              envelope_exponent: int = 5, activation_fn: ACTIVATION_KEY_TYPE = "swish",
              activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
              output_mode: Literal["feature", "last"] = "feature") -> ComposableCore:
    if min(out_emb_size, int_emb_size, basis_emb_size) < 1:
        raise ValueError("embedding sizes must be >= 1")
    core = _dimenet(
        dim_embedding, num_rbf, cutoff_sr, num_blocks, num_spherical, num_before_skip,
        num_after_skip, envelope_exponent, activation_fn, activation_params, output_mode,
        DimeNetPPInteractionBlock, lambda: DimeNetPPAtomReadout(num_rbf, dim_embedding, out_emb_size),
        out_emb_size, {"int_emb_size": int_emb_size, "basis_emb_size": basis_emb_size}, "dimenet++",
    )
    core.out_emb_size = out_emb_size
    return core
