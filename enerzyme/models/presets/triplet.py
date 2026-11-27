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
from ..mp.graph import GraphRequires, TripletRequest
from ..mp.head import FeatureHead, Head
from ..m3gnet.message import MaterialBlock, MaterialEmbed, build_material_stack
from ..mattersim.attention import CentralityEncoding, GraphormerMessage
from ..mp.nn.activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE


class DimeNetEmbed(Embed):
    input_fields = ("atom_embedding",)

    def __init__(self, basis, embedding):
        super().__init__()
        self.basis = basis
        self.embedding = embedding

    def forward(self, inputs, ctx):
        incoming, outgoing = ctx.triplets("path", exclude_backtracking=True)
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
        embed, blocks, head, GraphRequires(optional_cutoff_values=False, topology=(TripletRequest("path", exclude_backtracking=True),)),
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


def m3gnet(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    max_l: int = 3,
    num_blocks: int = 3,
    threebody_cutoff: float = 4.0,
    units: Optional[int] = None,
    n_conv_hidden: int = 2,
    dim_edge_embedding: Optional[int] = None,
    dropout: float = 0.0,
    envelope_exponent: int = 3,
) -> ComposableCore:
    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 1 or int(num_rbf) > 64:
        raise ValueError(f"num_rbf must lie in 1..64, got {num_rbf}")
    if int(max_l) < 1:
        raise ValueError(f"max_l must be >= 1, got {max_l}")
    if int(num_blocks) < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if float(threebody_cutoff) <= 0:
        raise ValueError(
            f"threebody_cutoff must be positive, got {threebody_cutoff}"
        )
    if float(threebody_cutoff) > float(cutoff_sr) + 1e-6:
        raise ValueError(
            "threebody_cutoff "
            f"({threebody_cutoff}) cannot exceed cutoff_sr ({cutoff_sr}): "
            "edges outside the neighbor list are not available"
        )
    if int(n_conv_hidden) < 0:
        raise ValueError(f"n_conv_hidden must be >= 0, got {n_conv_hidden}")
    if float(dropout) < 0 or float(dropout) >= 1:
        raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
    if int(envelope_exponent) < 1:
        raise ValueError(
            f"envelope_exponent must be >= 1, got {envelope_exponent}"
        )
    dim_edge = int(dim_embedding if dim_edge_embedding is None else dim_edge_embedding)
    units = int(dim_embedding if units is None else units)
    if dim_edge < 1 or units < 1:
        raise ValueError("units and dim_edge_embedding must be >= 1")
    degree = max_l * num_rbf
    embed = MaterialEmbed(max_l, num_rbf, dim_edge, threebody_cutoff, envelope_exponent)
    blocks = [MaterialBlock(degree, dim_embedding, dim_edge, units, n_conv_hidden, dropout)
              for _ in range(num_blocks)]
    return ComposableCore(
        embed, blocks, FeatureHead(), GraphRequires(optional_cutoff_values=False,
            topology=(TripletRequest("receiver", cutoff=threebody_cutoff),)),
        builder=build_material_stack, architecture="m3gnet",
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        dim_embedding=dim_embedding, dim_feature_out=dim_embedding, feature_irreps=f"{dim_embedding}x0e",
        units=units, readout_hidden_width=units, num_blocks=num_blocks, num_rbf=num_rbf, max_l=max_l,
        degree=degree, cutoff_sr=cutoff_sr, threebody_cutoff=threebody_cutoff, dim_edge=dim_edge,
        dropout=dropout, n_conv_hidden=n_conv_hidden, envelope_exponent=envelope_exponent,
        uses_graphormer_basis=False,
    )


def mattersim(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    max_l: int = 3,
    num_blocks: int = 3,
    threebody_cutoff: float = 4.0,
    units: Optional[int] = None,
    n_conv_hidden: int = 2,
    dim_edge_embedding: Optional[int] = None,
    dropout: float = 0.0,
    envelope_exponent: int = 3,
    num_attn_layers: Optional[int] = None,
    num_heads: int = 4,
    graphormer_num_basis: int = 16,
    attention_cutoff: Optional[float] = None,
    ffn_dim: Optional[int] = None,
    use_centrality: bool = True,
) -> ComposableCore:
    if num_attn_layers is None:
        num_attn_layers = num_blocks
    if int(num_attn_layers) < 0:
        raise ValueError(
            f"num_attn_layers must be >= 0, got {num_attn_layers}"
        )
    if int(num_heads) < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")
    if dim_embedding % int(num_heads) != 0:
        raise ValueError(
            f"dim_embedding ({dim_embedding}) must be divisible by "
            f"num_heads ({num_heads})"
        )
    if int(graphormer_num_basis) < 2:
        raise ValueError(
            "graphormer_num_basis must be >= 2, "
            f"got {graphormer_num_basis}"
        )
    resolved_cutoff = (
        cutoff_sr if attention_cutoff is None else float(attention_cutoff)
    )
    if resolved_cutoff <= 0:
        raise ValueError(
            f"attention_cutoff must be positive, got {resolved_cutoff}"
        )
    if resolved_cutoff > cutoff_sr + 1e-6:
        raise ValueError(
            "attention_cutoff "
            f"({resolved_cutoff}) cannot exceed cutoff_sr ({cutoff_sr}): "
            "edges outside the neighbor list are not available"
        )
    resolved_ffn = dim_embedding if ffn_dim is None else int(ffn_dim)
    if resolved_ffn < 1:
        raise ValueError(f"ffn_dim must be >= 1, got {ffn_dim}")
    core = m3gnet(dim_embedding, num_rbf, cutoff_sr, max_l, num_blocks, threebody_cutoff,
                  units, n_conv_hidden, dim_edge_embedding, dropout, envelope_exponent)
    core.architecture = "mattersim"
    core.num_attn_layers = int(num_attn_layers)
    core.num_heads = int(num_heads)
    core.graphormer_num_basis = int(graphormer_num_basis)
    core.attention_cutoff = resolved_cutoff
    core.ffn_dim = resolved_ffn
    core.use_centrality = bool(use_centrality)
    core.uses_graphormer_basis = core.use_centrality or num_attn_layers > 0
    core.positional_fields += ("graphormer_rbf",)
    if core.uses_graphormer_basis:
        core._input_fields.add("graphormer_rbf")
        core._relevant_fields.add("graphormer_rbf")
        core._name_mapping["graphormer_rbf"] = "graphormer_rbf"
        core.embed.attention_width = graphormer_num_basis
        core.embed.attention_cutoff = resolved_cutoff
        if use_centrality:
            core.embed.centrality = CentralityEncoding(graphormer_num_basis, dim_embedding)
    for _ in range(num_attn_layers):
        core.blocks.append(GraphormerMessage(dim_embedding, num_heads, graphormer_num_basis, resolved_ffn, dropout))
    core.schedule = []
    for step in range(max(num_blocks, num_attn_layers)):
        if step < num_blocks:
            core.schedule.append(step)
        if step < num_attn_layers:
            core.schedule.append(num_blocks + step)
    return core
