"""Tensor Atomic Cluster Expansion embedding and spherical product block."""

import torch
from torch.nn import functional as F

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.equivariant.e3nn import IrrepsLinear
from .interaction import EDGE_EMBEDDING


class TACEEmbed(Embed):
    input_fields = ("atom_embedding", "charge_embedding", "spin_embedding", "Za")

    def __init__(self, num_elements, dim_embedding, num_channel, num_rbf, edge_embedding, lmax,
                 cartesian_harmonics=None):
        super().__init__()
        self.num_elements = num_elements
        self.node_proj = (
            IrrepsLinear(f"{dim_embedding}x0e", f"{num_channel}x0e", bias=False)
            if dim_embedding != num_channel else torch.nn.Identity()
        )
        self.edge_embedding = EDGE_EMBEDDING[edge_embedding](
            num_radial_basis=num_rbf, num_channel=num_channel, num_elements=num_elements, bias=False,
        )
        self.lmax = lmax
        self.cartesian_harmonics = cartesian_harmonics

    def forward(self, inputs, ctx):
        features = inputs["atom_embedding"]
        for field in ("charge_embedding", "spin_embedding"):
            if inputs.get(field) is not None:
                features = features + inputs[field]
        features = self.node_proj(features)
        node_attrs = F.one_hot(inputs["Za"], num_classes=self.num_elements).to(dtype=features.dtype)
        edge_index = torch.stack((ctx.receiver, ctx.sender))
        cutoff = ctx.cutoff_values.reshape(-1, 1) if ctx.cutoff_values is not None else None
        embedding = self.edge_embedding(node_attrs, ctx.rbf, edge_index)
        if self.cartesian_harmonics is None:
            harmonics = ctx.spherical_harmonics(self.lmax)
        else:
            harmonics = self.cartesian_harmonics(ctx.vec)
            features = {0: features}
        return BlockState(equi=features, edge=embedding, aux={
            "node_attrs": node_attrs, "edge_index": edge_index, "edge_attrs": harmonics, "cutoff": cutoff,
        })


class SphericalTACEBlock(Block):
    def __init__(self, edge_update, interaction, product):
        super().__init__()
        self.edge_update = edge_update
        self.interaction = interaction
        self.product = product

    def forward(self, state, ctx):
        node_attrs, edge_index = state.aux["node_attrs"], state.aux["edge_index"]
        edge_feats = self.edge_update(node_attrs, state.edge, edge_index)
        features, skip = self.interaction(
            node_feats=state.equi, node_attrs=node_attrs, edge_feats=edge_feats,
            edge_attrs=state.aux["edge_attrs"], edge_index=edge_index, cutoff=state.aux["cutoff"],
        )
        return state._replace(equi=self.product(node_feats=features, node_attrs=node_attrs, sc=skip))
