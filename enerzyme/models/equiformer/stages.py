"""Equiformer degree embedding and equivariant attention readout."""

import torch
from e3nn import o3

from ..mp.core import Embed
from ..mp.block import BlockState
from ..mp.head import Head


class DegreeEmbed(Embed):
    def __init__(self, degree, irreps_sh):
        super().__init__()
        self.degree = degree
        self.sph = o3.SphericalHarmonics(irreps_sh, normalize=True, normalization="component")

    def forward(self, inputs, ctx):
        x = inputs["atom_embedding"]
        batch = ctx.batch
        if batch is None:
            batch = x.new_zeros(ctx.num_nodes, dtype=torch.long)
        sh = self.sph(ctx.vec)
        x = x + self.degree(x, sh, ctx.rbf, ctx.sender, ctx.receiver, batch)
        node_attr = torch.ones_like(x.narrow(1, 0, 1))
        return BlockState(scalar=node_attr, equi=x, edge=sh, aux={"batch": batch})


def graph_attention_update(layer, state, ctx):
    x = layer(node_input=state.equi, node_attr=state.scalar,
              edge_src=ctx.sender, edge_dst=ctx.receiver, edge_attr=state.edge,
              edge_scalars=ctx.rbf, batch=state.aux["batch"])
    return BlockState(state.scalar, x, state.edge, state.aux)


class AttentionHead(Head):
    def __init__(self, norm, dropout, readout, attention):
        super().__init__()
        self.norm = norm
        self.dropout = dropout
        self.readout = readout
        self.attention = attention
        self.output_fields = ("atom_feature",) if readout is None else ("Ea",)

    def features(self, state):
        x = self.norm(state.equi, batch=state.aux["batch"])
        return self.dropout(x) if self.dropout is not None else x

    def forward(self, state, ctx, taps, inputs):
        x = self.features(state)
        if self.readout is None:
            return {"atom_feature": x}
        if self.attention:
            x = self.readout(node_input=x, node_attr=state.scalar,
                             edge_src=ctx.sender, edge_dst=ctx.receiver, edge_attr=state.edge,
                             edge_scalars=ctx.rbf, batch=state.aux["batch"])
        else:
            x = self.readout(x)
        return {"Ea": x.view(-1)}
