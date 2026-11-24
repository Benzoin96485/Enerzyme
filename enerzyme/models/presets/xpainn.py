"""Compose XequiNet's optional xPaiNN embedding, message/update pairs and property head."""

from typing import Union

import torch
from torch import nn
from torch_geometric.data import Data

from ..mp.block import BlockState
from ..mp.core import ComposableCore, Embed
from ..mp.graph import GraphRequires
from ..mp.head import Head


class XPaiNNEmbed(Embed):
    input_fields = ("Ra", "Za", "offsets")

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, inputs, ctx):
        edge_index = torch.stack((ctx.receiver, ctx.sender))
        positions = inputs["Ra"]
        shifts = inputs.get("offsets")
        if shifts is None:
            shifts = positions.new_zeros((edge_index.shape[1], 3))
        scalar, rbf, cutoff, harmonics = self.embedding(inputs["Za"], positions, edge_index, shifts)
        equi = scalar.new_zeros((ctx.num_nodes, harmonics.shape[-1]))
        return BlockState(scalar=scalar, equi=equi, edge=(rbf, cutoff, harmonics, edge_index))


class XPaiNNBlock(nn.Module):
    def __init__(self, message, update):
        super().__init__()
        self.message = message
        self.update = update

    def forward(self, state, ctx):
        scalar, equi = self.message(state.scalar, state.equi, *state.edge)
        scalar, equi = self.update(scalar, equi)
        return BlockState(scalar=scalar, equi=equi, edge=state.edge)


class XPaiNNHead(Head):
    input_fields = ("Ra", "Za", "batch_seg")
    output_fields = ("E", "Fa")

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, state, ctx, taps, inputs):
        data = Data(pos=inputs["Ra"], at_no=inputs["Za"], batch=ctx.batch, edge_index=state.edge[-1])
        energy, forces = self.output(data, state.scalar, state.equi)
        return {"E": energy.squeeze(-1), "Fa": forces}


def xpainn(
    cutoff_sr: float, embed_basis: str, aux_basis: str, node_dim: int, edge_irreps: str,
    hidden_dim: int, hidden_irreps: str, rbf_kernel: str, num_basis: int, cutoff_fn: str,
    max_edges: int, action_blocks: int, activation: str, norm_type: str, output_mode: str,
    output_dim: int, node_average: Union[bool, float], default_dtype: str,
) -> ComposableCore:
    try:
        from xequinet.nn.model import XPaiNN
        from xequinet.utils.config import NetConfig
    except ImportError as exc:
        raise ImportError("xPaiNN requires XequiNet and its optional dependencies") from exc

    config = NetConfig(
        cutoff=cutoff_sr, embed_basis=embed_basis, aux_basis=aux_basis, node_dim=node_dim,
        edge_irreps=edge_irreps, hidden_dim=hidden_dim, hidden_irreps=hidden_irreps,
        rbf_kernel=rbf_kernel, num_basis=num_basis, cutoff_fn=cutoff_fn, max_edges=max_edges,
        action_blocks=action_blocks, activation=activation, norm_type=norm_type, output_mode=output_mode,
        output_dim=output_dim, node_average=node_average, default_dtype=default_dtype,
    )
    external = XPaiNN(config)
    return ComposableCore(
        XPaiNNEmbed(external.embed),
        [XPaiNNBlock(message, update) for message, update in zip(external.message, external.update)],
        XPaiNNHead(external.out),
        GraphRequires(edge_set="lr", vectors=False, distances=False, rbf=False, optional_cutoff_values=False),
        node_field="Za", positional_fields=("Ra", "Za", "batch_seg", "idx_i", "idx_j"), architecture="xpainn",
    )
