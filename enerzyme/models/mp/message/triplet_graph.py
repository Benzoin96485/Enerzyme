"""Shared message blocks for atom, bond, and angle graphs."""

import torch
from torch import nn

from ..block import Block, BlockState
from ..head import Head


def check_gemnet_edges(ctx, num_rbf):
    n = ctx.num_edges
    if (ctx.receiver.shape == (n,) and ctx.sender.shape == (n,) and ctx.dist.shape == (n,)
            and ctx.vec.shape == (n, 3) and ctx.rbf.shape == (n, num_rbf)):
        return
    raise ValueError(
        "rbf, Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same "
        f"edges (rbf width {num_rbf}, Dij_sr rank 1, vij_sr shape (E, 3)); "
        f"got rbf {tuple(ctx.rbf.shape)}, Dij_sr {tuple(ctx.dist.shape)}, vij_sr {tuple(ctx.vec.shape)}, "
        f"idx_i_sr {tuple(ctx.receiver.shape)}, idx_j_sr {tuple(ctx.sender.shape)}"
    )


class GemNetBlock(Block):
    def __init__(self, interaction):
        super().__init__()
        self.interaction = interaction

    def forward(self, state, ctx):
        args, extra = ctx.cache["gemnet_interaction"]
        atom, edge = self.interaction(state.scalar, state.edge, *args, **extra)
        return BlockState(scalar=atom, edge=edge)


class GemNetHead(Head):
    """Project the bonded features at the embedding and each interaction."""

    needs_taps = True

    def __init__(self, readouts, output_mode, include_nodes):
        super().__init__()
        self.readouts = nn.ModuleList(readouts)
        self.last_only = output_mode == "last"
        self.include_nodes = include_nodes

    def forward(self, state, ctx, taps, inputs):
        radial, receiver = ctx.cache["gemnet_readout"]
        selected = zip(self.readouts, taps)
        if self.include_nodes:
            features = [readout(tap.scalar, tap.edge, radial, receiver) for readout, tap in selected]
        else:
            features = [readout(tap.edge, radial, receiver, ctx.num_nodes) for readout, tap in selected]
        stack = torch.stack(features, dim=-1)
        return {"atom_feature": stack[..., -1] if self.last_only else stack}
