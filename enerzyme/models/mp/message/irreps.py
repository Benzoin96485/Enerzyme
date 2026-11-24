"""Edge harmonics and tensor-product convolutions for flat irreps."""

from torch import nn
from torch.nn import functional as F

from ..basis.envelope import polynomial_envelope
from ..block import Block, BlockState
from ..core import Embed
from ..ops import scatter_sum


def weighted_messages(tp, features, sender, edge_sh, weights):
    """Apply an externally weighted tensor product to each sender and edge."""
    return tp(features.index_select(0, sender), edge_sh, weights)


def irreps_convolution(tp, radial, features, sender, receiver, edge_sh, rbf):
    """Sum radial-weighted tensor-product messages onto receiver atoms."""
    if sender.numel() == 0:
        return features.new_zeros(features.shape[0], tp.output_mask.numel())
    messages = weighted_messages(tp, features, sender, edge_sh, radial(rbf))
    return scatter_sum(messages, receiver, features.shape[0])


class IrrepsEmbed(Embed):
    """Prepare node features and evaluate the edge harmonics once per stack."""

    def __init__(self, width, num_rbf, sph, projection=None, envelope=None):
        super().__init__()
        self.width = width
        self.num_rbf = num_rbf
        self.sph = sph
        self.projection = projection
        self.envelope = envelope

    def forward(self, inputs, ctx):
        x = inputs["atom_embedding"]
        if x.ndim != 2 or x.shape[-1] != self.width:
            raise ValueError(f"atom_embedding width must be {self.width}, got {tuple(x.shape)}")
        if ctx.rbf.ndim != 2 or ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf must be [n_edges, {self.num_rbf}], got {tuple(ctx.rbf.shape)}")
        if ctx.sender.shape != ctx.receiver.shape or ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError("idx_i_sr, idx_j_sr, and vij_sr must share the edge axis")
        if ctx.rbf.shape[0] != ctx.num_edges:
            raise ValueError("rbf and vij_sr must describe the same edges")
        sh = self.sph(ctx.vec)
        if self.projection is None:
            return BlockState(equi=x, edge=sh)
        x = self.projection(x)
        window = None
        if self.envelope is not None:
            window = polynomial_envelope(ctx.edge_length(), *self.envelope)
        return BlockState(scalar=x, equi=x, edge=sh, aux={"envelope": window})


def equflash_update(layer, state, ctx):
    x = layer(state.equi, state.edge, ctx.rbf, ctx.sender, ctx.receiver)
    return BlockState(equi=x, edge=state.edge)


def nequip_update(layer, state, ctx):
    x = layer(state.equi, ctx.inputs["atom_embedding"], ctx.sender, ctx.receiver, state.edge, ctx.rbf)
    return BlockState(equi=x, edge=state.edge)


def separable_update(layer, state, ctx):
    x = layer(state.equi, ctx.receiver, ctx.sender, ctx.rbf, state.edge)
    return BlockState(equi=x, edge=state.edge)


def dual_branch_update(layer, state, ctx):
    x, delta = layer(state.equi, ctx.receiver, ctx.sender, ctx.rbf, state.edge, state.aux["envelope"])
    return BlockState(state.scalar + F.silu(delta), x, state.edge, state.aux)


class ChannelwiseBlock(Block):
    """Apply a channelwise convolution followed by the node feed-forward network."""

    def __init__(self, interaction: nn.Module, feedforward: nn.Module):
        super().__init__()
        self.interaction = interaction
        self.feedforward = feedforward

    def forward(self, state, ctx):
        x = self.interaction(state.equi, ctx.receiver, ctx.sender, ctx.rbf, state.edge)
        return BlockState(equi=self.feedforward(x), edge=state.edge)
