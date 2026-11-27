"""Strictly local, edge-centred Allegro tensor and scalar tracks."""

import math
import torch
from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.head import Head
from ..mp.ops import scatter_sum


class PairEmbed(Embed):
    input_fields = ("Za",)

    def __init__(self, scalar_mlp, tensor_weight, pair_projection, weighter, sph,
                 dim_edge, scalar_width, avg_num_neighbors):
        super().__init__()
        self.scalar_mlp = scalar_mlp
        self.tensor_weight = tensor_weight
        self.pair_projection = pair_projection
        self.weighter = weighter
        self.sph = sph
        self.dim_edge = dim_edge
        self.scalar_width = scalar_width
        self.register_buffer("neighbor_scale", torch.tensor(1.0 / math.sqrt(avg_num_neighbors)), persistent=False)

    def forward(self, inputs, ctx):
        embedded = inputs["edge_embedding"]
        if embedded.ndim != 2 or embedded.shape[-1] != self.dim_edge:
            raise ValueError(f"edge_embedding width must be {self.dim_edge}")
        if ctx.receiver.shape != ctx.sender.shape or embedded.shape[0] != ctx.num_edges:
            raise ValueError("edge_embedding and the neighbor indices must describe the same edges")
        if ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError("vij_sr must have shape [edges, 3]")
        dtype = self.scalar_mlp.net[0].weight.dtype
        scalar = self.scalar_mlp(embedded.to(dtype))
        basis = self.sph(ctx.vec.to(dtype))
        tensor = self.weighter(basis, self.tensor_weight(scalar))
        projected = self.pair_projection(scalar)
        ctx.cache["pair_basis"] = basis
        species = inputs.get("Za")
        if torch.compiler.is_compiling() and species is not None:
            # The fixed node axis keeps this allocation inside the coordinate VJP.
            ctx.cache["pair_centers"] = species.shape[0]
        else:
            ctx.cache["pair_centers"] = int(ctx.receiver.max()) + 1 if ctx.num_edges else 0
        ctx.cache["neighbor_scale"] = self.neighbor_scale.to(dtype)
        return BlockState(equi=tensor, edge=(projected[:, :self.scalar_width],),
                          aux={"env_weight": projected[:, self.scalar_width:]})


class PairProduct(Block):
    def __init__(self, product, latent, weighter, scalar_width, tensor_width):
        super().__init__()
        self.product = product
        self.latent = latent
        self.weighter = weighter
        self.scalar_width = scalar_width
        self.tensor_width = tensor_width

    def forward(self, state, ctx):
        env = self.weighter(ctx.cache["pair_basis"], state.aux["env_weight"])
        env = scatter_sum(env, ctx.receiver, ctx.cache["pair_centers"]) * ctx.cache["neighbor_scale"]
        tensor = self.product(state.equi, env.index_select(0, ctx.receiver))
        updated = self.latent(torch.cat((*state.edge, tensor[:, :self.tensor_width]), dim=-1))
        return BlockState(equi=tensor, edge=(*state.edge, updated[:, :self.scalar_width]),
                          aux={"env_weight": updated[:, self.scalar_width:]})


class PairHead(Head):
    output_fields = ("edge_feature",)

    def forward(self, state, ctx, taps, inputs):
        return {"edge_feature": torch.cat(state.edge, dim=-1)}


def pair_stack(core, layers):
    core.adopt_radius_stack(layers)
    for layer in layers:
        width = getattr(layer, "dim_edge", core.dim_edge)
        if width is not None and int(width) != core.dim_edge:
            raise ValueError(f"Allegro dim_edge {core.dim_edge} != embedding dim_edge {width}")
