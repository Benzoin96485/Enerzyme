"""GRACE atomic basis, recursive product blocks and density head."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.graph import GraphContext
from ..mp.head import Head
from ..mp.equivariant.body_order import ProductLadder
from .basis import AtomicBasis
from .coupling import EquivariantIndicator, OrderSpec
from .features import split_spherical_harmonics
from .radial import RadialEmbedding


class GRACEEmbed(Embed):
    input_fields = ("atom_embedding", "Za")

    def __init__(self, width: int, avg_n_neigh: float, lmax: int):
        super().__init__()
        self.width = width
        self.lmax = lmax
        self.register_buffer("avg_n_neigh", torch.tensor(float(avg_n_neigh)))

    def forward(self, inputs, ctx):
        embedding = inputs["atom_embedding"]
        if embedding.shape[-1] != self.width:
            raise ValueError(f"atom_embedding width {embedding.shape[-1]} != dim_embedding {self.width}")
        harmonics = ctx.spherical_harmonics(self.lmax)
        scale = (1.0 / self.avg_n_neigh).to(dtype=embedding.dtype)
        return BlockState(scalar=embedding, edge=(harmonics, scale, inputs["Za"].long()))


class GRACEDensityHead(Head):
    needs_taps = True
    output_fields = ("atom_feature",)

    def __init__(self, density):
        super().__init__()
        self.density = density

    def forward(self, state, ctx, taps, inputs):
        return {"atom_feature": self.density([tap.aux["orders"] for tap in taps[1:]], state.edge[2])}


class GRACEBlock(Block):
    """Radial embedding, atomic basis, and product ladder for one message layer.

    ``incoming_lmax`` is the angular momentum of the neighbor message this
    layer consumes. ``outgoing_lmax`` is the message it emits for the next
    layer; ``None`` means this is the last layer.
    """

    def __init__(
        self,
        orders: Sequence[OrderSpec],
        *,
        n_basis: int,
        cutoff: float,
        cutoff_order: int,
        basis: str,
        radial_hidden: int,
        radial_activation: str,
        embedding_dim: int,
        incoming_lmax: Optional[int],
        outgoing_lmax: Optional[int],
        indicator_channels: int,
        indicator_out: Optional[int],
        n_species: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.orders = list(orders)
        self.lmax = int(orders[0].lmax)
        self.n_rad = int(orders[0].channels)
        self.incoming_lmax = None if incoming_lmax is None else int(incoming_lmax)
        self.outgoing_lmax = None if outgoing_lmax is None else int(outgoing_lmax)
        self.radial = RadialEmbedding(
            n_basis,
            self.n_rad,
            self.lmax,
            cutoff,
            cutoff_order,
            basis=basis,
            hidden=radial_hidden,
            activation=radial_activation,
        )
        self.atomic = AtomicBasis(
            self.n_rad,
            self.lmax,
            embedding_dim,
            indicator_lmax=self.incoming_lmax,
            epsilon=epsilon,
        )
        self.ladder = ProductLadder(self.orders, epsilon=epsilon)
        self.indicator: Optional[EquivariantIndicator] = None
        if indicator_out is not None:
            if self.outgoing_lmax is None:
                raise ValueError("a message-passing layer needs outgoing indicator_lmax")
            self.indicator = EquivariantIndicator(
                [order.channels for order in self.orders],
                [order.lmax for order in self.orders],
                self.outgoing_lmax,
                indicator_channels,
                int(indicator_out),
                n_species,
            )

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        embedding = state.scalar
        packed, neighbor_scale, species = state.edge
        harmonics = split_spherical_harmonics(packed[:, :(self.lmax + 1)**2], self.lmax)
        radial = self.radial(ctx.dist)
        atomic = self.atomic(
            radial,
            harmonics,
            embedding,
            ctx.receiver,
            ctx.sender,
            ctx.num_nodes,
            neighbor_scale,
            state.equi,
        )
        orders = self.ladder(atomic)
        outgoing = None
        if self.indicator is not None:
            outgoing = self.indicator(orders, species)
        return BlockState(scalar=embedding, equi=outgoing, edge=state.edge, aux={"orders": orders})
