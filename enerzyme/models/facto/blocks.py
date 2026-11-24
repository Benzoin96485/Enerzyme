"""FACTO chemical embedding, paired interacting/isolated states and tap head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..mp.block import BlockState
from ..mp.core import Embed
from ..mp.head import Head
from .descriptors import DESCRIPTOR_DIM, elemental_descriptor_table
from .geometry import FACTOGeometry
from .interaction import FACTOInteraction


class FACTOEmbed(Embed):
    input_fields = ("Za",)

    def __init__(self, max_Za, scalar_dim, num_radial, tensor_channels, cutoff_sr, r_short):
        super().__init__()
        self.tensor_channels = tensor_channels
        self.register_buffer("element_descriptors", elemental_descriptor_table(max_Za), persistent=False)
        if self.element_descriptors.shape[-1] != DESCRIPTOR_DIM:
            raise RuntimeError("elemental descriptor width drifted from 31")
        self.linear = nn.Linear(DESCRIPTOR_DIM, scalar_dim)
        self.geometry = FACTOGeometry(
            scalar_dim=scalar_dim, n_radial=num_radial, channels=tensor_channels, cutoff=cutoff_sr, r_short=r_short,
        )

    def _zero_moments(self, scalars):
        return [scalars.new_zeros(scalars.shape[0], 2 * ell + 1, width)
                for ell, width in enumerate(self.tensor_channels, start=1)]

    def forward(self, inputs, ctx):
        isolated = F.silu(self.linear(self.element_descriptors.to(self.linear.weight)))
        scalars = isolated[inputs["Za"]]
        active_geometry = self.geometry(ctx.dist, ctx.vec)
        isolated_geometry = self.geometry(ctx.dist[:0], ctx.vec[:0])
        return BlockState(
            scalar=scalars, equi=self._zero_moments(scalars), edge=(active_geometry, isolated_geometry),
            aux={"isolated": isolated, "isolated_equi": self._zero_moments(isolated)},
        )


class FACTOBlock(FACTOInteraction):
    def forward(self, state, ctx):
        geometry, empty_geometry = state.edge
        scalars, tensors = super().forward(state.scalar, state.equi, geometry, ctx.receiver, ctx.sender)
        isolated, isolated_equi = super().forward(
            state.aux["isolated"], state.aux["isolated_equi"], empty_geometry, ctx.receiver[:0], ctx.sender[:0],
        )
        return BlockState(scalar=scalars, equi=tensors, edge=state.edge,
                          aux={"isolated": isolated, "isolated_equi": isolated_equi})


class FACTOHead(Head):
    needs_taps = True
    output_fields = ("atom_feature", "isolated_atom_feature")

    def forward(self, state, ctx, taps, inputs):
        layers = taps[1:]
        return {
            "atom_feature": torch.stack([tap.scalar for tap in layers], dim=-1),
            "isolated_atom_feature": torch.stack([tap.aux["isolated"] for tap in layers], dim=-1)[inputs["Za"]],
        }
