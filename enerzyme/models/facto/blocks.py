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
        return BlockState(
            scalar=scalars, equi=self._zero_moments(scalars), edge=self.geometry(ctx.dist, ctx.vec),
            aux={"isolated": isolated},
        )


class FACTOBlock(FACTOInteraction):
    def forward(self, state, ctx):
        scalars, tensors = super().forward(state.scalar, state.equi, state.edge, ctx.receiver, ctx.sender)
        # Without neighbors every tensor rank and the aggregated scalar message
        # stay zero. The remaining affine bias terms give the same increment to
        # every isolated species. Keep Linear calls for the parameter and AMP dtypes.
        isolated = state.aux["isolated"]
        empty_message = isolated.new_zeros(1, self.attention.mix.in_features)
        intermediate = F.silu(self.attention.mix(empty_message))
        empty_dots = isolated.new_zeros(1, self.feedback.mix.in_features)
        delta = self.feedback.mix(empty_dots) * self.feedback.to_gate(intermediate)
        delta = delta + self.feedback.to_residual(intermediate)
        if torch.is_grad_enabled():
            # A zero neighbor sum and zero tensor channels still connect these
            # weights in the full empty-graph calculation. Preserve their zero
            # gradients (and optimizer weight decay) without evaluating the maps.
            zero_gradient_weights = (
                self.norm.scalar_gain, *self.norm.tensor_gain,
                self.attention.query.weight, self.attention.query.bias,
                *(layer.weight for layer in self.feedback.left),
                *(layer.weight for layer in self.feedback.right),
            )
            empty = torch.cat([weight.flatten()[:0] for weight in zero_gradient_weights])
            delta = delta + empty.sum().to(delta.dtype)
        isolated = isolated + self.residual_scale * delta
        return BlockState(scalar=scalars, equi=tensors, edge=state.edge, aux={"isolated": isolated})


class FACTOHead(Head):
    needs_taps = True
    output_fields = ("atom_feature", "isolated_atom_feature")

    def forward(self, state, ctx, taps, inputs):
        layers = taps[1:]
        return {
            "atom_feature": torch.stack([tap.scalar for tap in layers], dim=-1),
            "isolated_atom_feature": torch.stack([tap.aux["isolated"] for tap in layers], dim=-1)[inputs["Za"]],
        }
