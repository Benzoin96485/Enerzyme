# Copyright (c) EquiformerV2 authors (Liao et al., ICLR 2024).
# Ported from https://github.com/atomicarchitects/equiformer_v2 (MIT License).
"""SH-array activations used by EquiformerV2 attention / FFN blocks."""

from torch import nn

from ..s2grid import S2GridActivation, SphericalGate, activate_separable_grid


class GateActivation(SphericalGate):
    """EquiformerV2's gate takes the scalar gates first."""

    def __init__(self, lmax, mmax, num_channels):
        super().__init__(lmax, mmax)
        self.num_channels = num_channels

    def forward(self, gating_scalars, input_tensors):
        return self.gate(input_tensors, gating_scalars)


class S2Activation(S2GridActivation):
    """Use the caller's grid for the configured degree and order."""

    def __init__(self, lmax, mmax):
        super().__init__(lmax, mmax)

    def forward(self, inputs, SO3_grid):
        return super().forward(inputs, SO3_grid[self.lmax][self.mmax])


class SeparableS2Activation(nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = nn.SiLU()
        self.s2_act = S2Activation(lmax, mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        return activate_separable_grid(
            input_scalars, input_tensors, self.scalar_act, self.s2_act, SO3_grid, scalar_channel_dim=-1,
        )
