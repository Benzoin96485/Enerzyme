# Copyright (c) EquiformerV2 authors (Liao et al., ICLR 2024).
# Ported from https://github.com/atomicarchitects/equiformer_v2 (MIT License).
"""SH-array activations used by EquiformerV2 attention / FFN blocks."""

import torch
import torch.nn as nn


from ..s2grid import SphericalGate, activate_grid


class GateActivation(SphericalGate):
    """EquiformerV2's gate takes the scalar gates first."""

    def __init__(self, lmax, mmax, num_channels):
        super().__init__(lmax, mmax)
        self.num_channels = num_channels

    def forward(self, gating_scalars, input_tensors):
        return self.gate(input_tensors, gating_scalars)


class S2Activation(torch.nn.Module):
    """Assume we only have one resolution."""

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        grid = SO3_grid[self.lmax][self.mmax]
        return activate_grid(inputs, grid, self.act)


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        return torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )
