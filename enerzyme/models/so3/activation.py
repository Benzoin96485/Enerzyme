# Copyright (c) EquiformerV2 authors (Liao et al., ICLR 2024).
# Ported from https://github.com/atomicarchitects/equiformer_v2 (MIT License).
"""SH-array activations used by EquiformerV2 attention / FFN blocks."""

import torch
import torch.nn as nn


class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, num_channels):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for l in range(1, self.lmax + 1):
            length = min((2 * l + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = l - 1
            start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index)

        self.scalar_act = torch.nn.SiLU()
        self.gate_act = torch.nn.Sigmoid()

    def forward(self, gating_scalars, input_tensors):
        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)


class S2Activation(torch.nn.Module):
    """Assume we only have one resolution."""

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat = SO3_grid[self.lmax][self.mmax].get_to_grid_mat(inputs.device)
        from_grid_mat = SO3_grid[self.lmax][self.mmax].get_from_grid_mat(inputs.device)
        to_grid_mat = to_grid_mat.to(dtype=inputs.dtype)
        from_grid_mat = from_grid_mat.to(dtype=inputs.dtype)
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)


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
