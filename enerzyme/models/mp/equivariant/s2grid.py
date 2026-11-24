"""Pointwise spherical-grid activations and degree-wise scalar gates."""

import torch
from torch import Tensor, nn

from .indexing import build_m_major_l_index


def activate_grid(inputs, grid, activation):
    """Apply a pointwise nonlinearity between inverse and forward SH transforms."""
    return grid.from_grid(activation(grid.to_grid(inputs)))


class S2GridActivation(nn.Module):
    """Initialize the pointwise SiLU used with a selected spherical grid."""

    def __init__(self, lmax, mmax, grid=None):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        if grid is not None:
            self.so3_grid = grid
        self.act = nn.SiLU()

    def forward(self, inputs, grid):
        return activate_grid(inputs, grid, self.act)


def activate_separable_grid(scalars, inputs, scalar_act, tensor_act, *grid_args, scalar_channel_dim=1):
    """Replace degree zero after activating independent scalar and grid branches.

    Activate scalars first to preserve the order of dropout draws. V2 accepts
    a trailing channel axis; V3's scalar input has shape (nodes, channels).
    """
    scalars = scalar_act(scalars)
    scalars = scalars.reshape(scalars.shape[0], 1, scalars.shape[scalar_channel_dim])
    coefficients = tensor_act(inputs, *grid_args)
    return torch.cat((scalars, coefficients.narrow(1, 1, coefficients.shape[1] - 1)), dim=1)


class S2Activation(S2GridActivation):
    """Pointwise activation on a grid owned by the module."""

    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        from .s2.grid import SO3Grid

        grid = SO3Grid(lmax, mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        super().__init__(lmax, mmax, grid)

    def forward(self, inputs):
        return super().forward(inputs, self.so3_grid)


class SeparableS2Activation(S2Activation):
    """Use the same nonlinearity for the independent scalar and grid channels."""

    def forward(self, inputs, scalars):
        return activate_separable_grid(scalars, inputs, self.act, super().forward)


class SphericalGate(nn.Module):
    """Apply SiLU to scalars and one sigmoid gate per higher-degree channel."""

    def __init__(self, lmax: int, mmax: int, use_m_primary: bool = False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        if use_m_primary:
            degree = torch.as_tensor(build_m_major_l_index(lmax, mmax))[1:]
        else:
            degree = torch.tensor(
                [l for l in range(1, lmax + 1) for _ in range(2 * min(l, mmax) + 1)],
                dtype=torch.long,
            )
        self.register_buffer("expand_index", degree - 1)
        self.scalar_act = nn.SiLU()
        self.gate_act = nn.Sigmoid()

    def gate(self, inputs: Tensor, scalars: Tensor) -> Tensor:
        gates = self.gate_act(scalars).reshape(
            inputs.shape[0], self.lmax, inputs.shape[-1]
        )
        gates = gates.index_select(1, self.expand_index)
        return torch.cat((self.scalar_act(inputs[:, :1]), inputs[:, 1:] * gates), dim=1)
