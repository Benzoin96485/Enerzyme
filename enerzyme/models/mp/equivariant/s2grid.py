"""Pointwise spherical-grid activations and degree-wise scalar gates."""

import torch
from torch import Tensor, nn

from .indexing import build_m_major_l_index


def activate_grid(inputs, grid, activation):
    """Apply a pointwise nonlinearity between inverse and forward SH transforms."""
    return grid.from_grid(activation(grid.to_grid(inputs)))


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
