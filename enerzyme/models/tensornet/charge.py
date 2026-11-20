"""Charge channels that condition TensorNet2 message passing.

The head reads the Cartesian invariants ``(I, ||A||², ||S||²)``. Neutral charge
equilibration then makes every channel sum to the molecular total charge.
The Coulomb energy that consumes the concatenated channels lives in
:class:`~enerzyme.models.layers.electrostatics.WeightedCoulombEnergyLayer`.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, SiLU

from ..layers.electrostatics import neutral_charge_equilibration
from .ops import decompose_tensor, frobenius_squared


class TensorChargeHead(Module):
    """Predict ``q_dim`` charges and square the companion weights before NQE.

    The MLP is TorchMD-Net's ``MLP(3F, 2 q_dim, F, silu, num_hidden_layers=1)``:
    linear, SiLU, linear, SiLU, linear. ``I`` enters as the isotropic
    coefficient, not ``3 I²``.
    """

    def __init__(self, dim_embedding: int, q_dim: int) -> None:
        super().__init__()
        if int(q_dim) < 1:
            raise ValueError(f"q_dim must be >= 1, got {q_dim}")
        self.dim_embedding = int(dim_embedding)
        self.q_dim = int(q_dim)
        hidden = self.dim_embedding
        self.norm = LayerNorm(3 * hidden)
        self.mlp = torch.nn.Sequential(
            Linear(3 * hidden, hidden),
            SiLU(),
            Linear(hidden, hidden),
            SiLU(),
            Linear(hidden, 2 * self.q_dim),
        )

    def invariants(self, features: Tensor) -> Tensor:
        iso, skew, traceless = decompose_tensor(features)
        return torch.cat(
            (iso, frobenius_squared(skew), frobenius_squared(traceless)),
            dim=-1,
        )

    def forward(
        self,
        features: Tensor,
        total_charge: Tensor,
        batch_seg: Tensor | None = None,
    ) -> Tensor:
        raw = self.mlp(self.norm(self.invariants(features)))
        charges = raw[:, : self.q_dim]
        weights = raw[:, self.q_dim :].square()
        return neutral_charge_equilibration(charges, weights, total_charge, batch_seg)
