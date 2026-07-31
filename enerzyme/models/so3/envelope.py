"""Polynomial envelope for smooth attention cutoffs (EquiformerV3).

Distinct from PhysNet-style ``cutoff.polynomial_transition`` used in
range-separation; this matches fairchem eSEN / EquiformerV3
``PolynomialEnvelope``.
"""

from __future__ import annotations

import torch


class PolynomialEnvelope(torch.nn.Module):
    """Polynomial envelope that goes smoothly to zero at ``cutoff``.

    Reference:
        https://github.com/facebookresearch/fairchem (eSEN radial envelope)
        EquiformerV3 (Liao et al., 2026)
    """

    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d_scaled = distance / self.cutoff
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        outputs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        return outputs.view(-1, 1)

    def extra_repr(self) -> str:
        return f"cutoff={self.cutoff}, exponent={self.exponent}"
