"""Smooth radial envelopes for equivariant message passing.

* ``PolynomialEnvelope`` — EquiformerV3 / fairchem eSEN attention cutoff
  (distinct from PhysNet-style ``cutoff.polynomial_transition``).
* ``C3CutoffEnvelope`` — DPA4 / DeePMD C³-continuous envelope used with
  Bessel radial bases (Li et al., arXiv:2606.02419).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


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


class C3CutoffEnvelope(torch.nn.Module):
    """C³-continuous polynomial cutoff envelope ``E_p(x)``.

    For scaled distance ``x = r / rcut`` and ``u = 1 - x``::

        E_p(x) = u^4 * sum_{k=0}^{p-1} C(k+3, 3) x^k   (x < 1)
        E_p(x) = 0                                        (x >= 1)

    Default ``p=5`` gives ``E_5(x) = u^4 (1 + 4x + 10x^2 + 20x^3 + 35x^4)``.
    """

    def __init__(self, rcut: float, exponent: int = 5) -> None:
        super().__init__()
        if rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.p = int(exponent)
        coeffs = tuple(float(math.comb(k + 3, 3)) for k in range(self.p))
        self.register_buffer(
            "series_coefficients",
            torch.tensor(coeffs, dtype=torch.float64),
            persistent=False,
        )

    def forward(self, dst: Tensor) -> Tensor:
        u = ((self.rcut - dst) / self.rcut).clamp(min=0.0, max=1.0)
        x = 1.0 - u
        coeffs = self.series_coefficients.to(dtype=x.dtype, device=x.device)
        series = torch.full(
            x.shape, float(coeffs[-1].item()), dtype=x.dtype, device=x.device
        )
        for coefficient in reversed(coeffs[:-1].tolist()):
            series = coefficient + x * series
        return (u**4) * series

    def extra_repr(self) -> str:
        return f"rcut={self.rcut}, exponent={self.p}"
