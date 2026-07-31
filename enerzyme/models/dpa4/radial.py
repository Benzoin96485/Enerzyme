"""Radial basis and C^3 envelopes for DPA4.

Reimplemented after deepmd-kit ``dpa4_nn.radial`` (Li et al., arXiv:2606.02419).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class C3CutoffEnvelope(nn.Module):
    """C^3-continuous polynomial cutoff envelope ``E_p(x)``."""

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
        series = torch.full(x.shape, coeffs[-1].item(), dtype=x.dtype, device=x.device)
        for coefficient in reversed(coeffs[:-1].tolist()):
            series = coefficient + x * series
        return (u**4) * series


class RadialBasis(nn.Module):
    """Bessel / Gaussian radial basis multiplied by a C^3 envelope."""

    def __init__(
        self,
        rcut: float,
        n_radial: int = 16,
        basis_type: str = "bessel",
        exponent: int = 7,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        self.basis_type = str(basis_type).lower()
        if self.basis_type not in ("bessel", "gaussian"):
            raise ValueError("`basis_type` must be 'bessel' or 'gaussian'")
        self.envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=int(exponent))
        if self.basis_type == "bessel":
            freqs = torch.arange(1, self.n_radial + 1, dtype=torch.float32) * (
                math.pi / self.rcut
            )
        else:
            freqs = torch.linspace(0.0, self.rcut, self.n_radial)
        self.freqs = nn.Parameter(freqs.view(1, self.n_radial))
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.gaussian_coeff = -0.5 / (gaussian_width * gaussian_width)

    def forward(self, r: Tensor) -> Tensor:
        """``r`` shape ``(E, 1)`` → ``(E, n_radial)``."""
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        freqs = self.freqs.to(dtype=r.dtype, device=r.device)
        if self.basis_type == "bessel":
            x = r * freqs
            z = x / math.pi
            pz = math.pi * z
            sinc = torch.where(z == 0, torch.ones_like(pz), torch.sin(pz) / pz.clamp_min(1e-12))
            raw = freqs * sinc
        else:
            dr = r - freqs
            raw = torch.exp(dr * dr * self.gaussian_coeff)
        return raw * self.envelope(r)


class RadialMLP(nn.Module):
    """Per-degree radial MLP: ``(E, n_in)`` → ``(E, n_out)`` or broadcast over degrees."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: Optional[Sequence[int]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = list(hidden) if hidden is not None else []
        # Convention from DPA4: ``[0]`` means a single Linear to ``n_out``.
        if len(hidden) == 1 and hidden[0] == 0:
            hidden = []
        dims: List[int] = [n_in, *hidden, n_out]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
