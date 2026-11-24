"""Radial bases with C³ envelopes (DPA4 / DeePMD-style)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

from .envelope import C3CutoffEnvelope


class BesselC3RadialBasis(nn.Module):
    """Bessel or Gaussian radial basis, optionally multiplied by a C³ envelope.

    Distinct from YAML ``BesselRBFLayer`` (PhysNet-style ``sin(wr)/r`` ×
    polynomial/bump cutoff). With ``apply_envelope=True`` this matches DPA4
    EdgeCache radials (``w * sinc(w r / π)`` × :class:`C3CutoffEnvelope`).
    DPA4C sets ``apply_envelope=False`` and multiplies one explicit envelope
    onto the finished radial–chemical amplitude.
    """

    def __init__(
        self,
        rcut: float,
        n_radial: int = 16,
        basis_type: str = "bessel",
        exponent: int = 7,
        apply_envelope: bool = True,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        self.basis_type = str(basis_type).lower()
        self.apply_envelope = bool(apply_envelope)
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
        """``r`` shape ``(E,)`` or ``(E, 1)`` → ``(E, n_radial)``."""
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        freqs = self.freqs.to(dtype=r.dtype, device=r.device)
        if self.basis_type == "bessel":
            x = r * freqs
            z = x / math.pi
            pz = math.pi * z
            sinc = torch.where(
                z == 0, torch.ones_like(pz), torch.sin(pz) / pz.clamp_min(1e-12)
            )
            raw = freqs * sinc
        else:
            dr = r - freqs
            raw = torch.exp(dr * dr * self.gaussian_coeff)
        if not self.apply_envelope:
            return raw
        return raw * self.envelope(r)
