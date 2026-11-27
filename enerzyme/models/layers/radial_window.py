"""Smooth window on an existing radial basis.

Geodite multiplies the Bessel expansion by the DimeNet polynomial envelope
(value and first two derivatives vanish at the cutoff) and by the covalent
distance weight. Both factors are independent of the message-passing core,
so any later stack can reuse this layer on ``rbf``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from ..mp.basis.envelope import polynomial_envelope
from . import BaseFFLayer
from .covalent import (
    constrained_attenuation_exponents,
    covalent_distance_weight,
    covalent_radii_table,
)


class RadialWindowLayer(BaseFFLayer):
    """Multiply ``rbf`` by a cutoff envelope and an optional distance weight.

    ``envelope="polynomial"`` is Gasteiger et al. (p = 6 in Geodite-MP).
    ``attenuation="covalent"`` is Eq. 2 of Reschützegger et al. (2026).
    ``attenuation="none"`` leaves the basis unweighted.
    """

    def __init__(
        self,
        cutoff_sr: float,
        max_Za: int = 100,
        envelope: str = "polynomial",
        envelope_p: int = 6,
        attenuation: str = "none",
        trainable_attenuation: bool = True,
    ) -> None:
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if envelope not in ("polynomial", "none"):
            raise ValueError(
                f"envelope must be 'polynomial' or 'none', got {envelope!r}"
            )
        if attenuation not in ("covalent", "none"):
            raise ValueError(
                f"attenuation must be 'covalent' or 'none', got {attenuation!r}"
            )
        if int(envelope_p) < 2 and envelope == "polynomial":
            raise ValueError(
                f"polynomial envelope_p must be >= 2 so the force is continuous, got {envelope_p}"
            )
        fields = {"rbf", "Dij_sr"}
        if attenuation == "covalent":
            fields |= {"Za", "idx_i_sr", "idx_j_sr"}
        super().__init__(input_fields=fields, output_fields={"rbf"})
        self.cutoff_sr = float(cutoff_sr)
        self.max_Za = int(max_Za)
        self.envelope = envelope
        self.envelope_p = int(envelope_p)
        self.attenuation = attenuation
        self.trainable_attenuation = bool(trainable_attenuation)
        if self.attenuation == "covalent":
            self.register_buffer(
                "covalent_radii", covalent_radii_table(self.max_Za)
            )
            # IBM Geodite-MP init. softplus(beta=0.5) maps these to p ≈ 1, q ≈ 4.58.
            p_raw = torch.tensor(-15.2013)
            q_raw = torch.tensor(3.2139)
            if self.trainable_attenuation:
                self.p_unconstrained = Parameter(p_raw)
                self.q_unconstrained = Parameter(q_raw)
            else:
                self.register_buffer("p_unconstrained", p_raw)
                self.register_buffer("q_unconstrained", q_raw)

    def exponents(self) -> tuple[Tensor, Tensor]:
        if self.attenuation != "covalent":
            raise RuntimeError("exponents exist only for covalent attenuation")
        return constrained_attenuation_exponents(
            self.p_unconstrained, self.q_unconstrained
        )

    def get_rbf(
        self,
        rbf: Tensor,
        Dij_sr: Tensor,
        Za: Optional[Tensor] = None,
        idx_i_sr: Optional[Tensor] = None,
        idx_j_sr: Optional[Tensor] = None,
    ) -> Tensor:
        if rbf.shape[0] != Dij_sr.shape[0]:
            raise ValueError(
                f"rbf has {rbf.shape[0]} edges but Dij_sr has {Dij_sr.shape[0]}"
            )
        window = Dij_sr.new_ones(Dij_sr.shape)
        if self.envelope == "polynomial":
            window = window * polynomial_envelope(
                Dij_sr, self.cutoff_sr, self.envelope_p
            )
        if self.attenuation == "covalent":
            if Za is None or idx_i_sr is None or idx_j_sr is None:
                raise ValueError(
                    "covalent attenuation needs Za, idx_i_sr, and idx_j_sr"
                )
            radii = self.covalent_radii.to(dtype=Dij_sr.dtype)
            radius_sum = radii[Za[idx_i_sr]] + radii[Za[idx_j_sr]]
            p, q = self.exponents()
            window = window * covalent_distance_weight(Dij_sr, radius_sum, p, q)
        return rbf * window.unsqueeze(-1)
