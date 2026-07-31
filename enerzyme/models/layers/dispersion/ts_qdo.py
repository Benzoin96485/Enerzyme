"""Tkatchenko–Scheffler + vdW-QDO dispersion (SO3LR).

Distinct from Grimme D3/D4. Requires Hirshfeld volume ratios ``ha``.
Reference: DOI 10.1021/acs.jctc.3c00797; Kabylda et al., JACS 2025.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor
from torch_scatter import segment_sum_coo

from ...cutoff import CUTOFF_REGISTER
from .. import BaseFFLayer

_BOHR = 0.5291772105638411
_HARTREE = 27.211386245988
_FINE_STRUCTURE = 0.0072973525693

# Free-atom polarizabilities and C6 (mlff / SO3LR), Z = 1..
_ALPHAS = [
    4.5, 1.38, 164.2, 38.0, 21.0, 12.0, 7.4, 5.4, 3.8, 2.67, 162.7, 71.0,
    60.0, 37.0, 25.0, 19.6, 15.0, 11.1, 292.9, 160.0, 120.0, 98.0, 84.0, 78.0,
    63.0, 56.0, 50.0, 48.0, 42.0, 40.0, 60.0, 41.0, 29.0, 25.0, 20.0, 16.8,
    319.2, 199.0, 126.74, 119.97, 101.6, 88.42, 80.08, 65.89, 56.1, 23.68,
    50.6, 39.7, 70.22, 55.95, 43.67, 37.65, 35.0, 27.3, 399.9, 275.0, 213.7,
    204.7, 215.8, 208.4, 200.2, 192.1, 184.2, 158.3, 169.5, 164.64, 156.3,
    150.2, 144.3, 138.9, 137.2, 99.52, 82.53, 71.04, 63.04, 55.06, 42.51,
    39.68, 36.5, 33.9, 69.92, 61.8, 49.02, 45.01, 38.93, 33.54, 317.8, 246.2,
    203.3, 217.0, 154.4, 127.8, 150.5, 132.2, 131.2, 143.6, 125.3, 121.5,
    117.5, 113.4, 109.4, 105.4,
]
_C6_COEF = [
    6.5, 1.46, 1387.0, 214.0, 99.5, 46.6, 24.2, 15.6, 9.52, 6.38, 1556.0,
    627.0, 528.0, 305.0, 185.0, 134.0, 94.6, 64.3, 3897.0, 2221.0, 1383.0,
    1044.0, 832.0, 602.0, 552.0, 482.0, 408.0, 373.0, 253.0, 284.0, 498.0,
    354.0, 246.0, 210.0, 162.0, 129.6, 4691.0, 3170.0, 1968.58, 1677.91,
    1263.61, 1028.73, 1390.87, 609.75, 469.0, 157.5, 339.0, 452.0, 707.05,
    587.42, 459.32, 396.0, 385.0, 285.9, 6846.0, 5727.0, 3884.5, 3708.33,
    3911.84, 3908.75, 3847.68, 3708.69, 3511.71, 2781.53, 3124.41, 2984.29,
    2839.95, 2724.12, 2576.78, 2387.53, 2371.8, 1274.8, 1019.92, 847.93,
    710.2, 596.67, 359.1, 347.1, 298.0, 392.0, 717.44, 697.0, 571.0, 530.92,
    457.53, 390.63, 4224.44, 4851.32, 3604.41, 4047.54, 2876.77, 2375.89,
    3102.12, 2820.47, 2794.0, 3150.95, 2756.0, 2702.57, 2626.59, 2548.62,
    2468.69, 2386.8,
]


class TSQDODispersionEnergyLayer(BaseFFLayer):
    """TS + vdW-QDO dispersion with Hirshfeld ratios (SO3LR)."""

    def __init__(
        self,
        dispersion_energy_scale: float = 1.2,
        cutoff_lr: Optional[float] = None,
        cutoff_lr_damping: Optional[float] = None,
        neighborlist_format_lr: Literal["sparse", "ordered_sparse"] = "sparse",
    ) -> None:
        super().__init__(
            input_fields={"Za", "ha", "Dij_lr", "idx_i", "idx_j"},
            output_fields={"E_disp_a"},
        )
        if neighborlist_format_lr not in {"sparse", "ordered_sparse"}:
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        if cutoff_lr is not None and cutoff_lr_damping is None:
            raise ValueError(
                "cutoff_lr_damping is required when cutoff_lr is set "
                f"(got cutoff_lr={cutoff_lr})"
            )
        self.gamma_scale = float(dispersion_energy_scale)
        self.cutoff_lr = cutoff_lr
        self.cutoff_lr_damping = cutoff_lr_damping
        self.pair_factor = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        self._switch = CUTOFF_REGISTER["smooth"]
        self.register_buffer(
            "alphas", torch.tensor(_ALPHAS, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "c6_coef", torch.tensor(_C6_COEF, dtype=torch.float64), persistent=False
        )

    def _mixing(
        self, Za: Tensor, idx_i: Tensor, idx_j: Tensor, ha: Tensor
    ) -> tuple[Tensor, Tensor]:
        dtype = ha.dtype
        alphas = self.alphas.to(device=ha.device, dtype=dtype)
        c6 = self.c6_coef.to(device=ha.device, dtype=dtype)
        zi = Za[idx_i] - 1
        zj = Za[idx_j] - 1
        hi = ha[idx_i]
        hj = ha[idx_j]
        alpha_i = alphas[zi] * hi
        alpha_j = alphas[zj] * hj
        c6_i = c6[zi] * hi.square()
        c6_j = c6[zj] * hj.square()
        alpha_ij = 0.5 * (alpha_i + alpha_j)
        c6_ij = (
            2.0
            * c6_i
            * c6_j
            * alpha_i
            * alpha_j
            / (alpha_i.square() * c6_j + alpha_j.square() * c6_i)
        )
        return alpha_ij, c6_ij

    @staticmethod
    def _gamma_cubic_fit(alpha: Tensor) -> Tensor:
        vdW = (_FINE_STRUCTURE ** (-4.0 / 21.0)) * alpha ** (1.0 / 7.0)
        b0 = alpha.new_tensor(-0.00433008)
        b1 = alpha.new_tensor(0.24428889)
        b2 = alpha.new_tensor(0.04125273)
        b3 = alpha.new_tensor(-0.00078893)
        sigma = b3 * vdW.pow(3) + b2 * vdW.square() + b1 * vdW + b0
        return 0.5 / sigma.square()

    def get_E_disp_a(
        self, Za: Tensor, ha: Tensor, Dij_lr: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        if ha.dim() > 1:
            ha = ha.squeeze(-1)
        r = Dij_lr.reshape(-1)
        alpha_ij, c6_ij = self._mixing(Za, idx_i, idx_j, ha)
        gamma = self._gamma_cubic_fit(alpha_ij)
        r_au = r / _BOHR
        c8 = 5.0 / gamma * c6_ij
        c10 = 245.0 / 8.0 / gamma.square() * c6_ij
        p = self.gamma_scale * 2.0 * 2.54 * alpha_ij ** (1.0 / 7.0)
        v = (
            -c6_ij / (r_au.pow(6) + p.pow(6))
            - c8 / (r_au.pow(8) + p.pow(8))
            - c10 / (r_au.pow(10) + p.pow(10))
        )
        edge = self.pair_factor * v * _HARTREE
        if self.cutoff_lr is not None:
            w = self._switch(
                r,
                float(self.cutoff_lr),
                float(self.cutoff_lr) - float(self.cutoff_lr_damping),
            )
            edge = edge * torch.where(r > 0, w, torch.zeros_like(w))
        return segment_sum_coo(edge, idx_i, dim_size=len(Za))
