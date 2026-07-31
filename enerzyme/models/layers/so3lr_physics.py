"""SO3LR universal pairwise force-field priors (Kabylda et al., JACS 2025).

These layers mirror So3krates-torch / mlff physics modules and are distinct from
the SpookyNet / PhysNet :class:`ZBLRepulsionEnergyLayer`,
:class:`ElectrostaticEnergyLayer`, and Grimme D3/D4 stacks. Keep the older
layers unchanged for SpookyNet / PhysNet parity; use these for SO3LR stacks.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch_scatter import segment_sum_coo

from ..functional import softplus_inverse
from . import BaseFFLayer

# ase.units / SO3LR conventions (Å, eV)
_BOHR = 0.5291772105638411
_HARTREE = 27.211386245988
_FINE_STRUCTURE = 0.0072973525693
_KE = 14.399645351950548

# Free-atom polarizabilities and C6 (mlff / SO3LR reference tables), Z = 1..
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


def switching_fn(x: Tensor, x_on: float, x_off: float) -> Tensor:
    """Smooth switch used by SO3LR ZBL / electrostatics / dispersion."""

    def _sigma(t: Tensor) -> Tensor:
        return torch.where(
            t > 0,
            torch.exp(-1.0 / t.clamp_min(1e-12)),
            torch.zeros_like(t),
        )

    c = (x - x_on) / (x_off - x_on)
    s1 = _sigma(1 - c)
    s0 = _sigma(c)
    return s1 / (s1 + s0 + 1e-12)


class SO3LRZBLRepulsionEnergyLayer(BaseFFLayer):
    """ZBL short-range repulsion with SO3LR 1.5 Å hard switch.

    Matches So3krates-torch ``ZBLRepulsion`` (cutoffs × switch[0, 1.5]).
    Distinct from SpookyNet :class:`ZBLRepulsionEnergyLayer`.
    """

    def __init__(
        self,
        ke: float = _KE,
        switch_off: float = 1.5,
    ) -> None:
        super().__init__(
            input_fields={
                "Za",
                "Dij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "cutoff_values_sr",
            },
            output_fields={"E_zbl_a"},
        )
        self.ke = ke
        self.switch_off = switch_off
        self.register_parameter("_a1", Parameter(torch.Tensor(1)))
        self.register_parameter("_a2", Parameter(torch.Tensor(1)))
        self.register_parameter("_a3", Parameter(torch.Tensor(1)))
        self.register_parameter("_a4", Parameter(torch.Tensor(1)))
        self.register_parameter("_c1", Parameter(torch.Tensor(1)))
        self.register_parameter("_c2", Parameter(torch.Tensor(1)))
        self.register_parameter("_c3", Parameter(torch.Tensor(1)))
        self.register_parameter("_c4", Parameter(torch.Tensor(1)))
        self.register_parameter("_p", Parameter(torch.Tensor(1)))
        self.register_parameter("_d", Parameter(torch.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self._a1, softplus_inverse(3.20000))
        init.constant_(self._a2, softplus_inverse(0.94230))
        init.constant_(self._a3, softplus_inverse(0.40280))
        init.constant_(self._a4, softplus_inverse(0.20160))
        init.constant_(self._c1, softplus_inverse(0.18180))
        init.constant_(self._c2, softplus_inverse(0.50990))
        init.constant_(self._c3, softplus_inverse(0.28020))
        init.constant_(self._c4, softplus_inverse(0.02817))
        init.constant_(self._p, softplus_inverse(0.23))
        init.constant_(self._d, softplus_inverse(1.0 / (0.8854 * _BOHR)))

    def get_E_zbl_a(
        self,
        Za: Tensor,
        Dij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        cutoff_values_sr: Tensor,
    ) -> Tensor:
        a1 = F.softplus(self._a1)
        a2 = F.softplus(self._a2)
        a3 = F.softplus(self._a3)
        a4 = F.softplus(self._a4)
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        p = F.softplus(self._p)
        d = F.softplus(self._d)
        c_sum = c1 + c2 + c3 + c4
        c1, c2, c3, c4 = c1 / c_sum, c2 / c_sum, c3 / c_sum, c4 / c_sum

        z_i = Za[idx_i_sr].to(dtype=Dij_sr.dtype)
        z_j = Za[idx_j_sr].to(dtype=Dij_sr.dtype)
        r = Dij_sr.clamp_min(1e-6)
        x = self.ke * cutoff_values_sr * (z_i * z_j) / r
        rzd = r * (torch.pow(z_i, p) + torch.pow(z_j, p)) * d
        y = (
            c1 * torch.exp(-a1 * rzd)
            + c2 * torch.exp(-a2 * rzd)
            + c3 * torch.exp(-a3 * rzd)
            + c4 * torch.exp(-a4 * rzd)
        )
        w = switching_fn(r, 0.0, self.switch_off)
        e_edge = 0.5 * w * x * y
        return segment_sum_coo(e_edge, idx_i_sr, dim_size=len(Za))


class ErfCoulombEnergyLayer(BaseFFLayer):
    """Erf-damped Coulomb electrostatics (SO3LR Eq. 6).

    ``electrostatic_energy_scale`` is σ in erf(r/σ)/r (pretrained SO3LR: 4.0).
    With ``cutoff_lr``, blends energy-shifted and force-shifted forms between
    ``0.45 * cutoff_lr`` and ``cutoff_lr`` (So3krates-torch
    ``CoulombErfShiftedForceSmooth``).
    """

    def __init__(
        self,
        electrostatic_energy_scale: float = 4.0,
        cutoff_lr: Optional[float] = None,
        ke: float = _KE,
        neighborlist_format_lr: Literal["sparse", "ordered_sparse"] = "sparse",
    ) -> None:
        super().__init__(
            input_fields={"Dij_lr", "Qa", "idx_i", "idx_j"},
            output_fields={"E_ele_a"},
        )
        if neighborlist_format_lr not in {"sparse", "ordered_sparse"}:
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        self.sigma = float(electrostatic_energy_scale)
        self.cutoff_lr = cutoff_lr
        self.ke = ke
        self.pair_factor = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        if cutoff_lr is not None and cutoff_lr > 0:
            self.cuton = 0.45 * float(cutoff_lr)

    @staticmethod
    def _potential(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return torch.erf(r / sigma) / r

    @staticmethod
    def _force(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return (
            2.0 * r * torch.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)
            - torch.erf(r / sigma)
        ) / (r ** 2)

    def get_E_ele_a(
        self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        if Qa.dim() > 1:
            Qa = Qa.squeeze(-1)
        r = Dij_lr.reshape(-1)
        qi = Qa[idx_i]
        qj = Qa[idx_j]
        pairwise = self._potential(r, self.sigma)
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            edge = self.pair_factor * self.ke * qi * qj * pairwise
        else:
            cutoff = float(self.cutoff_lr)
            f = switching_fn(r, self.cuton, cutoff)
            shift = self._potential(torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma)
            force_shift = self._force(
                torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma
            )
            energy_shifted = pairwise - shift
            force_shifted = pairwise - shift - force_shift * (r - cutoff)
            blended = f * energy_shifted + (1.0 - f) * force_shifted
            edge = self.pair_factor * self.ke * qi * qj * blended
            edge = torch.where(r < cutoff, edge, torch.zeros_like(edge))
        return segment_sum_coo(edge, idx_i, dim_size=len(Qa))


class TSQDODispersionEnergyLayer(BaseFFLayer):
    """Tkatchenko–Scheffler + vdW-QDO dispersion (SO3LR / DOI 10.1021/acs.jctc.3c00797).

    Requires Hirshfeld volume ratios ``ha``. Distinct from Grimme D3/D4.
    """

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
            2.0 * c6_i * c6_j * alpha_i * alpha_j
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
            w = switching_fn(
                r,
                float(self.cutoff_lr) - float(self.cutoff_lr_damping),
                float(self.cutoff_lr),
            )
            edge = edge * torch.where(r > 0, w, torch.zeros_like(w))
        return segment_sum_coo(edge, idx_i, dim_size=len(Za))
