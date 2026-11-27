"""Learnable screened ZBL pair repulsion.

This is the short-range term in Geodite (Reschützegger et al., 2026) and the
same functional form as MACE-MP-0b3: a ZBL-like screen with free coefficients,
cut off by a polynomial envelope at the sum of covalent radii. It is not the
SpookyNet :class:`~enerzyme.models.layers.zbl.ZBLRepulsionEnergyLayer`.

The neighbor list must contain both directions. Each directed edge carries
half of the pair energy, so the atomic sum is the full pair potential.
Energies use ``Bohr_in_R * Hartree_in_E`` as the Coulomb prefactor. With
``Hartree_in_E ≈ 27.211386245988`` that prefactor is 14.3996 eV·Å, the
Geodite-MP unit.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter

from ..mp.basis.envelope import polynomial_envelope
from ..mp.ops import scatter_sum
from . import BaseFFLayer
from .covalent import covalent_radii_table


class ScreenedZBLRepulsionEnergyLayer(BaseFFLayer):
    """Per-atom screened repulsion ``E_zbl_a``. ``EnergyReduce`` adds it to ``Ea``."""

    def __init__(
        self,
        max_Za: int = 100,
        Bohr_in_R: float = 0.5291772105638411,
        Hartree_in_E: float = 1.0,
        envelope_p: int = 5,
    ) -> None:
        if int(envelope_p) < 2:
            raise ValueError(f"envelope_p must be >= 2, got {envelope_p}")
        if float(Bohr_in_R) <= 0 or float(Hartree_in_E) <= 0:
            raise ValueError("Bohr_in_R and Hartree_in_E must be positive")
        super().__init__(
            input_fields={"Za", "Dij_sr", "idx_i_sr", "idx_j_sr"},
            output_fields={"E_zbl_a"},
        )
        self.max_Za = int(max_Za)
        self.bohr = float(Bohr_in_R)
        self.hartree = float(Hartree_in_E)
        self.envelope_p = int(envelope_p)
        self.register_buffer("covalent_radii", covalent_radii_table(self.max_Za))
        # Published Geodite initialization (learnable, not softplus-constrained).
        self.a_factor = Parameter(torch.tensor(0.4546))
        self.z_power = Parameter(torch.tensor(0.0842))
        self.screen_coefs = Parameter(torch.full((4,), 0.1663))
        self.screen_exponents = Parameter(torch.full((4,), 0.0354))

    def get_E_zbl_a(
        self,
        Za: Tensor,
        Dij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Tensor:
        energy = self.a_factor.new_zeros(Za.shape[0])
        if Dij_sr.numel() == 0:
            return energy
        atomic_number = Za.to(dtype=energy.dtype)
        left = atomic_number[idx_i_sr]
        right = atomic_number[idx_j_sr]
        screening_length = self.a_factor * self.bohr / (
            left.pow(self.z_power) + right.pow(self.z_power)
        )
        reduced = (Dij_sr / screening_length.clamp_min(1e-8)).unsqueeze(-1)
        phi = (
            self.screen_coefs.view(1, -1)
            * torch.exp(-self.screen_exponents.view(1, -1) * reduced)
        ).sum(dim=-1)
        radii = self.covalent_radii.to(dtype=energy.dtype)
        radius_sum = radii[Za[idx_i_sr]] + radii[Za[idx_j_sr]]
        envelope = polynomial_envelope(
            Dij_sr, radius_sum.clamp_min(1e-6), self.envelope_p
        )
        # 0.5 because the graph stores ij and ji.
        pair = (
            0.5
            * self.bohr
            * self.hartree
            * left
            * right
            / Dij_sr.clamp_min(1e-8)
            * phi
            * envelope
        )
        return scatter_sum(pair, idx_i_sr, Za.shape[0])


class StandardZBLRepulsionEnergyLayer(BaseFFLayer):
    """Fixed ZBL wall from AceFF-2-OMOL / the Orb pair repulsion.

    Screening length ``0.4543 a0 / (Z_a^{0.3} + Z_b^{0.3})``, the standard
    four-term ZBL polynomial, and a polynomial envelope of order 6 that
    vanishes at the sum of covalent radii. The neighbor list stores both
    directions; each directed edge carries half of the pair. This is not the
    learnable Geodite screen in :class:`ScreenedZBLRepulsionEnergyLayer`.
    The third exponent is 0.4029, the value in TorchMD-Net's ``RepulsionZBL``.
    ``EnergyReduce`` adds ``E_zbl_a``. Set ``Hartree_in_E`` near 27.211386245988
    when the dataset energy is eV; the prefactor is then 14.3996 eV·Å.
    """

    def __init__(
        self,
        max_Za: int = 94,
        Bohr_in_R: float = 0.5291772105638411,
        Hartree_in_E: float = 1.0,
        envelope_p: int = 6,
    ) -> None:
        if int(max_Za) < 1:
            raise ValueError(f"max_Za must be >= 1, got {max_Za}")
        if int(envelope_p) < 2:
            raise ValueError(f"envelope_p must be >= 2, got {envelope_p}")
        if float(Bohr_in_R) <= 0 or float(Hartree_in_E) <= 0:
            raise ValueError("Bohr_in_R and Hartree_in_E must be positive")
        super().__init__(
            input_fields={"Za", "Dij_sr", "idx_i_sr", "idx_j_sr"},
            output_fields={"E_zbl_a"},
        )
        self.max_Za = int(max_Za)
        self.bohr = float(Bohr_in_R)
        self.hartree = float(Hartree_in_E)
        self.envelope_p = int(envelope_p)
        self.a_prefactor = 0.4543
        self.z_exponent = 0.3
        self.register_buffer("covalent_radii", covalent_radii_table(self.max_Za))
        self.register_buffer(
            "screen_coefs",
            torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]),
        )
        self.register_buffer(
            "screen_exponents",
            torch.tensor([3.2, 0.9423, 0.4029, 0.2016]),
        )

    def get_E_zbl_a(
        self,
        Za: Tensor,
        Dij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Tensor:
        energy = self.screen_coefs.new_zeros(Za.shape[0])
        if Dij_sr.numel() == 0:
            return energy
        if int(Za.max()) > self.max_Za or int(Za.min()) < 1:
            raise ValueError(
                f"Za must lie in 1..{self.max_Za}, got {int(Za.min())}..{int(Za.max())}"
            )
        atomic_number = Za.to(dtype=energy.dtype)
        left = atomic_number[idx_i_sr]
        right = atomic_number[idx_j_sr]
        screening_length = self.a_prefactor * self.bohr / (
            left.pow(self.z_exponent) + right.pow(self.z_exponent)
        )
        reduced = Dij_sr / screening_length.clamp_min(1e-8)
        phi = (
            self.screen_coefs.view(1, -1)
            * torch.exp(-self.screen_exponents.view(1, -1) * reduced.unsqueeze(-1))
        ).sum(dim=-1)
        radii = self.covalent_radii.to(dtype=energy.dtype)
        radius_sum = radii[Za[idx_i_sr]] + radii[Za[idx_j_sr]]
        envelope = polynomial_envelope(
            Dij_sr, radius_sum.clamp_min(1e-6), self.envelope_p
        )
        pair = (
            0.5
            * self.bohr
            * self.hartree
            * left
            * right
            / Dij_sr.clamp_min(1e-8)
            * phi
            * envelope
        )
        return scatter_sum(pair, idx_i_sr, Za.shape[0])
