"""Ziegler–Biersack–Littmark short-range repulsion.

Matches SpookyNet ``ZBLRepulsionEnergy`` (learnable softplus params) and
optionally SO3LR's extra hard switch that drives the term to zero by
``switch_off`` (default 1.5 Å in SO3LR stacks).

Coulomb prefactor
---------------
``kehalf = 0.5 * Bohr_in_R * Hartree_in_E`` when ``Hartree_in_E`` is set via
build params (Ha/Å → ~0.265; eV/Å → ~7.2). If neither ``ke`` nor a non-default
unit path is intended, pass ``ke=14.399...`` explicitly (SpookyNet / SO3LR eV).

Historical bug: older defaults with ``Hartree_in_E=1`` while comparing to
SpookyNet's hard-coded eV ``ke`` produced ~27× too-small energies; parity
tests had to patch ``kehalf``. Prefer setting units correctly or ``ke``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch_scatter import segment_sum_coo

from ..cutoff import CUTOFF_KEY_TYPE, CUTOFF_REGISTER
from ..functional import softplus_inverse
from . import BaseFFLayer

# SpookyNet / SO3LR default Coulomb constant (eV·Å/e²).
_KE_EV_ANG = 14.399645351950548
_BOHR = 0.5291772105638411


class ZBLRepulsionEnergyLayer(BaseFFLayer):
    """ZBL-inspired short-range nuclear repulsion (SpookyNet + optional SO3LR switch)."""

    def __init__(
        self,
        Bohr_in_R: float = _BOHR,
        Hartree_in_E: Optional[float] = None,
        ke: Optional[float] = None,
        cutoff_sr: Optional[float] = None,
        cutoff_fn: Optional[CUTOFF_KEY_TYPE] = None,
        switch_off: Optional[float] = None,
    ) -> None:
        """
        Args:
            Bohr_in_R / Hartree_in_E: unit conversion; ``kehalf = 0.5 * Bohr * Hartree``.
            ke: optional explicit Coulomb constant (overrides Bohr/Hartree). SpookyNet
                and SO3LR (eV) use ``14.399645351950548``.
            switch_off: if set (SO3LR: 1.5), multiply by smooth switch on ``[0, switch_off]``.
                ``None`` keeps SpookyNet behaviour (envelope only via ``cutoff_values_sr``).
        """
        super().__init__(output_fields={"E_zbl_a"})
        self.a0 = Bohr_in_R
        if ke is not None:
            self.kehalf = 0.5 * float(ke)
        elif Hartree_in_E is not None:
            self.kehalf = 0.5 * Bohr_in_R * float(Hartree_in_E)
        else:
            self.kehalf = 0.5 * _KE_EV_ANG
        self.switch_off = switch_off
        if cutoff_fn is not None:
            self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
            self.cutoff_sr = cutoff_sr
        if switch_off is not None:
            self._switch = CUTOFF_REGISTER["smooth"]
        self.register_parameter("_adiv", Parameter(torch.Tensor(1)))
        self.register_parameter("_apow", Parameter(torch.Tensor(1)))
        self.register_parameter("_c1", Parameter(torch.Tensor(1)))
        self.register_parameter("_c2", Parameter(torch.Tensor(1)))
        self.register_parameter("_c3", Parameter(torch.Tensor(1)))
        self.register_parameter("_c4", Parameter(torch.Tensor(1)))
        self.register_parameter("_a1", Parameter(torch.Tensor(1)))
        self.register_parameter("_a2", Parameter(torch.Tensor(1)))
        self.register_parameter("_a3", Parameter(torch.Tensor(1)))
        self.register_parameter("_a4", Parameter(torch.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters to the default ZBL potential."""
        init.constant_(self._adiv, softplus_inverse(1 / (0.8854 * self.a0)))
        init.constant_(self._apow, softplus_inverse(0.23))
        init.constant_(self._c1, softplus_inverse(0.18180))
        init.constant_(self._c2, softplus_inverse(0.50990))
        init.constant_(self._c3, softplus_inverse(0.28020))
        init.constant_(self._c4, softplus_inverse(0.02817))
        init.constant_(self._a1, softplus_inverse(3.20000))
        init.constant_(self._a2, softplus_inverse(0.94230))
        init.constant_(self._a3, softplus_inverse(0.40280))
        init.constant_(self._a4, softplus_inverse(0.20160))

    def get_E_zbl_a(
        self,
        Za: Tensor,
        Dij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Tensor:
        if cutoff_values_sr is None:
            cutoff_values_sr = self.cutoff_fn(Dij_sr, cutoff=self.cutoff_sr)
        Zf = Za.type_as(self._a1)
        z = Zf ** F.softplus(self._apow)
        a = (z[idx_i_sr] + z[idx_j_sr]) * F.softplus(self._adiv)
        a1 = F.softplus(self._a1) * a
        a2 = F.softplus(self._a2) * a
        a3 = F.softplus(self._a3) * a
        a4 = F.softplus(self._a4) * a
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        csum = c1 + c2 + c3 + c4
        c1, c2, c3, c4 = c1 / csum, c2 / csum, c3 / csum, c4 / csum
        zizj = Zf[idx_i_sr] * Zf[idx_j_sr]
        f = (
            c1 * torch.exp(-a1 * Dij_sr)
            + c2 * torch.exp(-a2 * Dij_sr)
            + c3 * torch.exp(-a3 * Dij_sr)
            + c4 * torch.exp(-a4 * Dij_sr)
        ) * cutoff_values_sr
        if self.switch_off is not None:
            # SO3LR: smooth decay to ~0 by switch_off (Å), on top of NN cutoff envelope.
            f = f * self._switch(Dij_sr, self.switch_off, 0.0)
        return segment_sum_coo(
            self.kehalf * f * zizj / Dij_sr.clamp_min(1e-6),
            idx_i_sr,
            dim_size=len(Za),
        )
