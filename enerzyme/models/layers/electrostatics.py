from typing import Dict, Literal, Optional, Tuple
import math
import torch
from torch import Tensor
from . import BaseFFLayer
from ..functional import segment_sum_coo
from ..cutoff import CUTOFF_KEY_TYPE, CUTOFF_REGISTER


class ChargeConservationLayer(BaseFFLayer):
    def __init__(self) -> None:
        r"""
        Correct the atomic charges to make their summation equal to the total charge by [1]

        q^{corrected}_i = q_i - 1 / N (\sum_{j=1}^N q_j - Q)

        References:
        -----
        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        """
        super().__init__()

    def get_output(
        self, Za: Tensor, Qa: Tensor, 
        Q: Optional[Tensor]=None, batch_seg: Optional[Tensor]=None
    ) -> Dict[Literal["Qa", "Q"], Tensor]:
        '''
        Correct the atomic charge

        Params:
        -----
        Za: Long tensor of atomic numbers, shape [N * batch_size]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        Q: Float tensor of total charges, shape [batch_size]

        batch_seg: Long tensor of batch indices, shape [N * batch_size]

        Returns:
        -----
        Qa_corrected: Float tensor of corrected atomic charge, shape [N * batch_size]

        raw_Q: Float tensor of total atomic charge before correction, shape [batch_size]
        '''
        if batch_seg is None:
            batch_seg = torch.zeros_like(Za, dtype=torch.long)
        #number of atoms per batch (needed for charge scaling)
        N_per_batch = segment_sum_coo(torch.ones_like(batch_seg), batch_seg)
        view_shape = (-1, ) if Qa.dim() == 1 else (-1, 1)
        raw_Q = segment_sum_coo(Qa, batch_seg)
        if Q is None: #assume desired total charge zero if not given
            Q = torch.zeros_like(N_per_batch)
        #return scaled charges (such that they have the desired total charge)
        return {
            "Qa": Qa + ((Q.view(view_shape) - raw_Q) / N_per_batch.view(view_shape))[batch_seg], 
            "Q": raw_Q
        }


class VelocityConservationLayer(BaseFFLayer):
    """Per graph, subtract the mean so each channel sums to zero over atoms."""

    def __init__(self) -> None:
        super().__init__(
            input_fields={"batch_seg", "Q_vel_a", "S_vel_a"},
            output_fields={"Q_vel_a", "S_vel_a"},
        )

    def get_output(
        self, batch_seg: Tensor, Q_vel_a: Tensor, S_vel_a: Tensor
    ) -> Dict[str, Tensor]:
        ones = torch.ones(batch_seg.shape[0], dtype=Q_vel_a.dtype, device=Q_vel_a.device)
        N_per = segment_sum_coo(ones, batch_seg)
        mean_q = segment_sum_coo(Q_vel_a, batch_seg) / N_per
        mean_s = segment_sum_coo(S_vel_a, batch_seg) / N_per
        return {
            "Q_vel_a": Q_vel_a - mean_q[batch_seg],
            "S_vel_a": S_vel_a - mean_s[batch_seg],
        }


class ElectrostaticEnergyLayer(BaseFFLayer):
    def __init__(
        self,
        cutoff_sr: Optional[float] = None,
        cutoff_lr: Optional[float] = None,
        Bohr_in_R: float = 0.5291772108,
        Hartree_in_E: float = 1,
        dielectric_constant: float = 1,
        cutoff_fn: CUTOFF_KEY_TYPE = "smooth",
        flavor: Literal["PhysNet", "SpookyNet", "SO3LR"] = "SpookyNet",
        electrostatic_energy_scale: float = 4.0,
        neighborlist_format_lr: Literal["sparse", "ordered_sparse"] = "sparse",
    ) -> None:
        r"""
        Pairwise electrostatic energy.

        Flavors:
        - ``PhysNet`` / ``SpookyNet``: shielded Coulomb with smooth SR blend [1].
        - ``SO3LR``: erf-damped Coulomb ``erf(r/σ)/r`` (Kabylda et al., JACS 2025);
          ``electrostatic_energy_scale`` is σ (pretrained SO3LR: 4.0). With
          ``cutoff_lr``, blends energy- and force-shifted forms on
          ``[0.45·Rc, Rc]``.

        Params:
        -----
        Bohr_in_R / Hartree_in_E: unit conversion
            (``kehalf = 0.5 * Bohr * Hartree``).

        cutoff_sr: short-range blend cutoff (required for PhysNet/SpookyNet).

        cutoff_lr: long-range cutoff; ignored interactions beyond this distance.

        References:
        -----
        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        """
        super().__init__(input_fields={"Dij_lr", "Qa", "idx_i", "idx_j"}, output_fields={"E_ele_a"})
        self.flavor = flavor
        self.cutoff_lr = cutoff_lr
        self.dielectric_constant = dielectric_constant
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E

        if flavor == "SO3LR":
            if neighborlist_format_lr not in {"sparse", "ordered_sparse"}:
                raise ValueError(
                    "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
                )
            self.sigma = float(electrostatic_energy_scale)
            # sparse bidirectional list → kehalf; ordered_sparse → full ke = 2*kehalf
            self.pair_kehalf = (
                self.kehalf
                if neighborlist_format_lr == "sparse"
                else 2.0 * self.kehalf
            )
            self._switch = CUTOFF_REGISTER["smooth"]
            if cutoff_lr is not None and cutoff_lr > 0:
                self.cuton = 0.45 * float(cutoff_lr)
            return

        if cutoff_sr is None:
            raise TypeError("cutoff_sr is required for PhysNet/SpookyNet electrostatics")
        if flavor == "PhysNet":
            self.cutoff = cutoff_sr / 2
            self.cuton = 0
        elif flavor == "SpookyNet":
            self.cutoff = cutoff_sr * 0.75
            self.cuton = cutoff_sr * 0.25
        else:
            raise ValueError(f"Unknown electrostatic flavor: {flavor}")
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]

        if cutoff_lr is not None and cutoff_lr > 0:
            self.cutoff_lr2 = self.cutoff_lr * self.cutoff_lr
            self.two_div_cut = 2.0 / self.cutoff_lr
            if flavor == "PhysNet":
                self.lr_shield = self._simple_lr_shield
            else:
                self.rcutconstant = self.cutoff_lr / (self.cutoff_lr ** 2 + 1.0) ** 1.5
                self.cutconstant = (2 * self.cutoff_lr ** 2 + 1.0) / (self.cutoff_lr** 2 + 1.0) ** 1.5
                self.lr_shield = self._smooth_lr_shield

    def _lr_ordinary(self, Dij: Tensor) -> Tensor:
        return 1.0 / Dij + Dij / self.cutoff_lr2 - self.two_div_cut

    def _shield(self, Dij: Tensor) -> Tensor:
        return torch.sqrt(Dij * Dij + 1.0)

    def _simple_lr_shield(self, Dij: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Dij_shield = self._shield(Dij)
        zeros = torch.zeros_like(Dij)
        condition = Dij < self.cutoff_lr
        return (
            torch.where(condition, self._lr_ordinary(Dij), zeros), 
            torch.where(condition, self._lr_ordinary(Dij_shield), zeros), condition, zeros
        )

    def _smooth_lr_shield(self, Dij: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Dij_shield = self._shield(Dij)
        zeros = torch.zeros_like(Dij)
        condition = Dij < self.cutoff_lr
        return (
            torch.where(condition, self._lr_ordinary(Dij), zeros), 
            torch.where(condition, 1.0 / Dij_shield + Dij * self.rcutconstant - self.cutconstant, zeros),
            condition, zeros
        )

    @staticmethod
    def _erf_potential(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return torch.erf(r / sigma) / r

    @staticmethod
    def _erf_force(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return (
            2.0 * r * torch.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)
            - torch.erf(r / sigma)
        ) / (r ** 2)

    def _get_E_ele_a_so3lr(
        self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        if Qa.dim() > 1:
            Qa = Qa.squeeze(-1)
        r = Dij_lr.reshape(-1)
        qi = Qa[idx_i]
        qj = Qa[idx_j]
        pairwise = self._erf_potential(r, self.sigma)
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            edge = self.pair_kehalf * qi * qj * pairwise / self.dielectric_constant
        else:
            cutoff = float(self.cutoff_lr)
            f = self._switch(r, cutoff, self.cuton)
            shift = self._erf_potential(
                torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma
            )
            force_shift = self._erf_force(
                torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma
            )
            energy_shifted = pairwise - shift
            force_shifted = pairwise - shift - force_shift * (r - cutoff)
            blended = f * energy_shifted + (1.0 - f) * force_shifted
            edge = self.pair_kehalf * qi * qj * blended / self.dielectric_constant
            edge = torch.where(r < cutoff, edge, torch.zeros_like(edge))
        return segment_sum_coo(edge, idx_i, dim_size=len(Qa))

    def get_E_ele_a(self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        '''
        Compute the atomic electrostatic energy

        Params:
        -----
        Dij: Float tensor of pair distances, shape [N_pair * batch_size]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        idx_i: Long tensor of the first indices of pairs, shape [N_pair * batch_size]

        idx_j: Long tensor of the second indices of pairs, shape [N_pair * batch_size]

        Returns:
        -----
        Ea: Float tensor of atomic electrostatic energy, shape [N * batch_size]
        '''
        if self.flavor == "SO3LR":
            return self._get_E_ele_a_so3lr(Dij_lr, Qa, idx_i, idx_j)

        if Qa.device.type == "cpu" or Qa.dim() > 1:
            fac = self.kehalf * Qa[idx_i] * Qa[idx_j] / self.dielectric_constant
        else:
            fac = self.kehalf * Qa.gather(0, idx_i) * Qa.gather(0, idx_j) / self.dielectric_constant
        switch = self.cutoff_fn(Dij_lr, self.cutoff, self.cuton)
        cswitch = 1 - switch
        view_shape = (-1, 1) if Qa.dim() > 1 else (-1,)
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            Eele_ordinary = 1.0 / Dij_lr
            Eele_shielded = 1.0 / self._shield(Dij_lr)
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary).view(view_shape)
        else:
            Eele_ordinary, Eele_shielded, condition, zeros = self.lr_shield(Dij_lr)
            # combine shielded and ordinary interactions and apply prefactors
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary).view(view_shape)
            Eele = torch.where(condition, Eele, zeros)
        return segment_sum_coo(Eele, idx_i, dim_size=len(Qa))


class AtomicCharge2DipoleLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"Qa", "Ra", "batch_seg"}, output_fields={"M2"})

    def get_M2(self, Qa: Tensor, Ra: Tensor, batch_seg: Optional[Tensor]=None) -> Tensor:
        if batch_seg is None:
            batch_seg = torch.zeros_like(Qa, dtype=torch.long)
        Pa = Qa.unsqueeze(1) * Ra.view((-1, 3, 1) if Qa.dim() > 1 else (-1, 3))
        return segment_sum_coo(Pa, batch_seg)
