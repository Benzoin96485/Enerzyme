from typing import Dict, Literal, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from .. import segment_sum_coo
from . import BaseFFLayer
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
        raw_Q = segment_sum_coo(Qa, batch_seg)
        if Q is None: #assume desired total charge zero if not given
            Q = torch.zeros_like(N_per_batch)
        #return scaled charges (such that they have the desired total charge)
        return {
            "Qa": Qa + ((Q - raw_Q) / N_per_batch).gather(0, batch_seg), 
            "Q": raw_Q
        }


class ElectrostaticEnergyLayer(BaseFFLayer):
    def __init__(
        self, cutoff_sr: float, cutoff_lr: Optional[float]=None, 
        Bohr_in_R: float=0.5291772108, Hartree_in_E: float=1,
        cutoff_fn: CUTOFF_KEY_TYPE="smooth", flavor: Literal["PhysNet", "SpookyNet"]="SpookyNet"
    ) -> None:
        r"""
        Calculate the electrostatic energy from distributed multipoles and atomic positions

        Params:
        -----
        bohr_in_Ra: the numerical value of one Bohr in the unit of atom positions.

        Hartree_in_Ea: the numerical value of one Hartree in the unit of energy.

        short_range_cutoff: the cutoff of short range interaction, the Coulomb's law at long-range
        and a damped term at short-range to avoid the singularity at r = 0 are smoothly interpolated by
        \phi [1]:

        \chi(r) = \phi(2r) + 1 / \sqrt{r^2 + 1} + (1 - \phi(2r)) / r

        long_range_cutoff: the cutoff of long range interaction, outside which the electrostatics are ignored

        References:
        -----
        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        """
        super().__init__(input_fields={"Dij_lr", "Qa", "idx_i", "idx_j"}, output_fields={"E_ele_a"})
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E
        if flavor == "PhysNet":
            self.cutoff = cutoff_sr / 2
            self.cuton = 0
        elif flavor == "SpookyNet":
            self.cutoff = cutoff_sr * 0.75
            self.cuton = cutoff_sr * 0.25
        self.cutoff_lr = cutoff_lr
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        
        if cutoff_lr is not None and cutoff_lr > 0:
            self.cutoff_lr2 = self.cutoff_lr * self.cutoff_lr
            self.two_div_cut = 2.0 / self.cutoff_lr
            if flavor == "PhysNet":
                self.lr_shield = self._simple_lr_shield
            elif flavor == "SpookyNet":
                self.rcutconstant = self.cutoff_lr / (self.cutoff_lr ** 2 + 1.0) ** 1.5
                self.cutconstant = (2 * self.cutoff_lr ** 2 + 1.0) / (self.cutoff_lr** 2 + 1.0) ** 1.5
                self.lr_shield = self._smooth_lr_shield

    def _lr_ordinary(self, Dij: Tensor) -> Tensor:
        return 1.0 / Dij + Dij / self.lr_cutoff2 - self.two_div_cut

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
            torch.where(condition, self._lr_ordinary(1.0 / Dij_shield + Dij * self.rcutconstant - self.cutconstant), zeros),
            condition, zeros
        )

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
        if Qa.device.type == "cpu":
            fac = self.kehalf * Qa[idx_i] * Qa[idx_j]
        else:
            fac = self.kehalf * Qa.gather(0, idx_i) * Qa.gather(0, idx_j)
        switch = self.cutoff_fn(Dij_lr, self.cutoff, self.cuton)
        cswitch = 1 - switch
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            Eele_ordinary = 1.0 / Dij_lr
            Eele_shielded = 1.0 / self._shield(Dij_lr)
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary)
        else:
            Eele_ordinary, Eele_shielded, condition, zeros = self.lr_shield(Dij_lr)
            # combine shielded and ordinary interactions and apply prefactors
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary)
            Eele = torch.where(condition, Eele, zeros)
        return segment_sum_coo(Eele, idx_i, dim_size=len(Qa))


class AtomicCharge2DipoleLayer(Module):
    def __init__(self) -> None:
        super().__init__()

    def get_dipole(self, Qa: Tensor, Ra: Tensor, batch_seg: Optional[Tensor]=None, **kwargs) -> Tensor:
        if batch_seg is None:
            batch_seg = torch.zeros_like(Qa, dtype=torch.long)
        Pa = Qa.unsqueeze(1) * Ra
        return segment_sum_coo(Pa, batch_seg)

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output["M2"] = self.get_dipole(**net_input)
        return output
