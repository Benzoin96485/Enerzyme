from typing import Dict
import torch
from torch import nn
from ..functional import smooth_cutoff_function, segment_sum


class ChargeConservationLayer(nn.Module):
    """
    Correct the atomic charges to make their summation equal to the total charge by [1]

    q^{corrected}_i = q_i - 1 / N (\sum_{j=1}^N q_j - Q)

    References:
    -----
    [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_corrected_Qa(
        self, 
        Za: torch.Tensor, Qa: torch.Tensor, 
        Q: torch.Tensor=None, batch_seg: torch.Tensor=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
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
            batch_seg = torch.zeros_like(Za)
        #number of atoms per batch (needed for charge scaling)
        N_per_batch = segment_sum(torch.ones_like(batch_seg), batch_seg)
        raw_Q = segment_sum(Qa, batch_seg)
        if Q is None: #assume desired total charge zero if not given
            Q = torch.zeros_like(N_per_batch)
        #return scaled charges (such that they have the desired total charge)
        return {
            "Qa": Qa + ((Q - raw_Q) / N_per_batch).gather(0, batch_seg), 
            "Q": raw_Q
        }
    
    def forward(self, **net_input: dict) -> dict:
        output = net_input.copy()
        output.update(self.get_corrected_Qa(**net_input))
        return output


class ElectrostaticEnergyLayer(nn.Module):
    """
    Calculate the electrostatic energy from distributed multipoles and atomic positions

    Params:
    -----
    kehalf: a half of the electrostatic force constant k (1/(4\pi\epsilon_0)) 
    multiplied by the element charge e at the current unit system.

    short_range_cutoff: the cutoff of short range interaction, the Coulomb's law at long-range
    and a damped term at short-range to avoid the singularity at r = 0 are smoothly interpolated by
    \phi [1]:

    \chi(r) = \phi(2r) + 1 / \sqrt{r^2 + 1} + (1 - \phi(2r)) / r

    long_range_cutoff: the cutoff of long range interaction, outside which the electrostatics are ignored

    References:
    -----
    [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
    """
    def __init__(self, kehalf: float, short_range_cutoff: float, long_range_cutoff: float) -> None:
        super().__init__()
        self.kehalf = kehalf
        self.sr_cutoff = short_range_cutoff
        self.lr_cutoff = long_range_cutoff
        if long_range_cutoff is not None and long_range_cutoff > 0:
            self.lr_cutoff2 = self.lr_cutoff * self.lr_cutoff

    def get_E_ele_a(self, Dij: torch.Tensor, Qa: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor, **kwargs) -> torch.Tensor:
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
        Qi = Qa.gather(0, idx_i)
        Qj = Qa.gather(0, idx_j)
        Dij_shielded = torch.sqrt(Dij * Dij + 1.0)
        switch = smooth_cutoff_function(Dij, self.sr_cutoff / 2, "poly")
        cswitch = 1 - switch
        if self.lr_cutoff is None or self.lr_cutoff <= 0:
            Eele_ordinary = 1.0 / Dij
            Eele_shielded = 1.0 / Dij_shielded
            Eele = self.kehalf * Qi * Qj * (switch * Eele_shielded + cswitch * Eele_ordinary)
        else:
            Eele_ordinary = 1.0 / Dij + Dij / self.lr_cutoff2 - 2.0 / self.lr_cutoff
            Eele_shielded = 1.0 / Dij_shielded + Dij_shielded / self.lr_cutoff2 - 2.0 / self.lr_cutoff
            #combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf * Qi * Qj * (switch * Eele_shielded + cswitch * Eele_ordinary)
            Eele = torch.where(Dij <= self.lr_cutoff, Eele, torch.zeros_like(Eele))
        return segment_sum(Eele, idx_i)
    
    def forward(self, **net_input: dict) -> dict:
        output = net_input.copy()
        output["E_ele_a"] = self.get_E_ele_a(**net_input)
        return output


class AtomicCharge2DipoleLayer(nn.Module):
    pass