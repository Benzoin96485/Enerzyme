from typing import Dict, Literal, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd.function import once_differentiable
from . import BaseFFLayer
from ..functional import segment_sum_coo
from ..cutoff import CUTOFF_KEY_TYPE, CUTOFF_REGISTER


def _inverse_distance_block(Ra: Tensor, start: int, end: int, batch_seg: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    '''
    Rows start:end of the all-pairs inverse distance matrix (zero on the diagonal and
    between different batch segments), together with the displacements R_i - R_j.
    '''
    diff = Ra[start:end].unsqueeze(1) - Ra.unsqueeze(0)
    inv_r = diff.square().sum(-1).rsqrt_()
    rows = torch.arange(end - start, device=Ra.device)
    inv_r[rows, rows + start] = 0
    if batch_seg is not None:
        inv_r.masked_fill_(batch_seg[start:end].unsqueeze(1) != batch_seg.unsqueeze(0), 0)
    return inv_r, diff


class DenseCoulombPotential(torch.autograd.Function):
    '''
    V_i = sum_{j != i} Q_j / |R_i - R_j| over all atom pairs (within the same batch
    segment), evaluated in row blocks so that memory is O(block * N) instead of
    materializing a pair list. Q may carry trailing channels, e.g. [N, M] for a
    shallow ensemble.
    '''
    @staticmethod
    def forward(ctx, Ra: Tensor, Q: Tensor, batch_seg: Optional[Tensor], block_size: int) -> Tensor:
        Q2 = Q.reshape(Q.shape[0], -1)
        V = torch.empty_like(Q2)
        N = Ra.shape[0]
        for start in range(0, N, block_size):
            end = min(N, start + block_size)
            inv_r, _ = _inverse_distance_block(Ra, start, end, batch_seg)
            V[start:end] = inv_r @ Q2
        ctx.save_for_backward(Ra, Q2, batch_seg)
        ctx.block_size = block_size
        ctx.q_shape = Q.shape
        return V.view(Q.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_V: Tensor):
        Ra, Q2, batch_seg = ctx.saved_tensors
        G = grad_V.reshape(Q2.shape)
        N = Ra.shape[0]
        grad_R = torch.zeros_like(Ra) if ctx.needs_input_grad[0] else None
        grad_Q = torch.empty_like(Q2) if ctx.needs_input_grad[1] else None
        for start in range(0, N, ctx.block_size):
            end = min(N, start + ctx.block_size)
            inv_r, diff = _inverse_distance_block(Ra, start, end, batch_seg)
            if grad_Q is not None:
                # the inverse distance matrix is symmetric
                grad_Q[start:end] = inv_r @ G
            if grad_R is not None:
                # L = sum_ij G_i w_ij Q_j, dw_ij/dR_i = -w_ij^3 (R_i - R_j)
                A = G[start:end] @ Q2.T + Q2[start:end] @ G.T
                B = A.mul_(inv_r.pow(3))
                grad_R[start:end] = torch.einsum("ij,ijc->ic", B, diff).neg_()
        return grad_R, (grad_Q.view(ctx.q_shape) if grad_Q is not None else None), None, None


def dense_coulomb_potential(Ra: Tensor, Q: Tensor, batch_seg: Optional[Tensor]=None, max_block_elements: int=1 << 24) -> Tensor:
    N = Ra.shape[0]
    block_size = max(1, min(N, max_block_elements // max(N, 1)))
    return DenseCoulombPotential.apply(Ra, Q, batch_seg, block_size)


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


class ElectrostaticEnergyLayer(BaseFFLayer):
    def __init__(
        self, cutoff_sr: float, cutoff_lr: Optional[float]=None, 
        Bohr_in_R: float=0.5291772108, Hartree_in_E: float=1, dielectric_constant: float=1,
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
        super().__init__(input_fields={"Dij_lr", "Qa", "idx_i", "idx_j", "Ra", "batch_seg"}, output_fields={"E_ele_a"})
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E
        # "pairs": sum over the provided (long-range) pair list
        # "dense": 1/r over all atom pairs + short-range shielding correction over the provided pair list,
        # which must contain every pair closer than `self.cutoff`; inference only and requires no cutoff_lr
        self.lr_mode: Literal["pairs", "dense"] = "pairs"
        if flavor == "PhysNet":
            self.cutoff = cutoff_sr / 2
            self.cuton = 0
        elif flavor == "SpookyNet":
            self.cutoff = cutoff_sr * 0.75
            self.cuton = cutoff_sr * 0.25
        self.cutoff_lr = cutoff_lr
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        self.dielectric_constant = dielectric_constant
        
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
            torch.where(condition, 1.0 / Dij_shield + Dij * self.rcutconstant - self.cutconstant, zeros),
            condition, zeros
        )

    @property
    def supports_dense_lr(self) -> bool:
        return self.cutoff_lr is None or self.cutoff_lr <= 0

    def _get_dense_E_ele_a(
        self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor, Ra: Tensor, batch_seg: Optional[Tensor]
    ) -> Tensor:
        '''
        Same energy as the "pairs" mode with a full pair list, using
        switch * shielded + (1 - switch) * ordinary = ordinary + switch * (shielded - ordinary),
        where the switch vanishes beyond `self.cutoff`.
        '''
        if self.training:
            raise RuntimeError("Dense long-range electrostatics is an inference-only path")
        if not self.supports_dense_lr:
            raise RuntimeError("Dense long-range electrostatics does not support cutoff_lr")
        view_shape = (-1, 1) if Qa.dim() > 1 else (-1,)
        if Qa.device.type == "cpu" or Qa.dim() > 1:
            Qj = Qa[idx_j]
        else:
            Qj = Qa.gather(0, idx_j)
        switch = self.cutoff_fn(Dij_lr, self.cutoff, self.cuton)
        correction = (switch * (1.0 / self._shield(Dij_lr) - 1.0 / Dij_lr)).view(view_shape) * Qj
        V = dense_coulomb_potential(Ra, Qa, batch_seg) + segment_sum_coo(correction, idx_i, dim_size=len(Qa))
        return self.kehalf / self.dielectric_constant * Qa * V

    def get_E_ele_a(
        self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor,
        Ra: Optional[Tensor]=None, batch_seg: Optional[Tensor]=None
    ) -> Tensor:
        '''
        Compute the atomic electrostatic energy

        Params:
        -----
        Dij: Float tensor of pair distances, shape [N_pair * batch_size]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        idx_i: Long tensor of the first indices of pairs, shape [N_pair * batch_size]

        idx_j: Long tensor of the second indices of pairs, shape [N_pair * batch_size]

        Ra: Float tensor of atom positions, shape [N * batch_size, 3], only used in the "dense" mode

        batch_seg: Long tensor of batch indices, shape [N * batch_size], only used in the "dense" mode

        Returns:
        -----
        Ea: Float tensor of atomic electrostatic energy, shape [N * batch_size]
        '''
        if self.lr_mode == "dense":
            return self._get_dense_E_ele_a(Dij_lr, Qa, idx_i, idx_j, Ra, batch_seg)
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
