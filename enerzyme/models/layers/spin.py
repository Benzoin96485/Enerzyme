from typing import Dict, Literal, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import BaseFFLayer
from ..functional import segment_sum_coo


class SpinConservationLayer(BaseFFLayer):
    def __init__(self) -> None:
        r"""
        Correct the atomic spin to make their summation equal to the total spin

        S^{corrected}_i = S_i - 1 / N (\sum_{j=1}^N S_j - S)
        """
        super().__init__()

    def get_output(
        self, Za: Tensor, Sa: Tensor, 
        S: Optional[Tensor]=None, batch_seg: Optional[Tensor]=None
    ) -> Dict[Literal["Sa", "S"], Tensor]:
        '''
        Correct the atomic spin

        Params:
        -----
        Za: Long tensor of atomic numbers, shape [N * batch_size]

        Sa: Float tensor of atomic spins, shape [N * batch_size]

        S: Float tensor of total spins, shape [batch_size]

        batch_seg: Long tensor of batch indices, shape [N * batch_size]

        Returns:
        -----
        Sa_corrected: Float tensor of corrected atomic spin, shape [N * batch_size]

        raw_S: Float tensor of total atomic spin before correction, shape [batch_size]
        '''
        if batch_seg is None:
            batch_seg = torch.zeros_like(Za, dtype=torch.long)
        #number of atoms per batch (needed for charge scaling)
        N_per_batch = segment_sum_coo(torch.ones_like(batch_seg), batch_seg)
        view_shape = (-1, ) if Sa.dim() == 1 else (-1, 1)
        raw_S = segment_sum_coo(Sa, batch_seg)
        if S is None: #assume desired total spin zero if not given
            S = torch.zeros_like(N_per_batch)
        #return scaled spins (such that they have the desired total spin)
        return {
            "Sa": Sa + ((S.view(view_shape) - raw_S) / N_per_batch.view(view_shape))[batch_seg], 
            "S": raw_S
        }