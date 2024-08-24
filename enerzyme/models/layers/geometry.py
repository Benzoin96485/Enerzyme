from typing import Optional, Dict, Literal
from torch import Tensor
import torch.nn.functional as F
from . import BaseFFLayer
from ..cutoff import CUTOFF_KEY_TYPE, CUTOFF_REGISTER


class DistanceLayer(BaseFFLayer):
    '''
    Compute the distance between atoms
    '''
    def __init__(self) -> None:
        super().__init__(input_fields={"Ra", "idx_i", "idx_j", "offsets"}, output_fields={"Dij", "vij"})
        self._with_vector = False

    def with_vector_on(self, vij_name: str="vij") -> None:
        self._with_vector = True
        self.reset_field_name(vij=vij_name)

    def get_output(self, Ra: Tensor, idx_i: Tensor, idx_j: Tensor, offsets: Optional[Tensor]=None) -> Dict[str, Tensor]:
        '''
        Compute the distance with atom pair indices

        Params:
        -----
        Ra: Float tensor of atom positions, shape [N * batch_size]

        idx_i: Long tensor of the first pair indices, shape [N_pair * batch_size]

        idx_j: Long tensor of the second pair indices, shape [N_pair * batch_size]

        offsets: Float tensor of distance offsets, shape [N_pair * batch_size]

        Returns:
        -----
        Dij: Float tensor of distances, shape [N_pair * batch_size]
        '''
        relevant_output = dict()
        if Ra.device.type == "cpu":  # indexing is faster on CPUs
            Ri = Ra[idx_i]
            Rj = Ra[idx_j]
        else:
            Ri = Ra.gather(0, idx_i.view(-1, 1).expand(-1, 3))
            Rj = Ra.gather(0, idx_j.view(-1, 1).expand(-1, 3))
        if offsets is not None:
            Rj_ = Rj + offsets
        else:
            Rj_ = Rj
        relevant_output["Dij"] = F.pairwise_distance(Ri, Rj_, eps=1e-15)
        if self._with_vector:
            relevant_output["vij"] = Rj_ - Ri
        return relevant_output


class RangeSeparationLayer(BaseFFLayer):
    def __init__(self, cutoff_sr, cutoff_fn: Optional[CUTOFF_KEY_TYPE]=None) -> None:
        super().__init__()
        self.cutoff_sr = cutoff_sr
        if cutoff_fn is not None:
            self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        else:
            self.cutoff_fn = None

    def get_output(self, Dij_lr: Tensor, idx_i_lr: Tensor, idx_j_lr: Tensor, vij_lr: Optional[Tensor]=None) -> Dict[
        Literal["Dij_sr", "idx_i_sr", "idx_j_sr", "vij_sr", "cutoff_values_sr"], Tensor
    ]:
        cutmask = Dij_lr < self.cutoff_sr
        relevant_output = {
            "Dij_sr": Dij_lr[cutmask],
            "idx_i_sr": idx_i_lr[cutmask],
            "idx_j_sr": idx_j_lr[cutmask],
        }
        if self.cutoff_fn is not None:
            relevant_output["cutoff_values_sr"] = self.cutoff_fn(relevant_output["Dij_sr"], self.cutoff_sr)
        if vij_lr is not None:
            relevant_output["vij_sr"] = vij_lr[cutmask]
        return relevant_output