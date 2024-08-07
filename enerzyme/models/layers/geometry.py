from typing import Dict, Optional
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F


class DistanceLayer(Module):
    '''
    Compute the distance between atoms
    '''
    def __init__(self) -> None:
        super().__init__()

    def get_distance(self, Ra: Tensor, idx_i: Tensor, idx_j: Tensor, offsets: Optional[Tensor]=None, with_vector: bool=False, **kwargs) -> Tensor:
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
        if with_vector:
            return F.pairwise_distance(Ri, Rj_, eps=1e-15), Rj - Ri
        else:
            return F.pairwise_distance(Ri, Rj_, eps=1e-15)

    def forward(self, net_input: Dict[str, Tensor], idx_i_name: str="idx_i", idx_j_name: str="idx_j", Dij_name: str="Dij", offsets_name: str="offsets") -> Dict[str, Tensor]:
        output = net_input.copy()
        output[Dij_name] = self.get_distance(
            Ra=net_input["Ra"],
            idx_i=net_input[idx_i_name],
            idx_j=net_input[idx_j_name],
            offsets=net_input.get(offsets_name, None)
        )
        return output