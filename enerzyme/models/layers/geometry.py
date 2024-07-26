from typing import Dict
import torch
from torch import Tensor
from torch import nn


class DistanceLayer(nn.Module):
    '''
    Compute the distance between atoms
    '''
    def __init__(self) -> None:
        super().__init__()

    def get_distance(self, Ra: Tensor, idx_i: Tensor, idx_j: Tensor, offsets: Tensor=None, **kwargs) -> Tensor:
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
        Ri = Ra.gather(0, idx_i.view(-1, 1).expand(-1, 3))
        Rj = Ra.gather(0, idx_j.view(-1, 1).expand(-1, 3))
        if offsets is not None:
            Rj += offsets
        Dij = torch.sqrt(torch.relu(torch.sum((Ri - Rj) ** 2, -1))) #relu prevents negative numbers in sqrt
        return Dij

    def forward(self, idx_i_name: str="idx_i", idx_j_name: str="idx_j", Dij_name: str="Dij", offsets_name: str="offsets", **net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output[Dij_name] = self.get_distance(
            Ra=net_input["Ra"],
            idx_i=net_input[idx_i_name],
            idx_j=net_input[idx_j_name],
            offsets=net_input.get(offsets_name, None)
        )
        return output