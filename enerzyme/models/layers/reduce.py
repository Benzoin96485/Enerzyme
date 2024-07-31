from typing import Dict
from torch import nn, Tensor
from ..functional import segment_sum

class EnergyReduceLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        for k, v in net_input.items():
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                output["Ea"] += v
        output["E"] = segment_sum(net_input["Ea"], net_input["N"])
        return output