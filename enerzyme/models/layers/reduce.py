from typing import Dict
import torch
from torch import Tensor
from torch.nn import Module
from ..functional import segment_sum

class EnergyReduceLayer(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        for k, v in net_input.items():
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                output["Ea"] += v
        if "batch_seg" in net_input:
            batch_seg = net_input["batch_seg"]
        else:
            batch_seg = torch.zeros_like(net_input["Za"])
        output["E"] = segment_sum(net_input["Ea"], batch_seg)
        return output