from typing import Dict, List
import torch
from torch import Tensor
from torch.nn import Module
from ..functional import segment_sum_coo


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
        output["E"] = segment_sum_coo(net_input["Ea"], batch_seg)
        return output


class ShallowEnsembleReduceLayer(Module):
    def __init__(self, reduce_mean: List[str]=[], var: List[str]=[], std: List[str]=[], relative_energy: bool=False, train_only: bool=False, eval_only: bool=False) -> None:
        super().__init__()
        self.reduce_mean = reduce_mean
        self.var = var
        self.std = std
        self.relative_energy = relative_energy
        self.train_only = train_only
        self.eval_only = eval_only

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.train_only and not self.training:
            return net_input
        if self.eval_only and self.training:
            return net_input
        
        net_output = net_input.copy()
        for name in self.var:
            if self.relative_energy and name.startswith("E"):
                net_output[name + "_var"] = (net_input[name] - net_input[name].mean(dim=0)).var(dim=-1, unbiased=True)
            else:
                net_output[name + "_var"] = net_input[name].var(dim=-1, unbiased=True)
        for name in self.std:
            if self.relative_energy and name.startswith("E"):
                net_output[name + "_std"] = (net_input[name] - net_input[name].mean(dim=0)).std(dim=-1)
            else:
                net_output[name + "_std"] = net_input[name].std(dim=-1)
        for name in self.reduce_mean:
            net_output[name] = net_input[name].mean(dim=-1)
        return net_output
