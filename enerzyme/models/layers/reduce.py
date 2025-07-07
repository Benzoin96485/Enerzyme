from typing import Dict, List, Set
import torch
from torch import Tensor
from . import BaseFFLayer
from ..functional import segment_sum_coo


class EnergyReduceLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"Ea"}, output_fields={"E", "Ea"})

    def get_relevant_input_fields(self, net_input_fields: Set[str]) -> Set[str]:
        relevant_input_fields = set()
        for k in net_input_fields:
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                relevant_input_fields.add(k)
        if "batch_seg" in net_input_fields:
            relevant_input_fields.add("batch_seg")
        else:
            relevant_input_fields.add("Za")
        return relevant_input_fields

    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        Ea = relevant_input["Ea"]
        for k, v in relevant_input.items():
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                Ea = Ea + v
        if "batch_seg" in relevant_input:
            batch_seg = relevant_input["batch_seg"]
        else:
            batch_seg = torch.zeros_like(relevant_input["Za"])
        return {"E": segment_sum_coo(Ea, batch_seg), "Ea": Ea}


class ShallowEnsembleReduceLayer(BaseFFLayer):
    def __init__(self, reduce_mean: List[str]=[], var: List[str]=[], std: List[str]=[], relative_energy: bool=False, train_only: bool=False, eval_only: bool=False) -> None:
        super().__init__(
            input_fields=set(reduce_mean) | set(var) | set(std), 
            output_fields=set(reduce_mean) | set(
                [name + "_var" for name in reduce_mean]
            ) | set(
                [name + "_std" for name in reduce_mean]
            ),
            train_only=train_only,
            eval_only=eval_only
        )
        for name in reduce_mean:
            setattr(self, f"get_{name}", lambda x: x.mean(dim=-1))
        for name in var:
            if relative_energy and name.startswith("E"):
                setattr(self, f"get_{name}_var", lambda x: (x - x.mean(dim=0)).var(dim=-1, unbiased=True))
            else:
                setattr(self, f"get_{name}_var", lambda x: x.var(dim=-1, unbiased=True))
        for name in std:
            if relative_energy and name.startswith("E"):
                setattr(self, f"get_{name}_std", lambda x: (x - x.mean(dim=0)).std(dim=-1, unbiased=True))
            else:
                setattr(self, f"get_{name}_std", lambda x: x.std(dim=-1, unbiased=True))
