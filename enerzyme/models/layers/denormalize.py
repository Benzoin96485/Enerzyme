from typing import Dict, Union, Literal
import torch
from torch import nn, Tensor
from . import BaseFFLayer
from ...data import PERIODIC_TABLE


class AtomicAffineLayer(BaseFFLayer):
    def __init__(
        self, 
        max_Za: int, 
        shifts: Dict[Literal["Ea", "Qa"], Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]],
        scales: Dict[Literal["Ea", "Qa"], Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]]
    ) -> None:
        atomic_properties = shifts.keys() | scales.keys()
        super().__init__(input_fields={"Za"} | atomic_properties, output_fields=atomic_properties)
        self.max_Za = max_Za
        self.shifts = self.build_affine(shifts, 0)
        self.scales = self.build_affine(scales, 1)

    def build_affine(
        self, 
        params: Dict[Literal["Ea", "Qa"], Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]],
        default_value: float
    ) -> nn.ParameterDict:
        affine_dict = dict()
        for name, param in params.items():
            values = param["values"]
            if isinstance(values, dict):
                affine_param = torch.full((self.max_Za + 1,), float(default_value))
                for idx, value in values.items():
                    if isinstance(idx, str):
                        affine_param[PERIODIC_TABLE.loc[idx]["Za"]] = value
                    else:
                        affine_param[idx] = value
            else:
                affine_param = torch.full((self.max_Za + 1,), float(values))
            affine_dict[name] = nn.Parameter(affine_param, requires_grad=param["learnable"])
        return nn.ParameterDict(affine_dict)
    
    def get_Ea(self, Ea: Tensor, Qa: Tensor, Za: Tensor) -> Tensor:
        return Ea + self.shifts.Ea.gather(0, Za).view((-1, ) if Ea.dim() == 1 else (-1, 1))
    
    def get_Qa(self, Ea: Tensor, Qa: Tensor, Za: Tensor) -> Tensor:
        return Qa + self.shifts.Qa.gather(0, Za).view((-1, ) if Qa.dim() == 1 else (-1, 1))
    
    def _load_from_state_dict(self, state_dict: Dict[str, Tensor], *args, **kwargs):
        for k, v in state_dict.items():
            if k.endswith("shifts.Ea") or k.endswith("shifts.Qa") or k.endswith("scales.Ea") or k.endswith("scales.Qa"):
                if len(v) > self.max_Za + 1:
                    state_dict[k] = v[:self.max_Za + 1]
                elif len(v) < self.max_Za + 1:
                    if k.endswith("shifts.Ea"):
                        state_dict[k] = torch.concat([v, self.shifts.Ea[len(v):]], dim=0)
                    if k.endswith("shifts.Qa"):
                        state_dict[k] = torch.concat([v, self.shifts.Qa[len(v):]], dim=0)
                    if k.endswith("scales.Ea"):
                        state_dict[k] = torch.concat([v, self.scales.Ea[len(v):]], dim=0)
                    if k.endswith("scales.Qa"):
                        state_dict[k] = torch.concat([v, self.scales.Qa[len(v):]], dim=0)
        super()._load_from_state_dict(state_dict, *args, **kwargs)
        