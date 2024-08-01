from typing import Dict, List, Union, Literal
import torch
from torch import nn, Tensor
from ...data import PERIODIC_TABLE


class AtomicAffineLayer(nn.Module):
    def __init__(
        self, 
        max_Za: int, 
        shifts: Dict[Literal["Ea", "Qa"], Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]],
        scales: Dict[Literal["Ea", "Qa"], Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]]
    ) -> None:
        super().__init__()
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

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        for name, shift in self.shifts.items():
            output[name] += shift.gather(0, output["Za"])
        for name, scale in self.scales.items():
            output[name] *= scale.gather(0, output["Za"])
        return output
        