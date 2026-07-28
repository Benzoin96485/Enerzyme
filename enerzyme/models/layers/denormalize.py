from typing import Dict, Union, Literal, Optional
import torch
from torch import nn, Tensor
from . import BaseFFLayer
from ...data.transform import PERIODIC_TABLE


class AtomicAffineLayer(BaseFFLayer):
    def __init__(
        self,
        max_Za: int,
        shifts: Optional[
            Dict[str, Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]]
        ] = None,
        scales: Optional[
            Dict[str, Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]]
        ] = None,
    ) -> None:
        if shifts is None:
            shifts = {
                "Ea": {"values": 0.0, "learnable": True},
                "Qa": {"values": 0.0, "learnable": True},
                "Sa": {"values": 0.0, "learnable": True},
            }
        if scales is None:
            scales = {
                "Ea": {"values": 1.0, "learnable": True},
                "Qa": {"values": 1.0, "learnable": True},
                "Sa": {"values": 1.0, "learnable": True},
            }

        atomic_properties = set(shifts.keys()) | set(scales.keys())
        super().__init__(input_fields={"Za"} | atomic_properties, output_fields=atomic_properties)
        self.max_Za = max_Za
        self.shifts = self.build_affine(shifts, 0.0)
        self.scales = self.build_affine(scales, 1.0)

    def build_affine(
        self,
        params: Dict[str, Dict[Literal["values", "learnable"], Union[Dict[str, float], float, bool]]],
        default_value: float,
    ) -> nn.ParameterDict:
        affine_dict: Dict[str, nn.Parameter] = {}
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
            affine_dict[name] = nn.Parameter(affine_param, requires_grad=bool(param["learnable"]))
        return nn.ParameterDict(affine_dict)

    def get_output(self, Za: Tensor, **kwargs) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        for name in self._output_fields:
            if name not in kwargs:
                continue
            values = kwargs[name]
            if values is None:
                continue

            shift = self.shifts[name].gather(0, Za)
            scale = self.scales[name].gather(0, Za)

            if values.dim() == 1:
                shift = shift.view(-1)
                scale = scale.view(-1)
            else:
                shift = shift.view(-1, 1)
                scale = scale.view(-1, 1)

            output[name] = (values + shift) * scale
        return output

    def _load_from_state_dict(self, state_dict: Dict[str, Tensor], *args, **kwargs):
        for k, v in list(state_dict.items()):
            if k.startswith("shifts.") or k.startswith("scales."):
                # k format: "shifts.<prop_name>" or "scales.<prop_name>"
                prefix, prop_name = k.split(".", 1)
                if len(v) > self.max_Za + 1:
                    state_dict[k] = v[: self.max_Za + 1]
                elif len(v) < self.max_Za + 1:
                    ref = (self.shifts if prefix == "shifts" else self.scales)[prop_name]
                    state_dict[k] = torch.concat([v, ref[len(v) :]], dim=0)
        super()._load_from_state_dict(state_dict, *args, **kwargs)
        