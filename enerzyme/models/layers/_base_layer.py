from typing import Dict, Set
from torch import Tensor
from torch.nn import Module


class BaseLayer(Module):
    def __init__(self, input_fields: Set[str], output_fields: Set[str]) -> None:
        super().__init__()
        self._input_fields = input_fields
        self._output_fields = output_fields
        self._relevant_fields = input_fields | output_fields
        self._name_mapping = dict()
        for field_name in self._relevant_fields:
            self._name_mapping[field_name] = field_name

    def reset_field_name(self, **mapping: Dict[str, str]) -> None:
        self._relevant_fields = self.input_fields | self.output_fields
        for k, v in mapping:
            if k in self._relevant_fields:
                self._name_mapping[k] = v
            else:
                raise KeyError(f"{k} is not a relevant field name!")

    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        relevant_output = dict()
        for k in self._output_fields:
            relevant_output[k] = getattr(self, f"get_{k}")(**relevant_input)
        return relevant_output

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        net_output = net_input.copy()
        relevant_input = dict()
        for k in self._input_fields:
            new_k = self._name_mapping[k]
            relevant_input[k] = net_input.get(new_k, None)
        relevant_output = self.get_output(**relevant_input)
        for k in self._output_fields:
            new_k = self._name_mapping[k]
            if k in relevant_output:
                net_output[new_k] = relevant_output[k]
        return net_output
