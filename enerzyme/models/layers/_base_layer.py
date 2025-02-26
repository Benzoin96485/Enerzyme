from typing import Dict, Set, List, Optional
from inspect import signature
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module, Sequential


def get_output_fields(func):
    key_type = signature(func).return_annotation.__args__[0]
    # if key_type is str:
    #     return set()
    # else:
    return set(key_type.__args__)


class BaseFFModule(ABC, Module):
    def __init__(self, input_fields: Optional[Set[str]]=None, output_fields: Optional[Set[str]]=None) -> None:
        super().__init__()
        
        # gather input fields
        if input_fields is None:
            input_fields = set()
            if output_fields is not None:
                for output_field in output_fields:
                    get_function = getattr(self, f"get_{output_field}", None)
                    if callable(get_function):
                        input_fields |= signature(get_function).parameters.keys() - {"self"}
            keys = signature(self.get_output).parameters.keys()
            if "relevant_input" in keys:
                if not input_fields:
                    raise KeyError("Input fields not specified!")
            else:
                input_fields |= keys - {"self"}
        self._input_fields = set(input_fields)
        
        # gather output fields
        if output_fields is None:
            key_type = signature(self.get_output).return_annotation.__args__[0]
            if key_type is str:
                raise KeyError("Output fields not specified!")
            else:
                output_fields = set(key_type.__args__)
        self._output_fields = set(output_fields)

        self._relevant_fields = self._input_fields | self._output_fields
        self._name_mapping = dict()
        for field_name in self._relevant_fields:
            self._name_mapping[field_name] = field_name

    @abstractmethod
    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ...

    def _forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
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


class BaseFFLayer(BaseFFModule):
    def __init__(self, input_fields: Optional[Set[str]]=None, output_fields: Optional[Set[str]]=None) -> None:
        super().__init__(input_fields, output_fields)

    def reset_field_name(self, **mapping: Dict[str, str]) -> None:
        self._relevant_fields = self._input_fields | self._output_fields
        for k, v in mapping.items():
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
        return self._forward(net_input)


class BaseFFCore(BaseFFModule):
    def __init__(self, input_fields: Optional[Set[str]]=None, output_fields: Optional[Set[str]]=None):
        super().__init__(input_fields, output_fields)
        self.pre_sequence = Sequential()
        self.post_sequence = Sequential()

    def forward(self, net_input):
        return self.post_sequence(self._forward(self.pre_sequence(net_input)))
    
    @abstractmethod
    def build(self, built_layers: List[Module]) -> None:
        ...
