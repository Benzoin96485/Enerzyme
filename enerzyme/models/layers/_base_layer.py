from typing import Dict, Set, List, Optional, Union
from inspect import signature
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module, Sequential
from torch_geometric.data import Data


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

        self.adapted_relevant_fields = False

    def _get_relevant_input_fields(self, net_input_fields: Optional[Set[str]]=None) -> Set[str]:
        new_input_fields = self.get_relevant_input_fields(net_input_fields)
        self._input_fields |= new_input_fields
        self._relevant_fields |= new_input_fields
        for field_name in new_input_fields:
            if field_name not in self._name_mapping:
                self._name_mapping[field_name] = field_name
        self.adapted_relevant_fields = True

    def get_relevant_input_fields(self, net_input_fields: Optional[Set[str]]=None) -> Set[str]:
        return self._input_fields

    @abstractmethod
    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ...

    def _forward(self, net_input: Union[Dict[str, Tensor], Data]) -> Dict[str, Tensor]:
        relevant_input = dict()
        if not self.adapted_relevant_fields:
            self._get_relevant_input_fields(net_input.keys())
        relevant_input_fields = self._input_fields
        if isinstance(net_input, Data):
            net_output = net_input
            for k in relevant_input_fields:
                if k == "batch_seg":
                    relevant_input[k] = net_input["batch"]
                else:
                    relevant_input[k] = net_input.get(self._name_mapping[k], None)
        else:
            net_output = net_input.copy()
            for k in relevant_input_fields:
                relevant_input[k] = net_input.get(self._name_mapping[k], None)
        relevant_output = self.get_output(**relevant_input)
        for k in self._output_fields:
            new_k = self._name_mapping[k]
            if k in relevant_output:
                net_output[new_k] = relevant_output[k]
        return net_output


class BaseFFLayer(BaseFFModule):
    def __init__(self, 
        input_fields: Optional[Set[str]]=None, 
        output_fields: Optional[Set[str]]=None,
        train_only: bool=False,
        eval_only: bool=False
    ) -> None:
        super().__init__(input_fields, output_fields)
        self.train_only = train_only
        self.eval_only = eval_only

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
        if self.train_only and not self.training:
            return net_input
        if self.eval_only and self.training:
            return net_input
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
