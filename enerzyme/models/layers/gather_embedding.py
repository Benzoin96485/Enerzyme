from typing import Dict, Set
import torch
from torch import Tensor
from . import BaseFFLayer


class GatherAtomEmbedding(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={}, output_fields={"atom_embedding"})

    def get_relevant_input_fields(self, net_input_fields: Set[str]) -> Set[str]:
        relevant_input_fields = set()
        for field in net_input_fields:
            if field.endswith("_embedding"):
                relevant_input_fields.add(field)
        return relevant_input_fields
        
    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {"atom_embedding": torch.sum(torch.stack([v for v in relevant_input.values()], dim=0), dim=0)}
