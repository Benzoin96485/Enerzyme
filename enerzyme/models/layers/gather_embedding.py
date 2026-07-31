from typing import Dict, Set
import torch
from torch import Tensor
from . import BaseFFLayer


class GatherAtomEmbedding(BaseFFLayer):
    def __init__(self, scale_by_sqrt_count: bool = False) -> None:
        """Sum all ``*_embedding`` fields into ``atom_embedding``.

        Args:
            scale_by_sqrt_count: If True, divide by ``sqrt(n_embeddings)``
                (SO3LR / So3krates-torch convention when charge+spin embeds
                are enabled). Default False preserves historical behaviour.
        """
        super().__init__(input_fields={}, output_fields={"atom_embedding"})
        self.scale_by_sqrt_count = scale_by_sqrt_count

    def get_relevant_input_fields(self, net_input_fields: Set[str]) -> Set[str]:
        relevant_input_fields = set()
        for field in net_input_fields:
            if field.endswith("_embedding"):
                relevant_input_fields.add(field)
        return relevant_input_fields
        
    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        stacked = torch.stack([v for v in relevant_input.values()], dim=0)
        emb = torch.sum(stacked, dim=0)
        if self.scale_by_sqrt_count:
            emb = emb / (stacked.shape[0] ** 0.5)
        return {"atom_embedding": emb}
