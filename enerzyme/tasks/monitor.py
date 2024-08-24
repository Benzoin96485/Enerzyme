from typing import Dict, List
import numpy as np
import torch
from torch import Tensor
from ..utils import logger


class Monitor:
    def __init__(self, **terms: Dict[str, List[str]]) -> None:
        self.terms = terms
        self._reset()
    
    def _reset(self) -> None:
        self.collection = {k: [] for k in self.terms}

    def collect(self, output: Dict[str, Tensor]) -> None:
        with torch.no_grad():
            for k in self.terms:
                if k in output:
                    self.collection[k].extend(output[k].detach().cpu().numpy())
                elif k + "_a" in output:
                    from ..models.functional import segment_sum
                    self.collection[k].extend(segment_sum(output[k + "_a"].detach(), output["batch_seg"]).cpu().numpy())

    def summary(self) -> None:
        message = []
        for term, stats in self.terms.items():
            for stat in stats:
                if stat == "mean":
                    message.append(f"{stat} of {term}: {np.mean(self.collection[term])}")
                if stat == "std":
                    message.append(f"{stat} of {term}: {np.std(self.collection[term])}")
                if stat == "max":
                    message.append(f"{stat} of {term}: {np.max(self.collection[term])}")
                if stat == "min":
                    message.append(f"{stat} of {term}: {np.min(self.collection[term])}")

        logger.info(", ".join(message))
        self._reset()