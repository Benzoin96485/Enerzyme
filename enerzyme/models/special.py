import numpy as np
import math
import torch
from torch import Tensor
from typing import Tuple


def get_berstein_coefficient(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logfactorial = np.zeros(order)
    for i in range(2, order):
        logfactorial[i] = logfactorial[i - 1] + np.log(i)
    v = np.arange(0, order)
    n = (order - 1) - v
    logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
    return v, n, logbinomial


def sinc(x: Tensor) -> Tensor:
    x = x * math.pi
    return torch.sin(x) / x