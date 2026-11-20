"""Fourier expansion of a bond angle.

Deng et al., Nat. Mach. Intell. 2023, eq. (1). The reference implementation
stores sine channels before cosine channels. The paper lists cosine first.
Both spans are the same Fourier space; this module follows the reference
order so a later weight port can line up channels.

``order`` is ``N`` in ``1 + 2N`` channels. Frequencies start at 1, 2, …, N.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module


class FourierExpansion(Module):
    """``[1/sqrt(2π), sin(nθ)/sqrt(π), cos(nθ)/sqrt(π)]`` for ``n = 1 … order``."""

    def __init__(self, order: int, learnable: bool = True) -> None:
        super().__init__()
        if int(order) < 0:
            raise ValueError(f"Fourier order must be >= 0, got {order}")
        self.order = int(order)
        frequencies = torch.arange(1, self.order + 1, dtype=torch.get_default_dtype())
        if self.order == 0:
            self.register_buffer("frequencies", frequencies)
        elif learnable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.learnable = bool(learnable) and self.order > 0

    def forward(self, angle: Tensor) -> Tensor:
        width = 1 + 2 * self.order
        if angle.ndim != 1:
            raise ValueError(f"angle must be (T,), got {tuple(angle.shape)}")
        if angle.numel() == 0:
            return angle.new_zeros(0, width)
        result = angle.new_empty(angle.shape[0], width)
        result[:, 0] = 1.0 / math.sqrt(2.0)
        if self.order == 0:
            return result / math.sqrt(math.pi)
        frequencies = self.frequencies.to(dtype=angle.dtype)
        phase = angle.unsqueeze(-1) * frequencies.unsqueeze(0)
        result[:, 1 : self.order + 1] = torch.sin(phase)
        result[:, self.order + 1 :] = torch.cos(phase)
        return result / math.sqrt(math.pi)
