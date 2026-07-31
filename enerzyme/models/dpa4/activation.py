"""DPA4 activations — thin re-exports of shared ``so3`` primitives."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..so3.gated import FocusLinear, SO3GatedActivation
from ..so3.linear import SO3FocusLinear

# Historical DPA4 names
SO3Linear = SO3FocusLinear
GatedActivation = SO3GatedActivation


class SwiGLU(nn.Module):
    """SwiGLU over the last dim: ``value * silu(gate)`` (DPA4 / SeZM order).

    Distinct from :class:`~enerzyme.models.so3.activation_v3.SwiGLU`, which
    applies ``silu`` to the first half (EquiformerV3 convention).
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even last dimension")
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


__all__ = [
    "FocusLinear",
    "GatedActivation",
    "SO3GatedActivation",
    "SO3FocusLinear",
    "SO3Linear",
    "SwiGLU",
]
