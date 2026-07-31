"""DPA4 norms — re-export shared degree-balanced RMSNorm."""

from __future__ import annotations

import torch.nn as nn

from ..so3.layer_norm import EquivariantDegreeRMSNorm

EquivariantRMSNorm = EquivariantDegreeRMSNorm


class IdentityNorm(nn.Module):
    def forward(self, x):
        return x


__all__ = ["EquivariantDegreeRMSNorm", "EquivariantRMSNorm", "IdentityNorm"]
