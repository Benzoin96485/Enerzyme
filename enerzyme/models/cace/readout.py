"""Per-atom energy from CACE invariants.

The published head is a linear map plus a SiLU MLP. The linear map keeps the
body-ordered polynomial; the MLP adds the higher-order remainder of the
truncated cluster expansion (Cheng, 2024, Eq. 15). Summation over atoms and
the force derivative stay in ``EnergyReduce`` and ``Force``. A per-element
reference belongs in ``AtomicAffine`` or in the dataset atomic-energy
transform: an isolated atom has a zero CACE descriptor.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

from ..layers._base_layer import BaseFFLayer


class CACEEnergyReadout(BaseFFLayer):
    """Sum of a linear head and a SiLU MLP on ``atom_feature``."""

    def __init__(
        self,
        built_layers: List[nn.Module],
        hidden: Optional[Sequence[int]] = None,
        add_linear: bool = True,
    ) -> None:
        super().__init__(input_fields={"atom_feature"}, output_fields={"Ea"})
        core = _find_cace_core(built_layers)
        self.dim_feature_in = int(core.dim_feature_out)
        widths = tuple(core.readout_hidden if hidden is None else (int(h) for h in hidden))
        if any(width < 1 for width in widths):
            raise ValueError(f"CACE readout widths must be >= 1, got {widths}")
        self.add_linear = bool(add_linear)
        self.linear: Optional[nn.Linear] = None
        if self.add_linear:
            self.linear = nn.Linear(self.dim_feature_in, 1)
        if widths:
            layers = []
            previous = self.dim_feature_in
            for width in widths:
                layers.extend([nn.Linear(previous, width), nn.SiLU()])
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.mlp: Optional[nn.Module] = nn.Sequential(*layers)
        else:
            self.mlp = None
        if self.linear is None and self.mlp is None:
            raise ValueError("CACE readout needs a linear map or an MLP")

    def get_Ea(self, atom_feature: Tensor) -> Tensor:
        if atom_feature.ndim != 2 or atom_feature.shape[-1] != self.dim_feature_in:
            raise ValueError(
                f"CACE atom_feature must be [N, {self.dim_feature_in}], "
                f"got {tuple(atom_feature.shape)}"
            )
        energy = atom_feature.new_zeros(atom_feature.shape[0])
        if self.linear is not None:
            energy = energy + self.linear(atom_feature).squeeze(-1)
        if self.mlp is not None:
            energy = energy + self.mlp(atom_feature).squeeze(-1)
        return energy


def _find_cace_core(built_layers: List[nn.Module]) -> nn.Module:
    for layer in reversed(built_layers):
        if getattr(layer, "is_cace_core", False):
            return layer
    raise TypeError("CACEEnergyReadout must follow a CACECore in the layer list")
