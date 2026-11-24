"""Per-atom energy from CAMP scalar features.

Early layers use a linear map. The last layer uses a SiLU MLP (Wen et al.,
2025, Eq. 10). Summation over atoms and the force derivative stay in
``EnergyReduce`` and ``Force``. A per-element reference belongs in
``AtomicAffine``: an isolated atom has a zero CAMP feature, so its energy is
only the readout bias until that reference is added.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ..layers._base_layer import BaseFFLayer
from .radial import resolve_hidden_layers, silu_mlp


class CAMPEnergyReadout(BaseFFLayer):
    """Sum of per-layer heads on ``atom_feature`` of shape ``[N, C, T]``."""

    def __init__(
        self,
        built_layers: List[nn.Module],
        hidden: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        super().__init__(input_fields={"atom_feature"}, output_fields={"Ea"})
        core = _find_camp_core(built_layers)
        self.n_channels = int(core.dim_embedding)
        self.num_layers = int(core.num_layers)
        spec = core.output_mlp_hidden_layers if hidden is None else hidden
        widths = resolve_hidden_layers(spec, self.n_channels)
        self.heads = nn.ModuleList(
            [nn.Linear(self.n_channels, 1) for _ in range(self.num_layers - 1)]
        )
        self.heads.append(silu_mlp(self.n_channels, widths, 1))

    def get_Ea(self, atom_feature: Tensor) -> Tensor:
        expected = (atom_feature.shape[0], self.n_channels, self.num_layers)
        if atom_feature.ndim != 3 or tuple(atom_feature.shape) != expected:
            raise ValueError(
                f"CAMP atom_feature must be [N, {self.n_channels}, {self.num_layers}], "
                f"got {tuple(atom_feature.shape)}"
            )
        energy = atom_feature.new_zeros(atom_feature.shape[0])
        for index, head in enumerate(self.heads):
            energy = energy + head(atom_feature[:, :, index]).reshape(-1)
        return energy


def _find_camp_core(built_layers: List[nn.Module]) -> nn.Module:
    for layer in reversed(built_layers):
        if getattr(layer, "is_camp_core", False):
            return layer
    raise TypeError("CAMPEnergyReadout must follow a CAMPCore in the layer list")
