"""Per-layer MACE energy head.

Each interaction product is mapped by its own gated MLP and the site energies
are summed. The head reads the full irreps tensor. A scalar hidden irrep only
contracts the ``0e`` paths, which is the MACE nonlinear readout. Set
``shallow_ensemble_size`` above 1 to widen the last linear; the polarizable
stack stays at 1 because the Coulomb term is a single energy.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from e3nn import o3
from torch import Tensor

from ..layers._base_layer import BaseFFLayer
from .blocks import GatedIrrepsMLP, feature_spec, stack_node_features


class MACEEnergyReadout(BaseFFLayer):
    def __init__(
        self,
        built_layers=None,
        feature_irreps: Optional[str] = None,
        num_output_blocks: Optional[int] = None,
        MLP_irreps: str = "16x0e",
        shallow_ensemble_size: int = 1,
    ) -> None:
        super().__init__(output_fields={"Ea"})
        self.feature_irreps, self.num_output_blocks = feature_spec(
            built_layers, feature_irreps, num_output_blocks
        )
        ensemble = int(shallow_ensemble_size)
        if ensemble < 1:
            raise ValueError(f"shallow_ensemble_size must be >= 1, got {ensemble}")
        self.shallow_ensemble_size = ensemble
        target = o3.Irreps(f"{ensemble}x0e")
        self.heads = torch.nn.ModuleList(
            [
                GatedIrrepsMLP(self.feature_irreps, MLP_irreps, target)
                for _ in range(self.num_output_blocks)
            ]
        )

    def get_output(self, atom_feature: Optional[Tensor] = None) -> Dict[str, Tensor]:
        layers = stack_node_features(atom_feature, self.num_output_blocks)
        energy = layers.new_zeros(layers.shape[0], self.shallow_ensemble_size)
        for index, head in enumerate(self.heads):
            energy = energy + head(layers[:, :, index])
        if self.shallow_ensemble_size == 1:
            energy = energy.squeeze(-1)
        return {"Ea": energy}
