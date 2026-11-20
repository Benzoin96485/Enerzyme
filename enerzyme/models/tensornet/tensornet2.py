"""TensorNet2 Core (Farr et al., arXiv:2601.00581).

The Cartesian embedding and tensor update are TensorNet. After the embedding
and after every interaction, a charge head reads ``(I, ||A||², ||S||²)``,
equilibrates each channel to the molecular total charge, and the next
interaction concatenates those channels onto the radial edge features.
The scalar norm embedding is unchanged. Concatenated charge channels are a
Core output so :class:`~enerzyme.models.layers.electrostatics.WeightedCoulombEnergyLayer`
can add the long-range Coulomb term. Species embedding, the radial basis, the
energy MLP, Coulomb, and ZBL stay outside the message passing.

The pure PyTorch rules follow ``torchmdnet.models.tensornet2`` (MIT). Warp
kernels are not used, and TorchMD-Net checkpoints do not load. AceFF-2 widths
are 2 interactions, 128 channels, 32 radial functions, charge width 16, and
the repository cutoff 4.5 Å. AceFF-2-OMOL adds
:class:`~enerzyme.models.layers.screened_zbl.StandardZBLRepulsionEnergyLayer`;
that term is not part of this Core.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers.electrostatics import WeightedCoulombEnergyLayer, expand_coulomb_weights
from .charge import TensorChargeHead
from .core import TensorNetCore
from .interaction import TensorNet2Interaction
from .ops import scalar_invariants, skew_tensor_to_vector


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

DEFAULT_BUILD_PARAMS_2 = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 4.5,
    "cutoff_fn": "cosine",
    "q_dim": 16,
}

# Stage weights 1, 2, 3 give the last charge block half of the Coulomb average
# (1+2+3). AceFF states that the last layer dominates; it does not publish the
# numeric weights, so this schedule is the Enerzyme default and can be replaced.
DEFAULT_LAYER_PARAMS_2 = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 2,
            "equivariance_group": "O(3)",
            "include_self": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {
        "name": "WeightedCoulombEnergy",
        "params": {
            "layer_weights": [1.0, 2.0, 3.0],
            "damping_cutoff": 4.6,
            "pairs": "all_to_all",
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class TensorNet2Core(TensorNetCore):
    """TensorNet stack with neutral charge equilibration inside the messages."""

    def __str__(self) -> str:
        return """
################################################################################
# TensorNet2 (Farr et al., arXiv:2601.00581)                                   #
# TensorNet messages conditioned on equilibrated charge channels               #
# atom_feature [N, F], charge_channels [N, (L+1) q]                           #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 32,
        cutoff_sr: float = 4.5,
        num_interactions: int = 2,
        equivariance_group: str = "O(3)",
        include_self: bool = True,
        q_dim: int = 16,
    ) -> None:
        if int(q_dim) < 1:
            raise ValueError(f"q_dim must be >= 1, got {q_dim}")
        super().__init__(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            cutoff_sr=cutoff_sr,
            num_interactions=num_interactions,
            equivariance_group=equivariance_group,
            charge_scale=0.0,
            include_self=include_self,
        )
        self.q_dim = int(q_dim)
        self.num_charge_blocks = self.num_interactions + 1
        self.charge_channels_dim = self.num_charge_blocks * self.q_dim
        self.interactions = ModuleList(
            [
                TensorNet2Interaction(
                    self.dim_embedding,
                    self.num_rbf,
                    self.q_dim,
                    self.equivariance_group,
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.charge_heads = ModuleList(
            [
                TensorChargeHead(self.dim_embedding, self.q_dim)
                for _ in range(self.num_charge_blocks)
            ]
        )
        self._output_fields = set(self._output_fields) | {"charge_channels"}
        self._relevant_fields = self._input_fields | self._output_fields
        self._name_mapping["charge_channels"] = "charge_channels"

    def build(self, built_layers: List[Module]) -> None:
        super().build(built_layers)
        after_core = False
        for layer in built_layers:
            if layer is self:
                after_core = True
                continue
            if not after_core or not isinstance(layer, WeightedCoulombEnergyLayer):
                continue
            try:
                expand_coulomb_weights(
                    layer.layer_weights, self.charge_channels_dim, torch.zeros(1)
                )
            except ValueError as exc:
                raise ValueError(
                    f"{exc} TensorNet2 emits {self.charge_channels_dim} charge "
                    f"channels ({self.num_charge_blocks} blocks x q_dim {self.q_dim}). "
                    "Set WeightedCoulombEnergy layer_weights to one weight per "
                    "block, one weight per channel, or omit them."
                ) from exc

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Embed, equilibrate, then update. ``Q`` is the molecular total charge.

        A missing ``Q`` equilibrates every channel to zero. The charge block
        from stage ``k`` is concatenated into ``charge_channels`` and, except
        for the last block, is the edge feature of the next interaction.
        """
        n_atoms, idx_i, idx_j, hat, basis, window = self._prepare_graph(
            atom_embedding,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            Dij_sr,
            cutoff_values_sr,
        )
        features = self.embedding(atom_embedding, idx_i, idx_j, hat, basis, window)
        total_charge = self.resolve_total_charge(n_atoms, Q, batch_seg, atom_embedding)
        blocks = []
        charges = self.charge_heads[0](features, total_charge, batch_seg)
        blocks.append(charges)
        for interaction, head in zip(self.interactions, list(self.charge_heads)[1:]):
            features = interaction(features, charges, idx_i, idx_j, basis, window)
            charges = head(features, total_charge, batch_seg)
            blocks.append(charges)

        invariants, _iso, skew, _traceless = scalar_invariants(features)
        atom_feature = F.silu(self.out_linear(self.out_norm(invariants)))
        return {
            "atom_feature": atom_feature,
            "atom_vector_feature": skew_tensor_to_vector(skew),
            "atom_tensor_feature": features,
            "charge_channels": torch.cat(blocks, dim=-1),
        }
