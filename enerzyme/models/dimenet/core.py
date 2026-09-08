"""DimeNet Core — directional message passing (Gasteiger et al., ICLR 2020).

Original DimeNet (not DimeNet++). Atom-type embeddings and Bessel RBF are
pre-core layers; per-block atom features are emitted for shared
:class:`~enerzyme.models.layers.readout.HierachicalReadout` /
:class:`~enerzyme.models.layers.readout.SimpleReadout`.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from .basis import SphericalFourierBesselBasis
from .interaction import (
    DimeNetAtomReadout,
    DimeNetEmbeddingBlock,
    DimeNetInteractionBlock,
)
from .triplets import directed_triplets, triplet_angles

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 6,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    "cutoff_fn": "polynomial",
}

_SWISH = {
    "dim_feature": 1,
    "initial_alpha": 1,
    "initial_beta": 1,
    "learnable": False,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "dimenet",
            "trainable": True,
            "envelope_exponent": 5,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_blocks": 6,
            "num_bilinear": 8,
            "num_spherical": 7,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "envelope_exponent": 5,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
        },
    },
    {
        "name": "HierachicalReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "num_hidden_layers": 3,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
            "initial_weight": "semi_orthogonal_glorot",
            "use_bias_out": False,
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class DimeNetCore(BaseFFCore):
    """Directional message-passing Core (original DimeNet bilinear SBF).

    Emits ``atom_feature`` with shape ``(N, F, num_blocks+1)`` in the default
    ``output_mode="feature"`` (one RBF-gated atom readout per block, including
    the embedding block). ``output_mode="last"`` keeps only the final slice
    ``(N, F)`` for :code:`SimpleReadout`.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        num_blocks: int = 6,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        envelope_exponent: int = 5,
        activation_fn: ACTIVATION_KEY_TYPE = "swish",
        activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
        output_mode: Literal["feature", "last"] = "feature",
    ) -> None:
        if num_spherical < 1:
            raise ValueError("num_spherical must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.output_mode = output_mode
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
            },
            output_fields={"atom_feature"},
        )
        if activation_params is None:
            activation_params = dict(_SWISH)
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr
        self.num_blocks = num_blocks
        self.num_bilinear = num_bilinear
        self.num_spherical = num_spherical
        self.num_output_blocks = num_blocks + 1
        self.dim_feature_out = dim_embedding
        self.feature_irreps = f"{dim_embedding}x0e"

        self.sbf = SphericalFourierBesselBasis(
            num_spherical=num_spherical,
            num_radial=num_rbf,
            cutoff=cutoff_sr,
            envelope_exponent=envelope_exponent,
        )
        self.emb_block = DimeNetEmbeddingBlock(
            dim_embedding, num_rbf, activation_fn, activation_params
        )
        self.interactions = ModuleList(
            [
                DimeNetInteractionBlock(
                    dim_embedding=dim_embedding,
                    num_bilinear=num_bilinear,
                    num_sbf=self.sbf.out_dim,
                    num_rbf=num_rbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    activation_fn=activation_fn,
                    activation_params=activation_params,
                )
                for _ in range(num_blocks)
            ]
        )
        self.atom_readouts = ModuleList(
            [
                DimeNetAtomReadout(num_rbf, dim_embedding)
                for _ in range(self.num_output_blocks)
            ]
        )

    def __str__(self) -> str:
        return """
#########################################################################
# DimeNet (ICLR 2020, arXiv:2003.03123) — directional message passing  #
# Original bilinear SBF interaction (not DimeNet++).                   #
#########################################################################
"""

    def build(self, built_layers: List[Module]) -> None:
        self.calculate_distance = DistanceLayer()
        self.calculate_distance.with_vector_on("vij_lr")
        self.calculate_distance.reset_field_name(Dij="Dij_lr")
        self.pre_sequence.append(self.calculate_distance)

        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, BaseAtomEmbedding):
                    self.atom_embedding = layer
                elif isinstance(layer, BaseRBF):
                    self.radial_basis_function = layer
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        num_nodes = atom_embedding.shape[0]
        idx_kj, idx_ji, _, _, _ = directed_triplets(
            idx_i_sr, idx_j_sr, num_nodes
        )
        angles = triplet_angles(vij_sr, idx_kj, idx_ji)
        sbf = self.sbf(Dij_sr, angles, idx_kj)

        messages = self.emb_block(atom_embedding, rbf, idx_i_sr, idx_j_sr)
        features = [
            self.atom_readouts[0](messages, rbf, idx_i_sr, num_nodes)
        ]
        for block, readout in zip(self.interactions, self.atom_readouts[1:]):
            messages = block(messages, rbf, sbf, idx_kj, idx_ji)
            features.append(readout(messages, rbf, idx_i_sr, num_nodes))

        atom_feature = torch.stack(features, dim=-1)
        if self.output_mode == "last":
            return {"atom_feature": atom_feature[:, :, -1]}
        return {"atom_feature": atom_feature}
