"""Native paper eSCN Core (Passaro & Zitnick, 2023).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs). Default energy/forces use shared
``SimpleReadout`` + ``EnergyReduce`` + ``Force``. Opt-in ``SphereSampleReadout``
consumes ``atom_sphere_feature`` for paper-style S² integration of any atomic
property fields.

Distinct from ``enerzyme.models.esen`` (Meta UMA / eSCN-MD fairchem wrappers).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, SiLU

from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from ..so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    init_edge_rot_mat,
)
from .interaction import LayerBlock

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "polynomial",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"cutoff_fn": "polynomial", "trainable": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 64,
            "hidden_channels": 128,
            "edge_channels": 64,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class eSCNCore(BaseFFCore):
    """Equivariant Spherical Channel Network Core (SO(3)→SO(2) convolutions)."""

    def __str__(self) -> str:
        return """
#################################################################################
# eSCN Core (Passaro & Zitnick, NeurIPS 2023 / arXiv:2302.03655)               #
# Native SO(2)-reduced equivariant GNN; atom_feature = spherical l=0 channels  #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        sphere_channels: int = 64,
        hidden_channels: int = 128,
        edge_channels: int = 64,
        lmax: int = 2,
        mmax: int = 2,
        num_layers: int = 2,
        resolution: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
            },
            output_fields={"atom_feature", "atom_sphere_feature"},
        )
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")

        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.lmax_list = [lmax]
        self.mmax_list = [mmax]
        self.num_layers = num_layers
        self.num_resolutions = len(self.lmax_list)
        # Align with shared SimpleReadout contract: dim_feature_out = 0e width;
        # feature_irreps advertises that atom_feature is pure even scalars.
        self.dim_feature_out = self.num_resolutions * self.sphere_channels
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.act = SiLU()
        self.sphere_proj = torch.nn.Linear(dim_embedding, self.dim_feature_out)

        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m, resolution=resolution))
            self.SO3_grid.append(SO3_m_grid)

        self.layer_blocks = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layer_blocks.append(
                LayerBlock(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.edge_channels,
                    self.lmax_list,
                    self.mmax_list,
                    self.num_rbf,
                    self.max_Za,
                    self.SO3_grid,
                    self.act,
                )
            )

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
                if isinstance(layer, DistanceLayer):
                    # Prefer the injected DistanceLayer above; skip YAML Distance if any.
                    continue
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, BaseAtomEmbedding):
                    pass
                elif isinstance(layer, BaseRBF):
                    pass
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def encode_sphere(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        edge_rot_mat: Optional[Tensor] = None,
    ) -> Tensor:
        """Return full spherical node coeffs after message layers (pre-readout).

        Used by numerical parity tests and :meth:`get_output`. When ``edge_rot_mat``
        is provided it is used as-is (parity harness injects a shared frame).
        """
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.long()

        edge_index = torch.stack([idx_i_sr.long(), idx_j_sr.long()], dim=0)
        if edge_rot_mat is None:
            edge_rot_mat = init_edge_rot_mat(vij_sr.to(dtype=dtype))

        SO3_edge_rot = [
            SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            for i in range(self.num_resolutions)
        ]
        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, device)

        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            dtype,
        )

        sphere0 = self.sphere_proj(atom_embedding)
        offset = 0
        offset_res = 0
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = sphere0[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        for i in range(self.num_layers):
            x_message = self.layer_blocks[i](
                x,
                Za,
                rbf,
                edge_index,
                SO3_edge_rot,
                mappingReduced,
            )
            if i > 0:
                x.embedding = x.embedding + x_message.embedding
            else:
                x = x_message

        return x.embedding

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        atom_sphere_feature = self.encode_sphere(
            atom_embedding, Za, rbf, idx_i_sr, idx_j_sr, vij_sr
        )

        # Scalar atom features: concatenate l=0,m=0 channels across resolutions
        features = []
        offset_res = 0
        for i in range(self.num_resolutions):
            features.append(atom_sphere_feature[:, offset_res, :])
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        atom_feature = torch.cat(features, dim=-1)
        return {
            "atom_feature": atom_feature,
            "atom_sphere_feature": atom_sphere_feature,
        }
