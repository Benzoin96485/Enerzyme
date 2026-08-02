"""E2Former Core (Li et al., NeurIPS 2025 Spotlight).

Wigner-6j equivariant transformer Core: emits ``atom_feature`` (spherical
``l=0``) and ``atom_sphere_feature`` (full SH coeffs). Embeddings / RBF stay
in pre-core layers; energy / force heads stay in post-core layers.

Adapted from https://github.com/liyy2/E2Former (MIT license).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from ..so3 import build_so3_grid_table, get_normalization_layer
from .embedding import EdgeDegreeEmbeddingHigherOrder
from .graph import build_topk_neighborhood
from .interaction import TransBlock

_AVG_DEGREE = 15.57930850982666

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "polynomial",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "GaussianSmearing"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "irreps_node_embedding": "64x0e+64x1e+64x2e",
            "irreps_head": "16x0e+16x1e+16x2e",
            "num_layers": 2,
            "num_attn_heads": 4,
            "attn_scalar_head": 32,
            "ffn_hidden_channels": 128,
            "max_neighbors": 32,
            "attn_type": "first-order",
            "tp_type": "QK_alpha",
            "norm_layer": "rms_norm_sh",
            "alpha_drop": 0.0,
            "proj_drop": 0.0,
            "use_atom_edge_embedding": True,
            "use_gate_act": False,
            "use_grid_mlp": False,
            "use_sep_s2_act": True,
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


class E2FormerCore(BaseFFCore):
    """E2Former Wigner-6j graph-attention transformer Core."""

    def __str__(self) -> str:
        return """
#################################################################################
# E2Former Core (Li et al., NeurIPS 2025, arXiv:2501.19216)                     #
# Wigner-6j equivariant attention; atom_feature = l=0 channels                  #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        irreps_node_embedding: str = "64x0e+64x1e+64x2e",
        irreps_head: str = "16x0e+16x1e+16x2e",
        num_layers: int = 2,
        num_attn_heads: int = 4,
        attn_scalar_head: int = 32,
        ffn_hidden_channels: int = 128,
        max_neighbors: int = 32,
        attn_type: str = "first-order",
        tp_type: str = "QK_alpha",
        norm_layer: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        avg_degree: float = _AVG_DEGREE,
        use_atom_edge_embedding: bool = True,
        grid_resolution: Optional[int] = None,
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        ffn_activation: str = "scaled_silu",
        **kwargs,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "Ra",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "batch_seg",
            },
            output_fields={"atom_feature", "atom_sphere_feature"},
        )
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_head = o3.Irreps(irreps_head)
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("irreps_node_embedding must start with 0e")
        # Equal multiplicity across degrees (E2Former layout)
        self.scalar_dim = self.irreps_node_embedding[0][0]
        for mul, _ir in self.irreps_node_embedding:
            if mul != self.scalar_dim:
                raise ValueError(
                    "E2Former requires equal multiplicity across degrees; "
                    f"got {irreps_node_embedding}"
                )
        self.lmax = self.irreps_node_embedding[-1][1].l
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.ffn_hidden_channels = ffn_hidden_channels
        self.max_neighbors = max_neighbors
        self.attn_type = attn_type
        self.tp_type = tp_type
        self.norm_layer = norm_layer
        self.avg_degree = avg_degree

        self.dim_feature_out = self.scalar_dim
        self.feature_irreps = f"{self.scalar_dim}x0e"

        self.sphere_proj = torch.nn.Linear(dim_embedding, self.scalar_dim)
        self.edge_deg_embed = EdgeDegreeEmbeddingHigherOrder(
            self.irreps_node_embedding,
            avg_aggregate_num=avg_degree,
            number_of_basis=num_rbf,
            use_atom_edge=use_atom_edge_embedding,
            max_num_elements=max_Za + 1,
        )
        self.SO3_grid = build_so3_grid_table(
            self.lmax,
            normalization="component",
            resolution=grid_resolution,
            rescale_by_mmax=True,
        )
        self.blocks = ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransBlock(
                    irreps_node_input=self.irreps_node_embedding,
                    attn_weight_input_dim=num_rbf,
                    num_attn_heads=num_attn_heads,
                    attn_scalar_head=attn_scalar_head,
                    irreps_head=self.irreps_head,
                    SO3_grid=self.SO3_grid,
                    ffn_hidden_channels=ffn_hidden_channels,
                    alpha_drop=alpha_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    attn_type=attn_type,
                    tp_type=tp_type,
                    use_gate_act=use_gate_act,
                    use_grid_mlp=use_grid_mlp,
                    use_sep_s2_act=use_sep_s2_act,
                    ffn_activation=ffn_activation,
                    atom_type_cnt=max_Za + 1,
                )
            )
        self.norm = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
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
                    continue
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, (BaseAtomEmbedding, BaseRBF)):
                    pass
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def encode_sphere(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        Ra: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Tensor:
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.long()
        if batch_seg is None:
            batch_seg = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # Relative edges for seed / distances; COM-centered abs positions for
        # Wigner-6j solid harmonics (translation-stable, same relative geometry).
        neigh = build_topk_neighborhood(
            Ra.to(dtype=dtype),
            idx_i_sr,
            idx_j_sr,
            vij_sr.to(dtype=dtype),
            rbf.to(dtype=dtype),
            max_neighbors=self.max_neighbors,
            batch_seg=batch_seg,
        )
        ra_wigner = neigh["f_node_pos_wigner"]

        # Seed: scalar atom embed on l=0 + higher-order edge-degree features
        node_irreps = atom_embedding.new_zeros(
            num_atoms, (self.lmax + 1) ** 2, self.scalar_dim
        )
        node_irreps[:, 0, :] = self.sphere_proj(atom_embedding)
        node_irreps = node_irreps + self.edge_deg_embed(
            Za,
            neigh["edge_vec"],
            neigh["attn_mask"],
            neigh["attn_weight"],
            neigh["f_sparse_idx_node"],
        )

        attn_weight = neigh["attn_weight"]
        for blk in self.blocks:
            node_irreps, attn_weight = blk(
                node_pos=ra_wigner,
                node_irreps=node_irreps,
                edge_dis=neigh["edge_dis"],
                edge_vec=neigh["edge_vec"],
                attn_weight=attn_weight,
                atomic_numbers=Za,
                attn_mask=neigh["attn_mask"],
                batched_data=neigh,
            )
        return self.norm(node_irreps)

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        Ra: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        atom_sphere_feature = self.encode_sphere(
            atom_embedding,
            Za,
            Ra,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            batch_seg=batch_seg,
        )
        return {
            "atom_feature": atom_sphere_feature[:, 0, :],
            "atom_sphere_feature": atom_sphere_feature,
        }
