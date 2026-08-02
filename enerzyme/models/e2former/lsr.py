"""E2Former-LSR Core (Wang et al., arXiv:2601.03774).

Short-range E2Former + fragment bipartite long-range attention + late fusion.
Adapted from UBio-MolFM tag E2Former-LSR (MIT):
https://github.com/IQuestLab/UBio-MolFM
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import ModuleList

from ..so3 import get_normalization_layer
from ..so3.linear import SO3Linear
from .cluster import (
    build_atom_fragment_topk,
    pool_fragment_irreps,
    resolve_fragments,
)
from .core import (
    DEFAULT_BUILD_PARAMS as _E2_BUILD,
    DEFAULT_LAYER_PARAMS as _E2_LAYERS,
    E2FormerCore,
)
from .interaction import ClusterTransBlock

DEFAULT_BUILD_PARAMS = {
    **_E2_BUILD,
    "cutoff_lr": 15.0,
}

DEFAULT_LAYER_PARAMS = deepcopy(_E2_LAYERS)
# Paper-ish short depth; long layers / fragment knobs live on Core params.
for _layer in DEFAULT_LAYER_PARAMS:
    if _layer.get("name") == "Core":
        _layer["params"] = {
            **_layer["params"],
            "num_layers": 4,
            "long_layers": 2,
            "long_max_neighbors": 64,
            "fragment_mode": "kmeans",
            "min_nodes_per_group": 24,
            "cutoff_lr": 15.0,
        }
        break


class _LongRangeGaussianRBF(torch.nn.Module):
    """Gaussian RBF on atom–fragment distances (inside Core; not the SR layer)."""

    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        offset = torch.linspace(0.0, float(cutoff), num_rbf)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        # dist: [N, K] → [N, K, num_rbf]
        d = dist.unsqueeze(-1) - self.offset.view(1, 1, -1)
        return torch.exp(self.coeff * torch.pow(d, 2))


class E2FormerLSRCore(E2FormerCore):
    """E2Former with long-range aware fragment message passing."""

    def __str__(self) -> str:
        return """
#################################################################################
# E2Former-LSR Core (Wang et al., arXiv:2601.03774)                             #
# Short-range E2Former + atom–fragment long-range attention + late fuse         #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        irreps_node_embedding: str = "64x0e+64x1e+64x2e",
        irreps_head: str = "16x0e+16x1e+16x2e",
        num_layers: int = 4,
        long_layers: int = 2,
        num_attn_heads: int = 4,
        attn_scalar_head: int = 32,
        ffn_hidden_channels: int = 128,
        max_neighbors: int = 32,
        long_max_neighbors: int = 64,
        attn_type: str = "first-order",
        tp_type: str = "QK_alpha",
        norm_layer: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        avg_degree: float = 15.57930850982666,
        use_atom_edge_embedding: bool = True,
        grid_resolution: Optional[int] = None,
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        ffn_activation: str = "scaled_silu",
        cutoff_lr: float = 15.0,
        fragment_mode: str = "kmeans",
        min_nodes_per_group: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            max_Za=max_Za,
            irreps_node_embedding=irreps_node_embedding,
            irreps_head=irreps_head,
            num_layers=num_layers,
            num_attn_heads=num_attn_heads,
            attn_scalar_head=attn_scalar_head,
            ffn_hidden_channels=ffn_hidden_channels,
            max_neighbors=max_neighbors,
            attn_type=attn_type,
            tp_type=tp_type,
            norm_layer=norm_layer,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            avg_degree=avg_degree,
            use_atom_edge_embedding=use_atom_edge_embedding,
            grid_resolution=grid_resolution,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            ffn_activation=ffn_activation,
            **kwargs,
        )
        # Optional BRICS / precomputed fragment fields (None when absent).
        self._input_fields |= {"cluster_ids", "cluster_centers"}
        self._relevant_fields |= {"cluster_ids", "cluster_centers"}
        for name in ("cluster_ids", "cluster_centers"):
            self._name_mapping[name] = name

        self.long_layers = int(long_layers)
        self.long_max_neighbors = int(long_max_neighbors)
        self.cutoff_lr = float(cutoff_lr)
        self.fragment_mode = fragment_mode
        self.min_nodes_per_group = int(min_nodes_per_group)

        self.rbf_long = _LongRangeGaussianRBF(num_rbf, self.cutoff_lr)
        self.cluster_blocks = ModuleList()
        for _ in range(self.long_layers):
            self.cluster_blocks.append(
                ClusterTransBlock(
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
        self.norm_fuse_short = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.norm_fuse_long = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.final_linear = SO3Linear(
            2 * self.scalar_dim, self.scalar_dim, lmax=self.lmax
        )

    def encode_sphere_lsr(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        Ra: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
        cluster_ids: Optional[Tensor] = None,
        cluster_centers: Optional[Tensor] = None,
    ) -> Tensor:
        short = self.encode_sphere_short(
            atom_embedding,
            Za,
            Ra,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            batch_seg=batch_seg,
            apply_final_norm=True,
        )
        node_irreps = short["node_irreps"]
        ra_wigner = short["ra_wigner"]
        batch_seg = short["batch_seg"]
        Za = short["Za"]
        node_irreps_short = node_irreps

        flat_ids, cluster_pos, cluster_batch, _local = resolve_fragments(
            ra_wigner,
            batch_seg,
            fragment_mode=self.fragment_mode,
            cluster_ids=cluster_ids,
            cluster_centers=cluster_centers,
            min_nodes_per_group=self.min_nodes_per_group,
        )
        # Centers are means of COM-centered atoms (same frame as ra_wigner).

        long_graph = build_atom_fragment_topk(
            atom_pos=ra_wigner,
            cluster_pos=cluster_pos,
            flat_cluster_ids=flat_ids,
            batch_seg=batch_seg,
            cluster_batch=cluster_batch,
            radius=self.cutoff_lr,
            max_neighbors=self.long_max_neighbors,
            remove_self_cluster=True,
        )
        edge_dis = long_graph["edge_dis"]
        edge_vec = long_graph["edge_vec"]
        attn_mask = long_graph["attn_mask"]
        attn_weight = self.rbf_long(edge_dis)
        attn_weight = attn_weight.masked_fill(attn_mask, 0.0)

        # Empty neighbor lists still run long blocks (atom FFN) + late fuse.
        # Only skip when there is no long stack or no fragments at all.
        if self.long_layers == 0 or cluster_pos.shape[0] == 0:
            return node_irreps_short

        for blk in self.cluster_blocks:
            cluster_irreps = pool_fragment_irreps(node_irreps, flat_ids)
            node_irreps, attn_weight = blk(
                node_pos=ra_wigner,
                node_irreps=node_irreps,
                edge_dis=edge_dis,
                edge_vec=edge_vec,
                attn_weight=attn_weight,
                atomic_numbers=Za,
                attn_mask=attn_mask,
                batched_data=long_graph,
                cluster_pos=cluster_pos,
                cluster_irreps=cluster_irreps,
            )

        fused = self.final_linear(
            torch.cat(
                [
                    self.norm_fuse_short(node_irreps_short),
                    self.norm_fuse_long(node_irreps),
                ],
                dim=-1,
            )
        )
        return fused

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
        cluster_ids: Optional[Tensor] = None,
        cluster_centers: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encode_sphere_lsr(
            atom_embedding,
            Za,
            Ra,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            batch_seg=batch_seg,
            cluster_ids=cluster_ids,
            cluster_centers=cluster_centers,
        )

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
        cluster_ids: Optional[Tensor] = None,
        cluster_centers: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        atom_sphere_feature = self.encode_sphere_lsr(
            atom_embedding,
            Za,
            Ra,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            batch_seg=batch_seg,
            cluster_ids=cluster_ids,
            cluster_centers=cluster_centers,
        )
        return {
            "atom_feature": atom_sphere_feature[:, 0, :],
            "atom_sphere_feature": atom_sphere_feature,
        }
