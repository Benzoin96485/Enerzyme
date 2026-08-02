# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
"""Edge-degree seed embedding for equal-multiplicity SH channels."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..blocks.radial_mlp import RadialProfile


class EdgeDegreeEmbeddingHigherOrder(nn.Module):
    """Seed node SH features from radial × spherical-harmonic edges.

    Simplified port of ``EdgeDegreeEmbeddingNetwork_higherorder`` (no electron-
    density branch). Operates on padded top-K neighborhoods.
    """

    def __init__(
        self,
        irreps_node_embedding: str | o3.Irreps,
        avg_aggregate_num: float = 15.0,
        number_of_basis: int = 32,
        use_layer_norm: bool = True,
        use_atom_edge: bool = True,
        max_num_elements: int = 95,
    ) -> None:
        super().__init__()
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must include a 0e irrep first")
        self.number_of_basis = number_of_basis
        self.avg_aggregate_num = float(avg_aggregate_num)
        self.use_atom_edge = use_atom_edge
        self.lmax = self.irreps_node_embedding[-1][1].l
        self.scalar_dim = self.irreps_node_embedding[0][0]

        self.source_embedding: Optional[nn.Embedding]
        self.target_embedding: Optional[nn.Embedding]
        if use_atom_edge:
            self.source_embedding = nn.Embedding(max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(max_num_elements, number_of_basis)
            nn.init.uniform_(self.source_embedding.weight, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight, -0.001, 0.001)
        else:
            self.source_embedding = None
            self.target_embedding = None

        in_dim = number_of_basis * 3 if use_atom_edge else number_of_basis
        self.gbf_projs = nn.ModuleList()
        for mul, _ir in self.irreps_node_embedding:
            self.gbf_projs.append(
                RadialProfile(
                    [
                        in_dim,
                        min(number_of_basis, 128),
                        min(number_of_basis, 128),
                        mul,
                    ],
                    use_layer_norm=use_layer_norm,
                    use_offset=False,
                )
            )

    def forward(
        self,
        atomic_numbers: Tensor,
        edge_vec: Tensor,
        attn_mask: Tensor,
        edge_scalars: Tensor,
        f_sparse_idx_node: Tensor,
    ) -> Tensor:
        """Return seeded node features ``[N, (lmax+1)^2, C]``."""
        top_k = edge_vec.shape[1]
        if self.use_atom_edge:
            assert self.source_embedding is not None and self.target_embedding is not None
            tgt = self.target_embedding(atomic_numbers).unsqueeze(1).expand(-1, top_k, -1)
            src = self.source_embedding(atomic_numbers)[f_sparse_idx_node]
            edge_dis_embed = torch.cat([edge_scalars, tgt, src], dim=-1)
        else:
            edge_dis_embed = edge_scalars

        node_features = []
        for idx, (_mul, ir) in enumerate(self.irreps_node_embedding):
            lx = o3.spherical_harmonics(
                l=ir.l,
                x=edge_vec,
                normalize=True,
                normalization="norm",
            )
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            edge_fea = torch.where(attn_mask, 0, edge_fea)
            lx_embed = torch.einsum("mnd,mnh->mdh", lx, edge_fea)
            node_features.append(lx_embed)
        return torch.cat(node_features, dim=1) / self.avg_aggregate_num
