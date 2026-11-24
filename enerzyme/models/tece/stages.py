"""TECE Core — Edge Cluster Expansion with Radial Rotary Attention.

Paper: Xu et al., arXiv:2607.10664
Upstream: https://github.com/xvzemin/tace (MIT)

Feature-mode Core only: emits ``atom_feature`` for external ``SimpleReadout``.
Embeddings / energy-force heads / physics priors stay outside the Core.
"""

from __future__ import annotations

from typing import Optional

from ..mp.block import BlockState, Block
from ..mp.core import FieldEmbed

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mp.nn.radial_mlp import RadialMLP
from ..mp.equivariant.e3nn import IrrepsLinear, O3ScatterTensorProduct

class _TensorSeed(torch.nn.Module):
    """Lift scalar node embedding to equivariant features via O3 edge TP (TACE tensor seed)."""

    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        num_radial_basis: int,
        avg_num_neighbors: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_channel = num_channel
        self.Lmax = Lmax
        self.lmax = lmax
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.source_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        self.target_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        with torch.no_grad():
            self.source_embedding.weight.uniform_(-0.001, 0.001)
            self.target_embedding.weight.uniform_(-0.001, 0.001)

        self.rejector = O3ScatterTensorProduct(
            [(num_channel, (0, 1))],
            [(1, (l, (-1) ** l)) for l in range(lmax + 1)],
            [(1, (l, (-1) ** l)) for l in range(Lmax + 1)],
        )
        self.irreps_out = self.rejector.irreps_out
        self.edge_info = RadialMLP(
            [
                num_radial_basis + num_channel * 2,
                num_channel,
                num_channel,
                self.rejector.weight_numel,
            ],
            use_layer_norm=True,
            use_offset=False,
            bias=True,
        )

    def forward(
        self,
        base_node_feats: Tensor,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_attrs: Tensor,
        edge_index: Tensor,
        cutoff: Optional[Tensor],
    ) -> Tensor:
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        if cutoff is not None:
            conv_weights = conv_weights * cutoff
        node_feats = (
            self.rejector(
                torch.ones_like(base_node_feats),
                edge_attrs,
                conv_weights,
                edge_index,
            )
            / self.avg_num_neighbors
        )
        node_feats = node_feats.clone()
        node_feats[:, : self.num_channel] = (
            node_feats.narrow(1, 0, self.num_channel) + base_node_feats
        )
        return node_feats



class ClusterEmbed(FieldEmbed):
    def __init__(self, num_elements, node_proj, sph, wigner, edge_embed, seed):
        super().__init__(add_fields=("charge_embedding", "spin_embedding"))
        self.num_elements = num_elements
        self.node_proj = node_proj
        self.sph = sph
        self.wigner = wigner
        self.edge_embed = edge_embed
        self.seed = seed

    def forward(self, inputs, ctx):
        x = self.node_proj(super().forward(inputs, ctx).scalar)
        attrs = F.one_hot(inputs["Za"], num_classes=self.num_elements).to(x.dtype)
        edge = torch.stack((ctx.sender, ctx.receiver))
        cutoff = ctx.cutoff_values.reshape(-1, 1) if ctx.cutoff_values is not None else None
        edge_embed = self.edge_embed(attrs, ctx.rbf, edge)
        vec = -ctx.vec
        sh = self.sph(vec)
        rotation, inverse = self.wigner.get_wigner(vec)
        x = self.seed(x, attrs, ctx.rbf, sh, edge, cutoff)
        return BlockState(scalar=attrs, equi=x, edge=edge_embed,
                          aux={"edge": edge, "rotation": rotation, "inverse": inverse, "cutoff": cutoff})


class ClusterExpansionBlock(Block):
    def __init__(self, edge_update, interaction, product):
        super().__init__()
        self.edge_update = edge_update
        self.interaction = interaction
        self.product = product

    def forward(self, state, ctx):
        aux = state.aux
        features = self.edge_update(state.scalar, state.edge, aux["edge"])
        x, skip = self.interaction(node_feats=state.equi, node_attrs=state.scalar, edge_feats=features,
                                   edge_index=aux["edge"], wigner=aux["rotation"], wigner_inv=aux["inverse"],
                                   radial_basis=ctx.rbf, cutoff=aux["cutoff"])
        x = self.product(node_feats=x, node_attrs=state.scalar, sc=skip)
        return state._replace(equi=x)
