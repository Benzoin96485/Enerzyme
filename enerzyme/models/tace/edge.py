"""Edge embedding / update blocks for TACE Core.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

from typing import Optional

import torch
from e3nn.nn import Activation
from torch import Tensor, nn

from .linear import IrrepsLinear


class IdentityEdgeEmbedding(nn.Module):
    def __init__(self, num_radial_basis: int, num_channel: int, num_elements: int, bias: bool = False):
        super().__init__()
        self.out_dim = num_radial_basis

    def forward(
        self,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        return edge_feats


class NonLinearEdgeEmbedding(nn.Module):
    def __init__(self, num_radial_basis: int, num_channel: int, num_elements: int, bias: bool = False):
        super().__init__()
        self.out_dim = num_channel
        self.radial_proj = IrrepsLinear(
            f"{num_radial_basis}x0e",
            f"{num_channel}x0e",
            bias=bias,
        )
        self.act = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

    def forward(
        self,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        return self.act(self.radial_proj(edge_feats))


class IdentityEdgeUpdate(nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ):
        super().__init__()
        self.out_dim = edge_embedding_channel

    def forward(
        self,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        return edge_feats


class Element2EdgeUpdate(nn.Module):
    """Concatenate edge feats with target/source element embeddings."""

    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ):
        super().__init__()
        self.out_dim = edge_embedding_channel + num_channel * 2
        self.source_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        self.target_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        with torch.no_grad():
            self.source_embedding.weight.uniform_(-0.001, 0.001)
            self.target_embedding.weight.uniform_(-0.001, 0.001)

    def forward(
        self,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # edge_index: [2, E] with row0=sender, row1=receiver (Enerzyme/MACE + TACE)
        tgt = self.target_embedding(node_attrs[edge_index[1]])
        src = self.source_embedding(node_attrs[edge_index[0]])
        return torch.cat([edge_feats, tgt, src], dim=-1)


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element2": Element2EdgeUpdate,
}
