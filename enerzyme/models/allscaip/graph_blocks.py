# SPDX-License-Identifier: MIT
# Adapted from fairchem AllScAIP (https://github.com/FAIR-Chem/fairchem), MIT License.
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .neighborhood_attention import NeighborhoodAttention
from .nn_utils import get_feedforward, get_normalization
from .node_attention import NodeAttention
from .types import GraphAttentionData


class FeedForwardNetwork(nn.Module):
    """Pre-norm FFN with residual scaling (fairchem AllScAIP)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        activation: str,
        ffn_hidden_layer_multiplier: int,
        mlp_dropout: float,
        use_residual_scaling: bool,
        normalization: str,
    ) -> None:
        super().__init__()
        self.ffn = get_feedforward(
            hidden_dim=hidden_size,
            activation_name=activation,
            hidden_layer_multiplier=ffn_hidden_layer_multiplier,
            bias=True,
            dropout=mlp_dropout,
        )
        self.ffn_norm = get_normalization(normalization, hidden_size)
        if use_residual_scaling:
            self.ffn_res_scale = nn.Parameter(
                torch.tensor(1.0 / num_layers), requires_grad=True
            )
        else:
            self.ffn_res_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn_res_scale * self.ffn(self.ffn_norm(x)) + x


class InputBlock(nn.Module):
    """
    Edge and neighbor embeddings from graph tensors plus Enerzyme pre-core node features.

    ``node_init`` replaces fairchem's atomic embedding + charge/spin linear (already fused
    from NuclearEmbedding / ElectronicEmbedding in Enerzyme).
    """

    def __init__(
        self,
        hidden_size: int,
        node_direction_expansion_size: int,
        edge_distance_expansion_size: int,
        edge_direction_sh_dim: int,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        act = _act_mod(activation)
        self.node_direction_embedding = nn.Linear(node_direction_expansion_size, hidden_size)
        self.node_norm = get_normalization(normalization, hidden_size * 2)
        self.node_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            act,
        )
        edge_in = edge_distance_expansion_size + edge_direction_sh_dim
        self.edge_attr_linear = nn.Sequential(
            nn.Linear(edge_in, hidden_size, bias=True),
            act,
        )
        self.edge_feature_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            act,
        )
        _init_linear_(self)

    def forward(self, data: GraphAttentionData, node_init: Tensor) -> Tensor:
        node_direction_embedding = self.node_direction_embedding(data.node_direction_expansion)
        node_embeddings = torch.cat([node_init, node_direction_embedding], dim=-1)
        node_embeddings = self.node_linear(self.node_norm(node_embeddings))

        edge_attr = self.edge_attr_linear(
            torch.cat([data.edge_distance_expansion, data.edge_direction_expansion], dim=-1)
        )
        src_nodes = data.neighbor_index[0].clamp(min=0)
        neighbor_embeddings = self.edge_feature_linear(
            torch.cat([node_embeddings[src_nodes], edge_attr], dim=-1)
        )
        valid = (data.neighbor_index[0] >= 0).to(neighbor_embeddings.dtype).unsqueeze(-1)
        neighbor_embeddings = neighbor_embeddings * valid
        return neighbor_embeddings


class GraphAttentionBlock(nn.Module):
    """Neighborhood attention → edge FFN → optional node attention → node FFN (fairchem AllScAIP)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        use_node_path: bool,
        attn_num_heads: int,
        attn_num_freq: int,
        atten_dropout: float,
        mlp_dropout: float,
        activation: str,
        ffn_hidden_layer_multiplier: int,
        use_freq_mask: bool,
        freequency_list: list[int],
        use_sincx_mask: bool,
        use_residual_scaling: bool,
        normalization: str,
    ) -> None:
        super().__init__()
        self.use_node_path = use_node_path
        self.neighborhood_attention = NeighborhoodAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_num_heads=attn_num_heads,
            atten_dropout=atten_dropout,
            use_freq_mask=use_freq_mask,
            freequency_list=freequency_list,
            use_residual_scaling=use_residual_scaling,
            normalization=normalization,
        )
        self.edge_ffn = FeedForwardNetwork(
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            ffn_hidden_layer_multiplier=ffn_hidden_layer_multiplier,
            mlp_dropout=mlp_dropout,
            use_residual_scaling=use_residual_scaling,
            normalization=normalization,
        )
        if use_node_path:
            self.node_attention = NodeAttention(
                hidden_size=hidden_size,
                num_layers=num_layers,
                attn_num_heads=attn_num_heads,
                attn_num_freq=attn_num_freq,
                atten_dropout=atten_dropout,
                use_sincx_mask=use_sincx_mask,
                use_residual_scaling=use_residual_scaling,
                normalization=normalization,
            )
            self.node_ffn = FeedForwardNetwork(
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
                ffn_hidden_layer_multiplier=ffn_hidden_layer_multiplier,
                mlp_dropout=mlp_dropout,
                use_residual_scaling=use_residual_scaling,
                normalization=normalization,
            )

    def forward(self, data: GraphAttentionData, neighbor_reps: Tensor, layer_idx: int = 0) -> Tensor:
        _ = layer_idx
        neighbor_reps = self.neighborhood_attention(data, neighbor_reps)
        node_reps = neighbor_reps[:, 0]
        edge_reps = self.edge_ffn(neighbor_reps[:, 1:])
        if self.use_node_path:
            node_reps = self.node_attention(data, node_reps)
            node_reps = self.node_ffn(node_reps)
        neighbor_reps = torch.cat([node_reps.unsqueeze(1), edge_reps], dim=1)
        return neighbor_reps


def _act_mod(name: str) -> nn.Module:
    n = name.lower()
    if n in ("silu", "swish"):
        return nn.SiLU()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU()
    return nn.SiLU()


def _init_linear_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
