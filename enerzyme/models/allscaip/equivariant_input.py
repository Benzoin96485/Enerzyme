# SPDX-License-Identifier: MIT
"""Equivariant local angular encoding: TP(node scalar, edge SH) → invariant scalars for messages."""
from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from torch import Tensor

from .nn_utils import get_normalization
from .types import GraphAttentionData


def _irreps_mid_from_sh(lmax: int, mul_per_l: int) -> o3.Irreps:
    pieces: list[str] = []
    for _, ir in o3.Irreps.spherical_harmonics(lmax):
        pieces.append(f"{mul_per_l}x{ir}")
    return o3.Irreps("+".join(pieces))


def scalarize_irreps(x: Tensor, irreps: o3.Irreps) -> Tensor:
    """Rotation-invariant scalars: 0e channels plus L2 norm of each (mul, ir.dim) steerable block."""
    parts: list[Tensor] = []
    i = 0
    for mul, ir in irreps:
        d = mul * ir.dim
        chunk = x[:, i : i + d].reshape(-1, mul, ir.dim)
        if ir.l == 0:
            parts.append(chunk.squeeze(-1))
        else:
            parts.append(chunk.norm(dim=-1))
        i += d
    return torch.cat(parts, dim=-1)


class EquivariantInputBlock(nn.Module):
    """
    LAE path: linearly compressed atom scalars tensor-multiply edge spherical harmonics;
    per-irrep norms (and 0e channels) yield rotation-invariant edge and node features.

    Attention over these scalar messages uses ordinary dot-product logits (invariant tensors).
    The steerable frequency mask on Q/K is disabled in equivariant mode (see core).
    """

    def __init__(
        self,
        hidden_size: int,
        tp_hidden: int,
        lmax_edge: int,
        edge_distance_expansion_size: int,
        tp_mul_per_l: int,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        self.lmax_edge = lmax_edge
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_edge)
        if self.irreps_sh.dim != (lmax_edge + 1) ** 2:
            raise ValueError("unexpected spherical_harmonics dim vs lmax")
        self.irreps_mid = _irreps_mid_from_sh(lmax_edge, tp_mul_per_l)
        irreps_tp_in = o3.Irreps(f"{tp_hidden}x0e")
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_tp_in,
            self.irreps_sh,
            self.irreps_mid,
            internal_weights=True,
        )
        self.inv_dim = sum(mul for mul, _ in self.irreps_mid)
        self.node_to_tp = nn.Linear(hidden_size, tp_hidden, bias=True)
        self.node_inv_linear = nn.Linear(self.inv_dim, hidden_size, bias=True)
        self.node_norm = get_normalization(normalization, hidden_size * 2)
        act = _act_mod(activation)
        self.node_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            act,
        )
        self.edge_scalar_mlp = nn.Sequential(
            nn.Linear(edge_distance_expansion_size + self.inv_dim, hidden_size, bias=True),
            act,
        )
        self.edge_feature_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            act,
        )
        _init_linear_(self)

    def forward(self, data: GraphAttentionData, node_init: Tensor) -> Tensor:
        tp_dtype = next(self.tp.parameters()).dtype
        node_tp = self.node_to_tp(node_init.to(tp_dtype))
        src_nodes = data.neighbor_index[0].clamp(min=0)
        h = node_tp[src_nodes]
        num_nodes, knn_k = h.shape[0], h.shape[1]
        h_flat = h.reshape(num_nodes * knn_k, -1)
        sh = data.edge_direction_expansion.reshape(num_nodes * knn_k, -1).to(
            dtype=h_flat.dtype, device=h_flat.device
        )
        tp_out = self.tp(h_flat, sh)
        inv = scalarize_irreps(tp_out, self.irreps_mid)
        inv = inv.to(dtype=node_init.dtype, device=node_init.device)
        inv_nk = inv.view(num_nodes, knn_k, -1)
        mask = (data.neighbor_index[0] >= 0).to(inv.dtype).unsqueeze(-1)
        neighbor_count = mask.sum(dim=1).clamp(min=1.0)
        node_inv = (inv_nk * mask).sum(dim=1) / neighbor_count
        node_dir_emb = self.node_inv_linear(node_inv)
        node_embeddings = torch.cat([node_init, node_dir_emb], dim=-1)
        node_embeddings = self.node_linear(self.node_norm(node_embeddings))

        radial = data.edge_distance_expansion.reshape(num_nodes * knn_k, -1).to(
            dtype=node_init.dtype, device=node_init.device
        )
        edge_in = torch.cat([radial, inv], dim=-1)
        edge_attr = self.edge_scalar_mlp(edge_in).view(num_nodes, knn_k, -1)
        neighbor_embeddings = self.edge_feature_linear(
            torch.cat([node_embeddings[src_nodes], edge_attr], dim=-1)
        )
        valid = (data.neighbor_index[0] >= 0).to(neighbor_embeddings.dtype).unsqueeze(-1)
        neighbor_embeddings = neighbor_embeddings * valid
        return neighbor_embeddings


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
