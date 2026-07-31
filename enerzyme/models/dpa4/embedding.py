"""Initial embeddings for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .edge_cache import EdgeCache
from .indexing import build_gie_zonal_index, get_so3_dim


class GeometricInitialEmbedding(nn.Module):
    """Geometric initial embedding: zonal (m=0) rotated radial features.

    For l>=1, rotates radial features using Dt_full zonal coupling.
    l=0 component is zero (comes from type embedding).
    """

    def __init__(self, lmax: int, channels: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.ebed_dim = get_so3_dim(lmax)

        row, m0_col, rad_l = build_gie_zonal_index(lmax)
        self.register_buffer("non_scalar_row", torch.from_numpy(row).long())
        self.register_buffer("zonal_m0_col", torch.from_numpy(m0_col).long())
        self.register_buffer("radial_slot", torch.from_numpy(rad_l).long())

    def forward(self, n_nodes: int, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            n_nodes: N
            edge_cache: EdgeCache with Dt_full, dst, inv_sqrt_deg
            radial_feat: (E, lmax, C) for l=1..lmax

        Returns:
            (N, D, C) initial features (l=0 row is zero)
        """
        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype

        if self.lmax == 0:
            return torch.zeros(n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype)

        Dt = edge_cache.Dt_full  # (E, D, D)
        n_edge = Dt.shape[0]
        dim_full = Dt.shape[-1]

        # Gather zonal coupling: Dt[e, row, m0_col] for each non-scalar position
        flat_idx = self.non_scalar_row * dim_full + self.zonal_m0_col
        zonal = Dt.reshape(n_edge, dim_full * dim_full)[:, flat_idx]  # (E, D-1)

        # Broadcast radial features
        rad_val = radial_feat[:, self.radial_slot, :]  # (E, D-1, C)
        message = zonal.unsqueeze(-1) * rad_val  # (E, D-1, C)

        # Scatter to nodes
        non_scalar_out = torch.zeros(n_nodes, self.ebed_dim - 1, self.channels,
                                      device=device, dtype=dtype)
        dst_expand = edge_cache.dst.unsqueeze(-1).unsqueeze(-1).expand_as(message)
        non_scalar_out.scatter_add_(0, dst_expand, message)

        # Prepend zero l=0 row
        zero_l0 = torch.zeros(n_nodes, 1, self.channels, device=device, dtype=dtype)
        out = torch.cat([zero_l0, non_scalar_out], dim=1)

        # Normalize by smooth degree
        out = out * edge_cache.inv_sqrt_deg.to(out.dtype)
        return out


class EnvironmentInitialEmbedding(nn.Module):
    """Environment matrix initial embedding for l=0 (FiLM conditioning).

    Builds r_tilde = [s, s*r_hat], computes G network, aggregates outer
    product by destination, constructs D matrix, projects to FiLM logits.
    """

    def __init__(
        self,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.channels = channels
        self.embed_dim = embed_dim
        self.axis_dim = axis_dim
        self.eps = eps
        self.coord_dim = 4

        rbf_out_dim = max(32, embed_dim - 2 * type_dim)
        self.rbf_proj = nn.Sequential(
            nn.Linear(n_radial, rbf_out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(rbf_out_dim, rbf_out_dim, bias=False),
        )

        self.env_type_embed = nn.Embedding(ntypes + 1, type_dim, padding_idx=ntypes)
        nn.init.normal_(self.env_type_embed.weight[:ntypes], std=1.0 / math.sqrt(ntypes + type_dim))
        self.env_type_embed.weight.data[ntypes] = 0.0

        g_in = rbf_out_dim + 2 * type_dim
        self.g_net = nn.Sequential(
            nn.Linear(g_in, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )

        self.output_proj = nn.Linear(embed_dim * axis_dim, 2 * channels, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, edge_cache: EdgeCache, atype: Tensor, n_nodes: int) -> Tensor:
        """Returns FiLM logits (N, 2*C)."""
        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec
        edge_rbf = edge_cache.edge_rbf
        edge_env = edge_cache.edge_env
        device = edge_vec.device

        r_sq = (edge_vec * edge_vec).sum(-1, keepdim=True)
        inv_r = torch.rsqrt(r_sq + self.eps ** 2)
        s = edge_env * inv_r
        r_hat = edge_vec * inv_r
        r_tilde = torch.cat([s, s * r_hat], dim=-1)  # (E, 4)

        atype_src = atype[src]
        atype_dst = atype[dst]
        type_src = self.env_type_embed(atype_src)
        type_dst = self.env_type_embed(atype_dst)

        rbf_proj = self.rbf_proj(edge_rbf)
        g_input = torch.cat([rbf_proj, type_src, type_dst], dim=-1)
        g = self.g_net(g_input)  # (E, embed_dim)

        outer = r_tilde.unsqueeze(-1) * g.unsqueeze(-2)  # (E, 4, embed_dim)
        outer_flat = outer.reshape(-1, self.coord_dim * self.embed_dim)

        env_agg = torch.zeros(n_nodes, self.coord_dim * self.embed_dim,
                              device=device, dtype=outer_flat.dtype)
        env_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(outer_flat), outer_flat)
        env_agg = env_agg.reshape(n_nodes, self.coord_dim, self.embed_dim)

        env_agg = env_agg * edge_cache.inv_sqrt_deg.to(env_agg.dtype)

        env_agg_t = env_agg.permute(0, 2, 1)  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, :self.axis_dim]  # (N, 4, axis_dim)
        D = torch.bmm(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        D_flat = D.reshape(n_nodes, self.embed_dim * self.axis_dim)
        return self.output_proj(D_flat)
