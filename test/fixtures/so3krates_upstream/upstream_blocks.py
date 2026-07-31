"""Vendored So3krates-torch FilterNet / Attention / Interaction (scatter adapted).

Source: https://github.com/TCPUniLU/So3krates-torch (MIT).
Uses torch_scatter instead of so3krates_torch.tools.scatter.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
from torch.nn import Module
from torch_scatter import scatter_sum

from so3_conv_invariants import L0Contraction


class FilterNet(Module):
    def __init__(
        self,
        degrees: List[int],
        num_radial_basis_fn: int,
        num_features: int,
        num_layers: int = 2,
        non_linearity: Type[Module] = torch.nn.SiLU,
    ):
        super().__init__()
        assert num_features % 4 == 0
        self.mlp_rbf = torch.nn.Sequential(
            torch.nn.Linear(num_radial_basis_fn, num_features),
            non_linearity(),
        )
        self.mlp_ev = torch.nn.Sequential(
            torch.nn.Linear(len(degrees), num_features // 4),
            non_linearity(),
        )
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                non_linearity = torch.nn.Identity
            self.mlp_rbf.add_module(
                f"mlp_rbf_layer_{i+1}",
                torch.nn.Sequential(
                    torch.nn.Linear(num_features, num_features),
                    non_linearity(),
                ),
            )
            if i == 0:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(num_features // 4, num_features),
                        non_linearity(),
                    ),
                )
            else:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(num_features, num_features),
                        non_linearity(),
                    ),
                )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, rbf, ev_difference_invariants):
        return self.mlp_rbf(rbf) + self.mlp_ev(ev_difference_invariants)


class EuclideanAttentionBlock(Module):
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        filter_net_inv: FilterNet,
        filter_net_ev: FilterNet,
        message_normalization: str = "sqrt_num_features",
        qk_non_linearity: Optional[Callable] = None,
        avg_num_neighbors: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.degrees = degrees
        self.num_heads = num_heads
        self.num_features = num_features
        self.ev_features_dim = int(sum(2 * y + 1 for y in degrees))
        self.inv_features_dim = num_features
        self.so3_conv_invariants = L0Contraction(degrees=degrees, device=device)
        self.filter_net_inv = filter_net_inv
        self.filter_net_ev = filter_net_ev
        self.inv_heads = num_heads
        self.inv_head_dim = num_features // num_heads
        self.W_q_inv = torch.nn.Parameter(
            torch.empty(self.inv_heads, self.inv_head_dim, self.inv_head_dim)
        )
        self.W_k_inv = torch.nn.Parameter(
            torch.empty(self.inv_heads, self.inv_head_dim, self.inv_head_dim)
        )
        self.W_v_inv = torch.nn.Parameter(
            torch.empty(self.inv_heads, self.inv_head_dim, self.inv_head_dim)
        )
        self.ev_heads = len(degrees)
        self.ev_head_dim = num_features // len(degrees)
        self.W_q_ev = torch.nn.Parameter(
            torch.empty(self.ev_heads, self.ev_head_dim, self.ev_head_dim)
        )
        self.W_k_ev = torch.nn.Parameter(
            torch.empty(self.ev_heads, self.ev_head_dim, self.ev_head_dim)
        )
        std_inv = 1.0 / (self.W_q_inv.size(-1) ** 0.5)
        torch.nn.init.normal_(self.W_q_inv, mean=0.0, std=std_inv)
        torch.nn.init.normal_(self.W_k_inv, mean=0.0, std=std_inv)
        torch.nn.init.normal_(self.W_v_inv, mean=0.0, std=std_inv)
        std_ev = 1.0 / (self.W_q_ev.size(-1) ** 0.5)
        torch.nn.init.normal_(self.W_q_ev, mean=0.0, std=std_ev)
        torch.nn.init.normal_(self.W_k_ev, mean=0.0, std=std_ev)

        if message_normalization == "sqrt_num_features":
            self.att_norm_inv = math.sqrt(self.inv_head_dim)
            self.att_norm_ev = math.sqrt(self.ev_head_dim)
        elif message_normalization == "identity":
            self.att_norm_inv = 1.0
            self.att_norm_ev = 1.0
        elif message_normalization == "avg_num_neighbors":
            assert avg_num_neighbors is not None
            self.att_norm_inv = avg_num_neighbors
            self.att_norm_ev = avg_num_neighbors
        else:
            raise ValueError(message_normalization)

        try:
            self.qk_non_linearity = qk_non_linearity()
        except TypeError:
            self.qk_non_linearity = qk_non_linearity

        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees], dtype=torch.long),
        )

    def _get_qkv(self, inv_features_inv, inv_features_ev, receivers, senders):
        q_inv = self.qk_non_linearity(
            torch.einsum("nhd,hde->nhe", inv_features_inv, self.W_q_inv)
        )[receivers]
        k_inv = self.qk_non_linearity(
            torch.einsum("nhd,hde->nhe", inv_features_inv, self.W_k_inv)
        )[senders]
        v_inv = torch.einsum("nhd,hde->nhe", inv_features_inv, self.W_v_inv)[
            senders
        ]
        q_ev = self.qk_non_linearity(
            torch.einsum("nhd,hde->nhe", inv_features_ev, self.W_q_ev)
        )[receivers]
        k_ev = self.qk_non_linearity(
            torch.einsum("nhd,hde->nhe", inv_features_ev, self.W_k_ev)
        )[senders]
        return q_inv, k_inv, v_inv, q_ev, k_ev

    def forward(
        self,
        inv_features,
        ev_features,
        rbf,
        senders,
        receivers,
        sh_vectors,
        cutoffs,
        return_att=False,
    ):
        ev_differences = ev_features[senders] - ev_features[receivers]
        ev_differences_invariants = self.so3_conv_invariants(ev_differences)
        filter_w_inv = self.filter_net_inv(rbf, ev_differences_invariants)
        filter_w_ev = self.filter_net_ev(rbf, ev_differences_invariants)
        filter_w_inv = filter_w_inv.reshape(-1, self.inv_heads, self.inv_head_dim)
        filter_w_ev = filter_w_ev.reshape(-1, self.ev_heads, self.ev_head_dim)
        inv_features_inv = inv_features.view(-1, self.inv_heads, self.inv_head_dim)
        inv_features_ev = inv_features.view(-1, self.ev_heads, self.ev_head_dim)
        q_inv, k_inv, v_inv, q_ev, k_ev = self._get_qkv(
            inv_features_inv, inv_features_ev, receivers, senders
        )
        filtered_k_inv = k_inv * filter_w_inv
        filtered_k_ev = k_ev * filter_w_ev
        alpha_inv = (q_inv * filtered_k_inv).sum(-1, keepdim=True) / self.att_norm_inv
        alpha_ev = (q_ev * filtered_k_ev).sum(-1) / self.att_norm_ev
        scaled_neighbors_inv = cutoffs[:, None, None] * alpha_inv * v_inv
        d_h = scatter_sum(
            scaled_neighbors_inv, receivers, dim=0, dim_size=inv_features.shape[0]
        )
        d_att_inv = d_h.view(-1, self.inv_features_dim)
        alpha_ev = torch.repeat_interleave(
            alpha_ev, self.degree_repeats, dim=-1, output_size=self.ev_features_dim
        )
        scaled_neighbors_ev = cutoffs[:, None] * alpha_ev * sh_vectors
        d_att_ev = scatter_sum(
            scaled_neighbors_ev, receivers, dim=0, dim_size=ev_features.shape[0]
        )
        if return_att:
            return d_att_inv, d_att_ev, (alpha_inv.detach(), alpha_ev.detach())
        return d_att_inv, d_att_ev


class InteractionBlock(Module):
    def __init__(self, degrees, num_features, bias=True, device="cpu"):
        super().__init__()
        len_degrees = len(degrees)
        self.linear_layer = torch.nn.Linear(
            num_features + len_degrees, num_features + len_degrees, bias=bias
        )
        self.so3_conv_invariants = L0Contraction(degrees=degrees, device=device)
        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees], dtype=torch.long),
        )

    def forward(self, inv_features, ev_features):
        ev_invariants = self.so3_conv_invariants(ev_features)
        cat = torch.cat([inv_features, ev_invariants], dim=-1)
        transformed = self.linear_layer(cat)
        d_inv, b_ev = torch.split(
            transformed, [inv_features.shape[-1], ev_invariants.shape[-1]], dim=-1
        )
        b_ev = torch.repeat_interleave(
            b_ev, self.degree_repeats, dim=-1, output_size=ev_features.shape[-1]
        )
        return d_inv, b_ev * ev_features
