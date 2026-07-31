"""So3krates interaction layers (Frank et al., NeurIPS 2022).

Module path mirrors PhysNet/SpookyNet ``interaction.py``. Adapted from
So3krates-torch EuclideanTransformer
(https://github.com/TCPUniLU/So3krates-torch, MIT license): fused FeatureBlock +
GeometricBlock attention with InteractionBlock coupling.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn import Identity, Module, Sequential, SiLU
from torch.nn import Linear, LayerNorm
from torch_scatter import scatter_sum

from ..so3 import L0Contraction


def _lecun_normal_(tensor: Tensor) -> None:
    std = 1.0 / (tensor.size(-1) ** 0.5)
    torch.nn.init.normal_(tensor, mean=0.0, std=std)


def _init_linear_lecun(module: Module) -> None:
    if isinstance(module, Linear):
        std = 1.0 / (module.in_features**0.5)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class FilterNet(Module):
    """Radial + SPHC-difference filter (paper / Nat. Commun. Fig. 3e)."""

    def __init__(
        self,
        degrees: Sequence[int],
        num_radial_basis_fn: int,
        num_features: int,
        num_layers: int = 2,
        non_linearity: Type[Module] = SiLU,
    ) -> None:
        super().__init__()
        if num_features % 4 != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by 4"
            )
        self.mlp_rbf = Sequential(
            Linear(num_radial_basis_fn, num_features),
            non_linearity(),
        )
        self.mlp_ev = Sequential(
            Linear(len(degrees), num_features // 4),
            non_linearity(),
        )
        for i in range(num_layers - 1):
            last = i == num_layers - 2
            act: Type[Module] = Identity if last else non_linearity
            self.mlp_rbf.add_module(
                f"mlp_rbf_layer_{i + 1}",
                Sequential(Linear(num_features, num_features), act()),
            )
            if i == 0:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i + 1}",
                    Sequential(
                        Linear(num_features // 4, num_features),
                        act(),
                    ),
                )
            else:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i + 1}",
                    Sequential(Linear(num_features, num_features), act()),
                )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mlp_rbf.apply(_init_linear_lecun)
        self.mlp_ev.apply(_init_linear_lecun)

    def forward(
        self, rbf: Tensor, ev_difference_invariants: Tensor
    ) -> Tensor:
        return self.mlp_rbf(rbf) + self.mlp_ev(ev_difference_invariants)


class EuclideanAttentionBlock(Module):
    """Dual-stream geometric attention (invariant values + SH equivariant values)."""

    def __init__(
        self,
        degrees: Sequence[int],
        num_heads: int,
        num_features: int,
        filter_net_inv: FilterNet,
        filter_net_ev: FilterNet,
        message_normalization: str = "avg_num_neighbors",
        qk_non_linearity: Optional[Union[Type[Module], Callable]] = None,
        avg_num_neighbors: Optional[float] = None,
    ) -> None:
        super().__init__()
        degrees = list(degrees)
        if num_features % num_heads != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by num_heads ({num_heads})"
            )
        if num_features % len(degrees) != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by len(degrees) ({len(degrees)})"
            )

        self.degrees = degrees
        self.num_heads = num_heads
        self.num_features = num_features
        self.message_normalization = message_normalization
        self.avg_num_neighbors = avg_num_neighbors

        self.ev_features_dim = int(sum(2 * y + 1 for y in degrees))
        self.inv_features_dim = num_features
        self.so3_conv_invariants = L0Contraction(degrees=degrees)

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

        _lecun_normal_(self.W_q_inv)
        _lecun_normal_(self.W_k_inv)
        _lecun_normal_(self.W_v_inv)
        _lecun_normal_(self.W_q_ev)
        _lecun_normal_(self.W_k_ev)

        if message_normalization not in {
            "sqrt_num_features",
            "identity",
            "avg_num_neighbors",
        }:
            raise ValueError(
                f"Unknown message_normalization: {message_normalization}"
            )
        if message_normalization == "sqrt_num_features":
            self.att_norm_inv = math.sqrt(self.inv_head_dim)
            self.att_norm_ev = math.sqrt(self.ev_head_dim)
        elif message_normalization == "identity":
            self.att_norm_inv = 1.0
            self.att_norm_ev = 1.0
        else:
            if avg_num_neighbors is None:
                raise ValueError(
                    "avg_num_neighbors required for message_normalization='avg_num_neighbors'"
                )
            self.att_norm_inv = float(avg_num_neighbors)
            self.att_norm_ev = float(avg_num_neighbors)

        if qk_non_linearity is None:
            self.qk_non_linearity: Callable = Identity()
        else:
            try:
                self.qk_non_linearity = qk_non_linearity()  # type: ignore[misc]
            except TypeError:
                self.qk_non_linearity = qk_non_linearity  # type: ignore[assignment]

        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees], dtype=torch.long),
        )

    def _get_qkv(
        self,
        inv_features_inv: Tensor,
        inv_features_ev: Tensor,
        receivers: Tensor,
        senders: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        inv_features: Tensor,
        ev_features: Tensor,
        rbf: Tensor,
        senders: Tensor,
        receivers: Tensor,
        sh_vectors: Tensor,
        cutoffs: Tensor,
        return_att: bool = False,
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
            inv_features_inv=inv_features_inv,
            inv_features_ev=inv_features_ev,
            receivers=receivers,
            senders=senders,
        )
        filtered_k_inv = k_inv * filter_w_inv
        filtered_k_ev = k_ev * filter_w_ev

        alpha_inv = (q_inv * filtered_k_inv).sum(-1, keepdim=True) / self.att_norm_inv
        alpha_ev = (q_ev * filtered_k_ev).sum(-1) / self.att_norm_ev

        scaled_neighbors_inv = cutoffs[:, None, None] * alpha_inv * v_inv
        d_h_att_inv_features = scatter_sum(
            scaled_neighbors_inv,
            receivers,
            dim=0,
            dim_size=inv_features.shape[0],
        )
        d_att_inv_features = d_h_att_inv_features.view(-1, self.inv_features_dim)

        alpha_ev_full = torch.repeat_interleave(
            alpha_ev,
            self.degree_repeats,
            dim=-1,
            output_size=self.ev_features_dim,
        )
        scaled_neighbors_ev = cutoffs[:, None] * alpha_ev_full * sh_vectors
        d_att_ev_features = scatter_sum(
            scaled_neighbors_ev,
            receivers,
            dim=0,
            dim_size=ev_features.shape[0],
        )

        if return_att:
            return (
                d_att_inv_features,
                d_att_ev_features,
                (alpha_inv.detach().clone(), alpha_ev_full.detach().clone()),
            )
        return d_att_inv_features, d_att_ev_features


class InteractionBlock(Module):
    """Couple invariant and SPHC streams via L0 contraction + linear gate."""

    def __init__(
        self,
        degrees: Sequence[int],
        num_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        degrees = list(degrees)
        len_degrees = len(degrees)
        self.linear_layer = Linear(
            num_features + len_degrees,
            num_features + len_degrees,
            bias=bias,
        )
        self.so3_conv_invariants = L0Contraction(degrees=degrees)
        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees], dtype=torch.long),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_linear_lecun(self.linear_layer)

    def forward(
        self, inv_features: Tensor, ev_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ev_invariants = self.so3_conv_invariants(ev_features)
        cat_features = torch.cat([inv_features, ev_invariants], dim=-1)
        transformed = self.linear_layer(cat_features)
        d_inv_features, b_ev_features = torch.split(
            transformed,
            [inv_features.shape[-1], ev_invariants.shape[-1]],
            dim=-1,
        )
        b_ev_features = torch.repeat_interleave(
            b_ev_features,
            self.degree_repeats,
            dim=-1,
            output_size=ev_features.shape[-1],
        )
        d_ev_features = b_ev_features * ev_features
        return d_inv_features, d_ev_features


class EuclideanTransformer(Module):
    """One So3krates layer: attention residual → optional LN/MLP → interaction."""

    def __init__(
        self,
        degrees: Sequence[int],
        num_heads: int,
        num_features: int,
        num_radial_basis_fn: int,
        activation_fn: Type[Module] = SiLU,
        interaction_bias: bool = True,
        message_normalization: str = "avg_num_neighbors",
        avg_num_neighbors: Optional[float] = None,
        filter_net_inv_layers: int = 2,
        filter_net_ev_layers: int = 2,
        layer_normalization_1: bool = False,
        layer_normalization_2: bool = False,
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        qk_non_linearity: Optional[Union[Type[Module], Callable]] = None,
    ) -> None:
        super().__init__()
        self.filter_net_inv = FilterNet(
            degrees=degrees,
            num_features=num_features,
            num_radial_basis_fn=num_radial_basis_fn,
            num_layers=filter_net_inv_layers,
            non_linearity=activation_fn,
        )
        self.filter_net_ev = FilterNet(
            degrees=degrees,
            num_features=num_features,
            num_radial_basis_fn=num_radial_basis_fn,
            num_layers=filter_net_ev_layers,
            non_linearity=activation_fn,
        )
        self.euclidean_attention_block = EuclideanAttentionBlock(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            filter_net_inv=self.filter_net_inv,
            filter_net_ev=self.filter_net_ev,
            message_normalization=message_normalization,
            avg_num_neighbors=avg_num_neighbors,
            qk_non_linearity=qk_non_linearity,
        )
        self.interaction_block = InteractionBlock(
            degrees=degrees,
            num_features=num_features,
            bias=interaction_bias,
        )

        self.layer_normalization_1 = layer_normalization_1
        if layer_normalization_1:
            self.layer_norm_inv_1 = LayerNorm(num_features, eps=1e-6)
        self.layer_normalization_2 = layer_normalization_2
        if layer_normalization_2:
            self.layer_norm_inv_2 = LayerNorm(num_features, eps=1e-6)

        self.residual_mlp_1 = residual_mlp_1
        if residual_mlp_1:
            self.mlp_1 = Sequential(
                activation_fn(),
                Linear(num_features, num_features),
                activation_fn(),
                Linear(num_features, num_features),
            )
        self.residual_mlp_2 = residual_mlp_2
        if residual_mlp_2:
            self.mlp_2 = Sequential(
                activation_fn(),
                Linear(num_features, num_features),
                activation_fn(),
                Linear(num_features, num_features),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.residual_mlp_1:
            self.mlp_1.apply(_init_linear_lecun)
        if self.residual_mlp_2:
            self.mlp_2.apply(_init_linear_lecun)

    def forward(
        self,
        inv_features: Tensor,
        ev_features: Tensor,
        rbf: Tensor,
        senders: Tensor,
        receivers: Tensor,
        sh_vectors: Tensor,
        cutoffs: Tensor,
        return_att: bool = False,
    ):
        att_output = self.euclidean_attention_block(
            inv_features,
            ev_features,
            rbf=rbf,
            senders=senders,
            receivers=receivers,
            sh_vectors=sh_vectors,
            cutoffs=cutoffs,
            return_att=return_att,
        )
        if return_att:
            d_att_inv, d_att_ev, alphas = att_output
        else:
            d_att_inv, d_att_ev = att_output
            alphas = None

        att_inv = inv_features + d_att_inv
        att_ev = ev_features + d_att_ev

        if self.layer_normalization_1:
            att_inv = self.layer_norm_inv_1(att_inv)
        if self.residual_mlp_1:
            att_inv = att_inv + self.mlp_1(att_inv)

        d_inv, d_ev = self.interaction_block(att_inv, att_ev)
        new_inv = att_inv + d_inv
        new_ev = att_ev + d_ev

        if self.residual_mlp_2:
            new_inv = new_inv + self.mlp_2(new_inv)
        if self.layer_normalization_2:
            new_inv = self.layer_norm_inv_2(new_inv)

        if return_att:
            return new_inv, new_ev, alphas
        return new_inv, new_ev
