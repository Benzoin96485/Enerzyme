# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
# SO2 / Triton paths adapted from IQuestLab/UBio-MolFM (MIT)
"""E2Former sparse equivariant attention (QK alpha + Wigner-6j / SO2 orders)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from torch import Tensor, nn

from ..activation import SmoothLeakyReLU
from ..blocks.radial_mlp import RadialFunction
from ..so3.linear import SO3Linear
from .so3_ops import SO3Linear2Scalar
from .so2_tensor_product import E2TensorProductSO2_FirstOrder
from .triton_sparse import (
    make_sparse_v_kernel,
    sparse_qk,
    triton_kernels_available,
)
from .wigner6j import E2TensorProductArbitraryOrder

DEFAULT_ATOM_TYPE_COUNT = 256
DEFAULT_HIDDEN_DIM = 128
EMBEDDING_INIT_RANGE = (-0.001, 0.001)


def init_embeddings(
    source_embedding: nn.Embedding,
    target_embedding: nn.Embedding,
    init_range: tuple[float, float] = EMBEDDING_INIT_RANGE,
) -> None:
    nn.init.uniform_(source_embedding.weight.data, *init_range)
    nn.init.uniform_(target_embedding.weight.data, *init_range)


def irreps_times(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    out = [(int(mul * factor), ir) for mul, ir in irreps if int(mul * factor) > 0]
    return o3.Irreps(out)


class QKAlphaModule(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        num_attn_heads: int,
        attn_scalar_head: int,
        edge_channel_list: list[int],
        lmax: int,
    ) -> None:
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.scalar_dim = irreps_node_input[0][0]
        self.query_linear = SO3Linear2Scalar(
            self.scalar_dim, num_attn_heads * attn_scalar_head, lmax=lmax
        )
        self.key_linear = SO3Linear2Scalar(
            self.scalar_dim, num_attn_heads * attn_scalar_head, lmax=lmax
        )
        self.alpha_dot = nn.Parameter(torch.randn(num_attn_heads, attn_scalar_head))
        std = 1.0 / math.sqrt(attn_scalar_head)
        nn.init.uniform_(self.alpha_dot, -std, std)
        self.fc_easy = RadialFunction(edge_channel_list + [num_attn_heads])
        self.alpha_act = SmoothLeakyReLU(0.2)

    def forward(
        self,
        x_edge: Tensor,
        node_irreps_input: Tensor,
        edge_vec: Optional[Tensor] = None,
        f_sparse_idx_node: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        f_n = node_irreps_input.shape[0]
        query = self.query_linear(node_irreps_input).reshape(
            f_n, self.num_attn_heads, -1
        )
        key = self.key_linear(node_irreps_input).reshape(f_n, self.num_attn_heads, -1)
        key = key[f_sparse_idx_node]
        return self.alpha_act(
            self.fc_easy(x_edge)
            * torch.sum(query.unsqueeze(1) * key, dim=3)
            / math.sqrt(query.shape[-1])
        )


class DotAlphaModule(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        num_attn_heads: int,
        attn_scalar_head: int,
        attn_weight_input_dim: int,
        edge_channel_list: list[int],
        lmax: int,
        small_version: bool = False,
    ) -> None:
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.lmax = lmax
        self.scalar_dim = irreps_node_input[0][0]
        dim_factor = 8 if small_version else 1
        self.attn_dim = max(attn_weight_input_dim // dim_factor, 1)
        self.dot_linear = SO3Linear(self.scalar_dim, self.attn_dim, lmax=lmax)
        self.alpha_norm = nn.LayerNorm(attn_scalar_head)
        self.alpha_dot = nn.Parameter(torch.randn(num_attn_heads, attn_scalar_head))
        std = 1.0 / math.sqrt(attn_scalar_head)
        nn.init.uniform_(self.alpha_dot, -std, std)
        self.fc_m0 = nn.Linear(
            2 * self.attn_dim * (lmax + 1), num_attn_heads * attn_scalar_head
        )
        self.rad_func_m0 = RadialFunction(
            edge_channel_list + [2 * self.attn_dim * (lmax + 1)]
        )
        self.alpha_act = SmoothLeakyReLU(0.2)

    def forward(
        self,
        x_edge: Tensor,
        node_irreps_input: Tensor,
        edge_vec: Tensor,
        f_sparse_idx_node: Tensor,
        **kwargs,
    ) -> Tensor:
        f_n = node_irreps_input.shape[0]
        node_dot = self.dot_linear(node_irreps_input)
        extras = []
        for lval in range(self.lmax + 1):
            rij_l = e3nn.o3.spherical_harmonics(
                lval, edge_vec, normalize=True
            ).unsqueeze(-1)
            node_l = node_dot[:, lval**2 : (lval + 1) ** 2]
            extras.append(torch.sum(rij_l * node_l.unsqueeze(1), dim=-2))
            extras.append(torch.sum(rij_l * node_l[f_sparse_idx_node], dim=-2))
        x0 = self.fc_m0(torch.cat(extras, dim=-1) * self.rad_func_m0(x_edge))
        x0 = x0.reshape(f_n, -1, self.num_attn_heads, self.attn_scalar_head)
        x0 = self.alpha_act(self.alpha_norm(x0))
        return torch.einsum("qeik, ik -> qei", x0, self.alpha_dot)


def create_alpha_module(
    tp_type: str,
    irreps_node_input: o3.Irreps,
    num_attn_heads: int,
    attn_scalar_head: int,
    attn_weight_input_dim: int,
    edge_channel_list: list[int],
    lmax: int,
) -> nn.Module:
    tp_type = tp_type.split("+")[0]
    if tp_type == "QK_alpha":
        return QKAlphaModule(
            irreps_node_input,
            num_attn_heads,
            attn_scalar_head,
            edge_channel_list,
            lmax,
        )
    if tp_type.startswith("dot_alpha_small"):
        return DotAlphaModule(
            irreps_node_input,
            num_attn_heads,
            attn_scalar_head,
            attn_weight_input_dim,
            edge_channel_list,
            lmax,
            small_version=True,
        )
    if tp_type.startswith("dot_alpha"):
        return DotAlphaModule(
            irreps_node_input,
            num_attn_heads,
            attn_scalar_head,
            attn_weight_input_dim,
            edge_channel_list,
            lmax,
            small_version=False,
        )
    raise ValueError(f"Unknown tp_type: {tp_type}")


class BaseAttentionOrder(nn.Module, ABC):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_head: o3.Irreps,
        num_attn_heads: int,
        edge_channel_list: list[int],
        lmax: int,
    ) -> None:
        super().__init__()
        self.irreps_node_input = irreps_node_input
        self.irreps_head = irreps_head
        self.num_attn_heads = num_attn_heads
        self.lmax = lmax
        self.scalar_dim = irreps_node_input[0][0]

    @abstractmethod
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        raise NotImplementedError


class ZeroOrderAttention(BaseAttentionOrder):
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax)
        self.rad_func_intputhead = RadialFunction(edge_channel_list + [self.scalar_dim])
        self.proj_zero = SO3Linear(self.scalar_dim, self.scalar_dim, lmax=lmax)

    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        f_n = value.shape[0]
        f_sparse_idx_node = batched_data["f_sparse_idx_node"]
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_n, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        return self.proj_zero(
            torch.sum(alpha.unsqueeze(2) * value[f_sparse_idx_node], dim=1)
        )


class FirstOrderAttention(BaseAttentionOrder):
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax)
        self.rad_func_intputhead = RadialFunction(edge_channel_list + [self.scalar_dim])
        self.first_order_tp = E2TensorProductArbitraryOrder(
            irreps_node_input,
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=1,
            head=self.scalar_dim,
            learnable_weight=True,
            connection_mode="uvw",
        )
        self.proj_first = SO3Linear(
            num_attn_heads * irreps_head[0][0], self.scalar_dim, lmax=lmax
        )

    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        f_n = value.shape[0]
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_n, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))

        exp_node_pos = batched_data.get("f_exp_node_pos", node_pos)
        outcell_index = batched_data.get("f_outcell_index", None)
        f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
        exp_value = value if outcell_index is None else value[outcell_index]

        return self.proj_first(
            self.first_order_tp(
                node_pos,
                exp_node_pos,
                None,
                exp_value,
                alpha / (edge_dis.unsqueeze(-1) + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )


class SecondOrderAttention(BaseAttentionOrder):
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax)
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [self.scalar_dim // 2]
        )
        self.proj_value = SO3Linear(self.scalar_dim, self.scalar_dim // 2, lmax=lmax)
        self.second_order_tp = E2TensorProductArbitraryOrder(
            irreps_times(irreps_node_input, 0.5),
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=2,
            head=self.scalar_dim // 2,
            learnable_weight=True,
            connection_mode="uvw",
        )
        self.proj_sec = SO3Linear(
            num_attn_heads * irreps_head[0][0], self.scalar_dim, lmax=lmax
        )

    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        f_n = value.shape[0]
        value = self.proj_value(value)
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_n, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        exp_node_pos = batched_data.get("f_exp_node_pos", node_pos)
        outcell_index = batched_data.get("f_outcell_index", None)
        f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
        exp_value = value if outcell_index is None else value[outcell_index]
        return self.proj_sec(
            self.second_order_tp(
                node_pos,
                exp_node_pos,
                None,
                exp_value,
                alpha / (edge_dis.unsqueeze(-1) ** 2 + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )


class So2FirstOrderAttention(BaseAttentionOrder):
    """E2Former-V2 value path: EAAS / SO2 first-order TP + on-the-fly Wigner."""

    def __init__(
        self, irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
    ):
        super().__init__(
            irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
        )
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [num_attn_heads]
        )
        self.proj_value = SO3Linear(self.scalar_dim, self.scalar_dim, lmax=lmax)
        self.first_order_tp = E2TensorProductSO2_FirstOrder(
            irreps_node_input,
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=1,
            head=num_attn_heads,
        )
        self.proj_first = SO3Linear(
            num_attn_heads * irreps_head[0][0], self.scalar_dim, lmax=lmax
        )

    @property
    def l3_sequential(self):
        return self.first_order_tp.so2_tp.l3_sequential

    def forward(
        self,
        alpha,
        value,
        x_edge,
        node_pos,
        edge_dis,
        batched_data,
        use_triton: bool = False,
        **kwargs,
    ):
        f_n = value.shape[0]
        value = self.proj_value(value)
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_n, -1, self.num_attn_heads) * inputhead.reshape(
            f_n, -1, self.num_attn_heads
        )

        exp_node_pos = batched_data.get("f_exp_node_pos", node_pos)
        outcell_index = batched_data.get("f_outcell_index", None)
        f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
        exp_value = value if outcell_index is None else value[outcell_index]

        triton_kernel = None
        if use_triton and f_sparse_idx_expnode is not None:
            triton_kernel = make_sparse_v_kernel(
                f_sparse_idx_expnode.contiguous(), use_triton=True
            )

        return self.proj_first(
            self.first_order_tp(
                node_pos,
                exp_node_pos,
                None,
                exp_value,
                alpha / (edge_dis.unsqueeze(-1) + 1e-8),
                triton_kernel=triton_kernel,
                f_sparse_idx_expnode=f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )


def create_attention_order(
    attn_type: str,
    irreps_node_input: o3.Irreps,
    irreps_head: o3.Irreps,
    num_attn_heads: int,
    edge_channel_list: list[int],
    lmax: int,
    attn_weight_input_dim: Optional[int] = None,
) -> BaseAttentionOrder:
    if attn_type == "zero-order":
        return ZeroOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
        )
    if attn_type == "first-order":
        return FirstOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
        )
    if attn_type == "second-order":
        return SecondOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
        )
    if attn_type == "so2-first-order":
        return So2FirstOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, edge_channel_list, lmax
        )
    raise ValueError(
        f"Unknown attention type: {attn_type}. "
        "Supported: zero-order, first-order, second-order, so2-first-order"
    )


class E2AttentionSparse(nn.Module):
    """Sparse E2 attention over padded top-K neighborhoods."""

    def __init__(
        self,
        irreps_node_input: str | o3.Irreps = "64x0e+64x1e+64x2e",
        attn_weight_input_dim: int = 32,
        num_attn_heads: int = 4,
        attn_scalar_head: int = 32,
        irreps_head: str | o3.Irreps = "16x0e+16x1e+16x2e",
        alpha_drop: float = 0.0,
        tp_type: str = "QK_alpha",
        attn_type: str = "first-order",
        atom_type_cnt: int = DEFAULT_ATOM_TYPE_COUNT,
        node_embed_dim: int = DEFAULT_HIDDEN_DIM,
        **kwargs,
    ) -> None:
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.attn_type = attn_type
        tp_parts = tp_type.split("+")
        self.tp_type = tp_parts[0]
        self.want_triton = "triton" in tp_parts[1:]
        self.scalar_dim = self.irreps_node_input[0][0]
        self.lmax = self.irreps_node_input[-1][1].l
        self.node_embed_dim = node_embed_dim

        self.source_embedding = nn.Embedding(atom_type_cnt, node_embed_dim)
        self.target_embedding = nn.Embedding(atom_type_cnt, node_embed_dim)
        init_embeddings(self.source_embedding, self.target_embedding)

        self.edge_channel_list = [
            attn_weight_input_dim + node_embed_dim * 2,
            min(DEFAULT_HIDDEN_DIM, max(attn_weight_input_dim // 2, 8)),
            min(DEFAULT_HIDDEN_DIM, max(attn_weight_input_dim // 2, 8)),
        ]
        self.alpha_module = create_alpha_module(
            self.tp_type,
            self.irreps_node_input,
            num_attn_heads,
            attn_scalar_head,
            attn_weight_input_dim,
            self.edge_channel_list,
            self.lmax,
        )
        self.attention_order_module = create_attention_order(
            attn_type,
            self.irreps_node_input,
            self.irreps_head,
            num_attn_heads,
            self.edge_channel_list,
            self.lmax,
            attn_weight_input_dim,
        )
        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop > 0 else None

    def _use_triton(self) -> bool:
        return bool(self.want_triton and triton_kernels_available())

    def forward(
        self,
        node_pos: Tensor,
        node_irreps_input: Tensor,
        edge_dis: Tensor,
        edge_vec: Tensor,
        attn_weight: Tensor,
        atomic_numbers: Tensor,
        attn_mask: Tensor,
        batched_data: Dict[str, Tensor],
        poly_dist: Optional[Tensor] = None,
        **kwargs,
    ):
        f_n = node_irreps_input.shape[0]
        top_k = attn_weight.shape[1]
        f_sparse_idx_node = batched_data["f_sparse_idx_node"]
        use_triton = self._use_triton()

        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        src_node = self.source_embedding(atomic_numbers)
        tgt_node = self.target_embedding(atomic_numbers)
        x_edge = torch.cat(
            [
                attn_weight,
                tgt_node.reshape(f_n, 1, -1).expand(-1, top_k, -1),
                src_node[f_sparse_idx_node],
            ],
            dim=-1,
        )

        if (
            use_triton
            and self.tp_type == "QK_alpha"
            and isinstance(self.alpha_module, QKAlphaModule)
        ):
            query = self.alpha_module.query_linear(node_irreps_input).reshape(
                f_n, self.num_attn_heads, -1
            )
            key = self.alpha_module.key_linear(node_irreps_input).reshape(
                f_n, self.num_attn_heads, -1
            )
            outcell = batched_data.get("f_outcell_index", None)
            if outcell is not None:
                key = key[outcell]
            idx = batched_data.get("f_sparse_idx_expnode", f_sparse_idx_node)
            gate = self.alpha_module.fc_easy(x_edge)
            scale = 1.0 / math.sqrt(query.shape[-1])
            alpha = sparse_qk(
                query, key, idx.contiguous(), gate, scale, use_triton=True
            )
            alpha = self.alpha_module.alpha_act(alpha)
        else:
            alpha = self.alpha_module(
                x_edge=x_edge,
                node_irreps_input=node_irreps_input,
                edge_vec=edge_vec,
                f_sparse_idx_node=f_sparse_idx_node,
            )
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, dim=1)
        alpha = alpha.masked_fill(attn_mask, 0)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        node_output = self.attention_order_module(
            alpha=alpha,
            value=node_irreps_input,
            x_edge=x_edge,
            node_pos=node_pos,
            edge_dis=edge_dis,
            batched_data=batched_data,
            use_triton=use_triton,
        )
        return node_output, attn_weight
