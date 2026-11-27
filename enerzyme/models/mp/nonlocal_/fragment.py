"""Atom–fragment long-range block for ViSNet-LSRM.

The update follows ``LongShortIneractModel_dis_direct_vector2_drop`` in
``liyy2/LSR-MP`` (the ``Visnorm_shared_LSRMNorm2_2branchSerial`` model).
Query is the atom scalar, key and value are the fragment scalar. Attention is
SiLU of the distance-filtered product, divided by ``sqrt(head_dim)``. The
vector message is gated along ``(P_j - p_i) / r`` and along the fragment
vector. The released code uses that direction; the paper writes
``(p_i - P_j) / r``.

The radial features are an exponential-normal basis of the raw distance.
Section 4.1 also writes a size-dependent distance ``w(Z_i, H_j) r + b``.
That map is not in the released class, so this block does not apply it.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Sequential, SiLU

from ..nn.field_norm import VecLayerNorm
from ..nn.mlp import xavier_linear
from ..ops import scatter_sum


class _FragmentDropout(Dropout):
    def forward(self, values: Tensor) -> Tensor:
        # Functional dropout queries numel under a coordinate VJP. The native
        # operator accepts a symbolic fragment count and saves the sampled mask.
        if torch.compiler.is_compiling() and self.training and 0 < self.p < 1 and not self.inplace:
            return torch.ops.aten.native_dropout.default(values, self.p, True)[0]
        return super().forward(values)


class FragmentLR(Module):
    """One pre-norm fragment→atom update. Returns residuals ``(dx, dvec)``."""

    def __init__(self, hidden_channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"dim_embedding ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout_p = float(dropout)
        self.norm_node = LayerNorm(hidden_channels)
        self.norm_group = LayerNorm(hidden_channels)
        self.norm_node_vec = VecLayerNorm(
            hidden_channels, trainable=False, norm_type="max_min"
        )
        self.norm_group_vec = VecLayerNorm(
            hidden_channels, trainable=False, norm_type="max_min"
        )
        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.mlp_pos = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mlp_vec = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mix = ModuleList(
            Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)
        )
        self.drop_scalar = _FragmentDropout(self.dropout_p)
        self.drop_vector = _FragmentDropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm_node.reset_parameters()
        self.norm_group.reset_parameters()
        self.norm_node_vec.reset_parameters()
        self.norm_group_vec.reset_parameters()
        for module in self.modules():
            if isinstance(module, Linear):
                xavier_linear(module)

    def forward(
        self,
        node_scalar: Tensor,
        node_vector: Tensor,
        group_scalar: Tensor,
        group_vector: Tensor,
        idx_atom: Tensor,
        idx_fragment: Tensor,
        edge_attr: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self._update(
            node_scalar, node_vector, group_scalar, group_vector, idx_atom, idx_fragment,
            edge_attr, direction, defer_vector=False,
        )

    def _vector_message(self, scalar_message, group_vector, idx_fragment, direction):
        fragment_vector = group_vector.index_select(0, idx_fragment)
        message = self.mlp_pos(scalar_message).unsqueeze(1) * direction.unsqueeze(-1)
        return message + self.mlp_vec(scalar_message).unsqueeze(1) * fragment_vector

    def _vector_residual(self, aggregated_vector, aggregated_scalar, vector):
        return aggregated_vector + self.mix[0](aggregated_scalar).unsqueeze(1) * self.mix[1](vector)

    def _update(self, node_scalar, node_vector, group_scalar, group_vector, idx_atom,
                idx_fragment, edge_attr, direction, *, defer_vector):
        n_atoms = int(node_scalar.shape[0])
        scalar = self.norm_node(node_scalar)
        vector = self.norm_node_vec(node_vector)
        group_scalar = self.norm_group(group_scalar)
        group_vector = self.norm_group_vec(group_vector)
        if self.dropout_p > 0.0:
            group_scalar = self.drop_scalar(group_scalar)
            group_vector = self.drop_vector(group_vector)

        n_edges = int(idx_atom.shape[0])
        if not defer_vector and not torch.compiler.is_compiling() and n_edges == 0:
            return node_scalar.new_zeros(n_atoms, self.hidden_channels), node_vector.new_zeros(
                n_atoms, 3, self.hidden_channels
            )

        query = self.q_proj(scalar).reshape(n_atoms, self.num_heads, self.head_dim)
        key = self.k_proj(group_scalar).reshape(-1, self.num_heads, self.head_dim)
        value = self.v_proj(group_scalar).reshape(-1, self.num_heads, self.head_dim)
        query_i = query.index_select(0, idx_atom)
        key_j = key.index_select(0, idx_fragment)
        value_j = value.index_select(0, idx_fragment)
        filt = edge_attr.reshape(n_edges, self.num_heads, self.head_dim)
        attention = (query_i * key_j * filt).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = F.silu(attention)
        scalar_message = (value_j * attention.unsqueeze(-1)).reshape(
            n_edges, self.hidden_channels
        )
        if not defer_vector:
            vector_message = self._vector_message(scalar_message, group_vector, idx_fragment, direction)
        aggregated_scalar = scatter_sum(scalar_message, idx_atom, n_atoms)
        if not defer_vector:
            aggregated_vector = scatter_sum(vector_message, idx_atom, n_atoms)
        left = self.mix[2](vector)
        right = self.mix[3](vector)
        dot = (left * right).sum(dim=1)
        delta_scalar = dot * self.mix[4](aggregated_scalar) + self.mix[5](aggregated_scalar)
        if defer_vector:
            return delta_scalar, (scalar_message, group_vector, aggregated_scalar, vector)
        delta_vector = self._vector_residual(aggregated_vector, aggregated_scalar, vector)
        return delta_scalar, delta_vector
