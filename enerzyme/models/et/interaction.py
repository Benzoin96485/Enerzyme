"""Equivariant Transformer blocks (Thölke and De Fabritiis, ICLR 2022).

The update is the pure PyTorch path in ``torchmdnet.models.torchmd_et``
(MIT). Warp neighbor kernels are not used. ``idx_i`` is the receiver and
``idx_j`` is the sender, so ``direction`` is the unit vector
``(R_j - R_i) / r``. That is TorchMD-Net's ``edge_vec`` once both
directions of a pair are present: their ``edge_index[1]`` is the receiver
and ``edge_index[0]`` is the sender.

Attention is SiLU of the distance-filtered query-key product. It is not
softmax, and it is not scaled by ``sqrt(head_dim)``. The scalar value is
weighted by that score. The vector value is a gate on the sender's vector
plus a coefficient times the unit direction. A pre-norm residual then mixes
the current vector channels the way PaiNN's update mixes ``u`` and ``v``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module

from ..mp.ops import scatter_sum
from ..mp.message.attention import dot_product_attention_logits
from ..layers.atom_embedding import RandomAtomEmbedding


DISTANCE_INFLUENCES = ("keys", "values", "both", "none")


def normalize_distance_influence(value: str) -> str:
    """Return ``keys``, ``values``, ``both``, or ``none``."""
    key = str(value).strip().lower()
    if key not in DISTANCE_INFLUENCES:
        choices = ", ".join(DISTANCE_INFLUENCES)
        raise ValueError(
            f"distance_influence must be one of {choices}, got {value!r}"
        )
    return key


def _xavier_linear(layer: Linear) -> None:
    torch.nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0)


class ETNeighborEmbedding(Module):
    """Second species table, distance-filtered and mixed with the atom state.

    TorchMD-Net assigns each element two vectors. :class:`RandomAtomEmbedding`
    owns the first. This block owns the neighborhood table (paper Eq. 3) and
    reuses the same embedding module, including its checkpoint resize. Self
    pairs are dropped, and the sum is written onto ``idx_i`` (the receiver).
    TorchMD-Net's neighbor embedding scatters onto ``edge_index[0]`` instead.
    On a bidirectional list the two sums match. Attention still reads
    ``edge_index[1]`` as the receiver.
    """

    def __init__(self, hidden_channels: int, num_rbf: int, max_Za: int) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_rbf = int(num_rbf)
        self.max_Za = int(max_Za)
        self.species = RandomAtomEmbedding(self.max_Za, self.hidden_channels)
        self.distance_proj = Linear(self.num_rbf, self.hidden_channels)
        self.combine = Linear(self.hidden_channels * 2, self.hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _xavier_linear(self.distance_proj)
        _xavier_linear(self.combine)

    def forward(
        self,
        atomic_numbers: Tensor,
        scalar: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        window: Tensor,
    ) -> Tensor:
        n_atoms = int(scalar.shape[0])
        keep = idx_i != idx_j
        if not bool(keep.all()):
            idx_i = idx_i[keep]
            idx_j = idx_j[keep]
            rbf = rbf[keep]
            window = window[keep]
        species = self.species.get_embedding(atomic_numbers).to(dtype=scalar.dtype)
        weight = self.distance_proj(rbf) * window.unsqueeze(-1)
        message = weight * species.index_select(0, idx_j)
        aggregated = scatter_sum(message, idx_i, n_atoms)
        return self.combine(torch.cat((scalar, aggregated), dim=-1))


class EquivariantMultiHeadAttention(Module):
    """One pre-norm equivariant attention block.

    Returns the residual ``(dx, dvec)`` added by the caller. ``vector_cutoff``
    multiplies the value tensor by the cosine window. The 2022 architecture
    multiplies the attention score instead, which leaves the value bias active
    at the cutoff.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        num_heads: int,
        distance_influence: str,
        vector_cutoff: bool,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"dim_embedding ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.hidden_channels = hidden_channels
        self.num_rbf = int(num_rbf)
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.distance_influence = normalize_distance_influence(distance_influence)
        self.vector_cutoff = bool(vector_cutoff)
        self.layernorm = LayerNorm(hidden_channels)
        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.dk_proj: Optional[Linear]
        self.dv_proj: Optional[Linear]
        if self.distance_influence in ("keys", "both"):
            self.dk_proj = Linear(self.num_rbf, hidden_channels)
        else:
            self.dk_proj = None
        if self.distance_influence in ("values", "both"):
            self.dv_proj = Linear(self.num_rbf, hidden_channels * 3)
        else:
            self.dv_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.vec_proj):
            _xavier_linear(layer)
        if self.dk_proj is not None:
            _xavier_linear(self.dk_proj)
        if self.dv_proj is not None:
            _xavier_linear(self.dv_proj)

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        window: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n_atoms = int(scalar.shape[0])
        normed = self.layernorm(scalar)
        query = self.q_proj(normed).reshape(n_atoms, self.num_heads, self.head_dim)
        key = self.k_proj(normed).reshape(n_atoms, self.num_heads, self.head_dim)
        value = self.v_proj(normed).reshape(
            n_atoms, self.num_heads, self.head_dim * 3
        )
        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vector), self.hidden_channels, dim=-1
        )
        vector_dot = (vec1 * vec2).sum(dim=1)
        vector_heads = vector.reshape(n_atoms, 3, self.num_heads, self.head_dim)

        filter_key = None
        if self.dk_proj is not None:
            filter_key = F.silu(self.dk_proj(rbf)).reshape(
                -1, self.num_heads, self.head_dim
            )
        filter_value = None
        if self.dv_proj is not None:
            filter_value = F.silu(self.dv_proj(rbf)).reshape(
                -1, self.num_heads, self.head_dim * 3
            )

        query_i = query.index_select(0, idx_i)
        key_j = key.index_select(0, idx_j)
        value_j = value.index_select(0, idx_j)
        vector_j = vector_heads.index_select(0, idx_j)
        attention = dot_product_attention_logits(query_i, key_j, filter_key, activation=F.silu)
        cutoff = window.reshape(-1, 1)
        if self.vector_cutoff:
            value_j = value_j * cutoff.unsqueeze(-1)
        else:
            attention = attention * cutoff
        if filter_value is not None:
            value_j = value_j * filter_value
        scalar_message, gate, coeff = torch.split(value_j, self.head_dim, dim=2)
        scalar_message = scalar_message * attention.unsqueeze(-1)
        vector_message = vector_j * gate.unsqueeze(1) + coeff.unsqueeze(1) * direction.reshape(
            -1, 3, 1, 1
        )

        aggregated_scalar = scatter_sum(scalar_message, idx_i, n_atoms).reshape(n_atoms, self.hidden_channels)
        aggregated_vector = scatter_sum(vector_message, idx_i, n_atoms).reshape(n_atoms, 3, self.hidden_channels)

        out_scalar, out_gate, out_bias = torch.split(
            self.o_proj(aggregated_scalar), self.hidden_channels, dim=-1
        )
        delta_scalar = vector_dot * out_gate + out_bias
        delta_vector = vec3 * out_scalar.unsqueeze(1) + aggregated_vector
        return delta_scalar, delta_vector
