"""ViSNet vector-scalar blocks (Wang et al., Nat. Commun. 2024).

The update follows ``visnet.models.visnet_block`` on
``microsoft/AI2BMD`` branch ``ViSNet`` (MIT). ``idx_i`` is the receiver and
``idx_j`` is the sender, so ``direction`` is the polynomial image of
``(R_j - R_i) / r``. That matches PyG's ``edge_vec = pos[source] - pos[target]``
once ``edge_index[1]`` is the receiver.

Attention is SiLU of the edge-filtered query-key product, times the cosine
window. It is not softmax and it is not scaled by ``sqrt(head_dim)``. The
edge state is part of the representation: every block except the last adds
the inner product of a vector rejection. ``vertex_type="edge"`` also rejects
two maps of the receiver onto the bond, which is the reference angle term.
``vertex_type="node"`` scatters that same receiver term into the scalar
update. Both rejections use the receiver; the reference does not reject the
sender for that term.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module

from ..mp.nn.mlp import xavier_linear as _xavier_linear
from ..mp.nn.field_norm import VECNORM_TYPES, VecLayerNorm, normalize_vecnorm_type  # noqa: F401
from ..mp.ops import scatter_sum
from ..mp.ops.rejection import vector_rejection


VERTEX_TYPES = ("none", "edge", "node")


def normalize_vertex_type(value: object) -> str:
    """Return ``none``, ``edge``, or ``node``.

    The published YAML writes ``vertex_type: None``. That is a null, and the
    reference ``VIS_MP_MAP.get(None, ViS_MP)`` selects the base block. The
    string ``"None"`` selects the same class.
    """
    if value is None:
        return "none"
    key = str(value).strip().lower()
    if key in {"none", "null"}:
        return "none"
    if key not in VERTEX_TYPES:
        choices = ", ".join(VERTEX_TYPES)
        raise ValueError(f"vertex_type must be one of {choices}, got {value!r}")
    return key


def normalize_activation(value: object) -> str:
    """ViSNet's published runs use SiLU. ``swish`` here is that same map."""
    key = str(value).strip().lower()
    if key not in {"silu", "swish"}:
        raise ValueError(
            f"activation must be 'silu' or 'swish' (exact SiLU), got {value!r}"
        )
    return "silu"


class EdgeEmbedding(Module):
    """``f_ij = (h_i + h_j) Linear(rbf)``, including self pairs.

    The radial basis already contains its cosine window. This block does not
    multiply a second one. ``idx_i`` is the receiver. QuinNet uses the same map.
    """

    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = Linear(int(num_rbf), int(hidden_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _xavier_linear(self.edge_proj)

    def forward(
        self,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        scalar: Tensor,
    ) -> Tensor:
        mixed = scalar.index_select(0, idx_i) + scalar.index_select(0, idx_j)
        return mixed * self.edge_proj(rbf)


def _rejection_inner(
    left: Tensor,
    right: Tensor,
    direction: Tensor,
) -> Tensor:
    """Inner product of the two rejections. Shape ``[E, C]``."""
    rejected_left = vector_rejection(left, direction)
    rejected_right = vector_rejection(right, -direction)
    return (rejected_left * rejected_right).sum(dim=1)


class ViSInteraction(Module):
    """One pre-norm ViS-MP block. Returns residuals ``(dx, dvec, df)``.

    ``df`` is ``None`` on the last layer. The caller adds the residuals.
    ``vector`` is ``[N, M, F]`` with ``M`` 3 or 8. ``direction`` is ``[E, M]``.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        vertex_type: str,
        vecnorm_type: str,
        trainable_vecnorm: bool,
        last_layer: bool,
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
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.vertex_type = normalize_vertex_type(vertex_type)
        self.last_layer = bool(last_layer)
        self.layernorm = LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )
        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.dk_proj = Linear(hidden_channels, hidden_channels)
        self.dv_proj = Linear(hidden_channels, hidden_channels)
        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        out_chunks = 4 if self.vertex_type == "node" else 3
        self.o_proj = Linear(hidden_channels, hidden_channels * out_chunks)
        self.f_proj: Optional[Linear]
        self.w_src_proj: Optional[Linear]
        self.w_trg_proj: Optional[Linear]
        self.t_src_proj: Optional[Linear]
        self.t_trg_proj: Optional[Linear]
        if self.last_layer:
            self.f_proj = None
            self.w_src_proj = None
            self.w_trg_proj = None
        else:
            edge_out = hidden_channels * (2 if self.vertex_type == "edge" else 1)
            self.f_proj = Linear(hidden_channels, edge_out)
            self.w_src_proj = Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = Linear(hidden_channels, hidden_channels, bias=False)
        if self.vertex_type == "node" or (
            self.vertex_type == "edge" and not self.last_layer
        ):
            self.t_src_proj = Linear(hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = Linear(hidden_channels, hidden_channels, bias=False)
        else:
            self.t_src_proj = None
            self.t_trg_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        for name in (
            "vec_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "dk_proj",
            "dv_proj",
            "s_proj",
            "o_proj",
            "f_proj",
            "w_src_proj",
            "w_trg_proj",
            "t_src_proj",
            "t_trg_proj",
        ):
            layer = getattr(self, name)
            if isinstance(layer, Linear):
                _xavier_linear(layer)

    def _receiver_angle(self, receiver: Tensor, direction: Tensor) -> Tensor:
        if self.t_src_proj is None or self.t_trg_proj is None:
            raise RuntimeError("receiver angle requested without t projections")
        return _rejection_inner(
            self.t_trg_proj(receiver),
            self.t_src_proj(receiver),
            direction,
        )

    def _edge_residual(
        self,
        vector: Tensor,
        edge: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        direction: Tensor,
    ) -> Tensor:
        if (
            self.f_proj is None
            or self.w_src_proj is None
            or self.w_trg_proj is None
        ):
            raise RuntimeError("edge residual requested on the last layer")
        dihedral = _rejection_inner(
            self.w_trg_proj(vector.index_select(0, idx_i)),
            self.w_src_proj(vector.index_select(0, idx_j)),
            direction,
        )
        filtered = F.silu(self.f_proj(edge))
        if self.vertex_type != "edge":
            return filtered * dihedral
        angle = self._receiver_angle(vector.index_select(0, idx_i), direction)
        first, second = torch.split(filtered, self.hidden_channels, dim=-1)
        return first * dihedral + second * angle

    def _attention(
        self,
        query_i: Tensor,
        key_j: Tensor,
        filter_key: Tensor,
        vector: Tensor,
        idx_j: Tensor,
        direction: Tensor,
        window: Tensor,
    ) -> Tensor:
        """SiLU of the edge-filtered query-key product, times the cosine window.

        ``vector``, ``idx_j``, and ``direction`` are unused. FreeCG overrides
        this method and adds an invariant built from the sender and the bond.
        """
        del vector, idx_j, direction
        attention = (query_i * key_j * filter_key).sum(dim=-1)
        return F.silu(attention) * window.reshape(-1, 1)

    def _vector_residual(
        self,
        vector_gate: Tensor,
        out_scale: Tensor,
        aggregated_vector: Tensor,
    ) -> Tensor:
        """ViSNet's vector update: gated node state plus the aggregated message.

        FreeCG replaces this with a group Clebsch–Gordan product. The residual
        added by the caller is this tensor, not a second copy of the node state.
        """
        return vector_gate * out_scale.unsqueeze(1) + aggregated_vector

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        edge: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        window: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        n_atoms = int(scalar.shape[0])
        if idx_i.shape != idx_j.shape or idx_i.ndim != 1:
            raise ValueError(
                "idx_i and idx_j must be 1D edge indices of the same length, "
                f"got {tuple(idx_i.shape)} and {tuple(idx_j.shape)}"
            )
        n_edges = int(idx_i.shape[0])
        if edge.shape != (n_edges, self.hidden_channels):
            raise ValueError(
                f"edge features must have shape [{n_edges}, {self.hidden_channels}], "
                f"got {tuple(edge.shape)}"
            )
        angular = int(vector.shape[1])
        if vector.shape != (n_atoms, angular, self.hidden_channels):
            raise ValueError(
                "vector features must have shape "
                f"[{n_atoms}, {angular}, {self.hidden_channels}], "
                f"got {tuple(vector.shape)}"
            )
        if direction.shape != (n_edges, angular):
            raise ValueError(
                f"direction must have shape [{n_edges}, {angular}], "
                f"got {tuple(direction.shape)}"
            )
        scalar_n = self.layernorm(scalar)
        vector_n = self.vec_layernorm(vector)
        query = self.q_proj(scalar_n).reshape(n_atoms, self.num_heads, self.head_dim)
        key = self.k_proj(scalar_n).reshape(n_atoms, self.num_heads, self.head_dim)
        value = self.v_proj(scalar_n).reshape(n_atoms, self.num_heads, self.head_dim)
        filter_key = F.silu(self.dk_proj(edge)).reshape(
            n_edges, self.num_heads, self.head_dim
        )
        filter_value = F.silu(self.dv_proj(edge)).reshape(
            n_edges, self.num_heads, self.head_dim
        )
        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vector_n), self.hidden_channels, dim=-1
        )
        vector_dot = (vec1 * vec2).sum(dim=1)

        query_i = query.index_select(0, idx_i)
        key_j = key.index_select(0, idx_j)
        value_j = value.index_select(0, idx_j)
        attention = self._attention(
            query_i, key_j, filter_key, vector_n, idx_j, direction, window
        )
        value_j = value_j * filter_value
        scalar_message = (value_j * attention.unsqueeze(-1)).reshape(
            n_edges, self.hidden_channels
        )
        gate, coeff = torch.split(F.silu(self.s_proj(scalar_message)), self.hidden_channels, dim=-1)
        vector_message = vector_n.index_select(0, idx_j) * gate.unsqueeze(1)
        vector_message = vector_message + coeff.unsqueeze(1) * direction.unsqueeze(-1)

        aggregated_scalar = scatter_sum(scalar_message, idx_i, n_atoms)
        aggregated_vector = scatter_sum(vector_message, idx_i, n_atoms)
        mixed = self.o_proj(aggregated_scalar)
        if self.vertex_type == "node":
            angle = self._receiver_angle(vector_n.index_select(0, idx_i), direction)
            aggregated_angle = scatter_sum(angle, idx_i, n_atoms)
            out_scale, out_dot, out_angle, out_bias = torch.split(
                mixed, self.hidden_channels, dim=-1
            )
            delta_scalar = vector_dot * out_dot + aggregated_angle * out_angle + out_bias
        else:
            out_scale, out_dot, out_bias = torch.split(
                mixed, self.hidden_channels, dim=-1
            )
            delta_scalar = vector_dot * out_dot + out_bias
        delta_vector = self._vector_residual(vec3, out_scale, aggregated_vector)
        delta_edge = None
        if not self.last_layer:
            delta_edge = self._edge_residual(
                vector_n, edge, idx_i, idx_j, direction
            )
        return delta_scalar, delta_vector, delta_edge
