"""QuinNet interaction (Wang et al., NeurIPS 2023).

The update follows ``src/models/quinnet.py`` in
https://github.com/Zun-Wang/QuinNet (MIT). ``idx_i`` is the receiver and
``idx_j`` is the sender, so ``direction`` is the polynomial image of
``(R_j - R_i) / r``. That is TorchMD-Net's ``edge_vec`` once
``edge_index[1]`` is the receiver.

Attention is SiLU of the edge-filtered query-key product, times the cosine
window. It is not softmax and it is not scaled by ``sqrt(head_dim)``. The
value filter, when present, multiplies the value before that window. The
vector message is a gate on the sender's angular state plus a coefficient
times ``direction``. There is no vector LayerNorm. The reference call is
commented out.

Six scalar channels are mixed by a learned ratio and then gated like the
Equivariant Transformer's vector-dot update:

* three-body, Eq. 4 — inner product of two maps of the angular state
* five-body I, Eq. 7 — squared channel-wise rejection, on the node
* five-body III, Eq. 9 — inner product of the two endpoints' rejected states
* five-body II, Eq. 8 — squared rejection of the receiver, written per edge
* four-body torsion, Eq. 5 — ViSNet rejection inner product along the bond
* four-body improper, Eq. 6 — rejected state dotted with a third map

The three edge channels are summed into the edge state on every layer except
the last. Their node reductions are scattered onto ``idx_j``, which is
TorchMD-Net ``edge_index[0]``. On the bidirectional list that radius graph
builds, the reverse ordered edge is what writes a neighbor's invariant onto
the center. A one-way list would not.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter

from ..et.interaction import normalize_distance_influence
from ..mp.ops import scatter_sum
from ..layers.rejection import vector_rejection
from ..visnet.interaction import normalize_activation


# Paper Eqs. 4, 7, 9, 8, 5, 6, in the order ``ratio`` multiplies them.
BODY_TERMS = (
    "three_body",
    "five_body_i",
    "five_body_iii",
    "five_body_ii",
    "four_body_torsion",
    "four_body_improper",
)


def _xavier_linear(layer: Linear) -> None:
    torch.nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0)


class QuinNetInteraction(Module):
    """One pre-norm QuinNet block. Returns residuals ``(dx, dvec, df)``.

    ``df`` is ``None`` on the last layer. The caller adds the residuals.
    ``vector`` is ``[N, 8, F]`` and ``direction`` is ``[E, 8]``.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        distance_influence: str,
        last_layer: bool,
        activation: object = "silu",
        attn_activation: object = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"dim_embedding ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        normalize_activation(activation)
        normalize_activation(attn_activation)
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.distance_influence = normalize_distance_influence(distance_influence)
        self.last_layer = bool(last_layer)
        self.layernorm = LayerNorm(hidden_channels)
        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        self.o_proj = Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.f_proj1 = Linear(hidden_channels, hidden_channels)
        self.f_proj2 = Linear(hidden_channels, hidden_channels)
        self.f_proj3 = Linear(hidden_channels, hidden_channels)
        self.vec_proj_improper1 = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec_proj_improper2 = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec_proj_improper3 = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec_proj_improper4 = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec_proj_improper5 = Linear(hidden_channels, hidden_channels, bias=False)
        self.dst_proj1 = Linear(hidden_channels, hidden_channels, bias=False)
        self.src_proj1 = Linear(hidden_channels, hidden_channels, bias=False)
        self.src_proj2 = Linear(hidden_channels, hidden_channels, bias=False)
        self.trg_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.src_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.ratio = Parameter(torch.ones(len(BODY_TERMS)))
        self.dk_proj: Optional[Linear]
        self.dv_proj: Optional[Linear]
        if self.distance_influence in ("keys", "both"):
            self.dk_proj = Linear(hidden_channels, hidden_channels)
        else:
            self.dk_proj = None
        if self.distance_influence in ("values", "both"):
            self.dv_proj = Linear(hidden_channels, hidden_channels)
        else:
            self.dv_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        self.ratio.data.fill_(1)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "s_proj",
            "o_proj",
            "vec_proj",
            "f_proj1",
            "f_proj2",
            "f_proj3",
            "vec_proj_improper1",
            "vec_proj_improper2",
            "vec_proj_improper3",
            "vec_proj_improper4",
            "vec_proj_improper5",
            "dst_proj1",
            "src_proj1",
            "src_proj2",
            "trg_proj",
            "src_proj",
            "dk_proj",
            "dv_proj",
        ):
            layer = getattr(self, name)
            if isinstance(layer, Linear):
                _xavier_linear(layer)

    def _edge_invariants(
        self,
        vector: Tensor,
        normal_i: Tensor,
        edge: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Edge channels for five-body III, five-body II, and torsion.

        ``normal_i`` is the five-body I rejection at every atom. The
        reference names ``edge_index[0]`` ``dst`` and ``edge_index[1]``
        ``src``, so ``dst`` is the sender and ``src`` is the receiver.
        """
        sender = vector.index_select(0, idx_j)
        receiver = vector.index_select(0, idx_i)
        normal_sender = normal_i.index_select(0, idx_j)
        normal_receiver = normal_i.index_select(0, idx_i)
        five_iii = F.silu(self.f_proj2(edge)) * (
            self.dst_proj1(normal_sender) * self.src_proj1(normal_receiver)
        ).sum(dim=1)
        five_ii = F.silu(self.f_proj3(edge)) * self.src_proj2(
            normal_receiver
        ).square().sum(dim=1)
        rejected_receiver = vector_rejection(self.trg_proj(receiver), direction)
        rejected_sender = vector_rejection(self.src_proj(sender), -direction)
        torsion = F.silu(self.f_proj1(edge)) * (
            rejected_receiver * rejected_sender
        ).sum(dim=1)
        return five_iii, five_ii, torsion

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
        normed = self.layernorm(scalar)
        query = self.q_proj(normed).reshape(n_atoms, self.num_heads, self.head_dim)
        key = self.k_proj(normed).reshape(n_atoms, self.num_heads, self.head_dim)
        value = self.v_proj(normed).reshape(n_atoms, self.num_heads, self.head_dim)
        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vector), self.hidden_channels, dim=-1
        )
        three_body = (vec1 * vec2).sum(dim=1)
        normal_i = vector_rejection(
            self.vec_proj_improper1(vector), self.vec_proj_improper2(vector)
        )
        five_i = normal_i.square().sum(dim=1)
        normal_improper = vector_rejection(
            self.vec_proj_improper3(vector), self.vec_proj_improper4(vector)
        )
        improper = (normal_improper * self.vec_proj_improper5(vector)).sum(dim=1)
        five_iii_edge, five_ii_edge, torsion_edge = self._edge_invariants(
            vector, normal_i, edge, idx_i, idx_j, direction
        )
        node_shape = (n_atoms, self.hidden_channels)
        five_iii = scatter_sum(five_iii_edge, idx_j, node_shape[0])
        five_ii = scatter_sum(five_ii_edge, idx_j, node_shape[0])
        torsion = scatter_sum(torsion_edge, idx_j, node_shape[0])

        filter_key = None
        if self.dk_proj is not None:
            filter_key = F.silu(self.dk_proj(edge)).reshape(
                n_edges, self.num_heads, self.head_dim
            )
        filter_value = None
        if self.dv_proj is not None:
            filter_value = F.silu(self.dv_proj(edge)).reshape(
                n_edges, self.num_heads, self.head_dim
            )
        query_i = query.index_select(0, idx_i)
        key_j = key.index_select(0, idx_j)
        value_j = value.index_select(0, idx_j)
        if filter_key is None:
            attention = (query_i * key_j).sum(dim=-1)
        else:
            attention = (query_i * key_j * filter_key).sum(dim=-1)
        attention = F.silu(attention) * window.reshape(-1, 1)
        if filter_value is not None:
            value_j = value_j * filter_value
        scalar_message = (value_j * attention.unsqueeze(-1)).reshape(
            n_edges, self.hidden_channels
        )
        gate, coeff = torch.split(
            F.silu(self.s_proj(scalar_message)), self.hidden_channels, dim=-1
        )
        vector_message = vector.index_select(0, idx_j) * gate.unsqueeze(1)
        vector_message = vector_message + coeff.unsqueeze(1) * direction.unsqueeze(-1)
        aggregated_scalar = scatter_sum(scalar_message, idx_i, node_shape[0])
        aggregated_vector = scatter_sum(vector_message, idx_i, n_atoms)
        out_scale, out_gate, out_bias = torch.split(
            self.o_proj(aggregated_scalar), self.hidden_channels, dim=-1
        )
        terms = torch.stack(
            (three_body, five_i, five_iii, five_ii, torsion, improper), dim=0
        )
        mixed = (terms * self.ratio.to(dtype=terms.dtype).view(-1, 1, 1)).sum(dim=0)
        delta_scalar = mixed * out_gate + out_bias
        delta_vector = vec3 * out_scale.unsqueeze(1) + aggregated_vector
        delta_edge = None
        if not self.last_layer:
            delta_edge = torsion_edge + five_iii_edge + five_ii_edge
        return delta_scalar, delta_vector, delta_edge
