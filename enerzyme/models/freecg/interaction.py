"""FreeCG vector-scalar block on top of ViSNet.

The scalar update, the edge rejection, and the runtime geometry stay in
:class:`~enerzyme.models.visnet.interaction.ViSInteraction`. This block
changes two steps:

* Attention adds an invariant of the sender's angular state and the bond
  direction (Eqs. 6–8). ``linear`` is the released projection onto the
  heads. ``max`` is the channel-wise maximum written in Eq. 8. ``none``
  is ViSNet's score.
* The vector residual is the grouped sparse Clebsch–Gordan product of the
  aggregated abstract edge with ``o1 * Linear(E)``, rolled along channels,
  plus the abstract edge (Eqs. 5 and 14). The caller still adds that
  residual onto the node state.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from ..visnet.interaction import ViSInteraction, _xavier_linear
from .cg import GroupSparseCG


ATTENTION_ENHANCERS = ("linear", "max", "none")


def normalize_attention_enhancer(value: object) -> str:
    """Return ``linear``, ``max``, or ``none``."""
    key = str(value).strip().lower()
    if key not in ATTENTION_ENHANCERS:
        choices = ", ".join(ATTENTION_ENHANCERS)
        raise ValueError(
            f"attention_enhancer must be one of {choices}, got {value!r}"
        )
    return key


class FreeCGInteraction(ViSInteraction):
    """One pre-norm FreeCG block. Returns residuals ``(dx, dvec, df)``."""

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        vertex_type: str,
        vecnorm_type: str,
        trainable_vecnorm: bool,
        last_layer: bool,
        num_groups: int,
        shuffle_scale: float = 1.5,
        attention_enhancer: object = "linear",
    ) -> None:
        super().__init__(
            hidden_channels,
            num_heads,
            vertex_type,
            vecnorm_type,
            trainable_vecnorm,
            last_layer,
        )
        self.num_groups = int(num_groups)
        self.shuffle_scale = float(shuffle_scale)
        self.attention_enhancer = normalize_attention_enhancer(attention_enhancer)
        self.group_cg = GroupSparseCG(
            self.hidden_channels,
            self.num_groups,
            self.shuffle_scale,
        )
        if self.attention_enhancer == "linear":
            self.vec_edge_proj = Linear(self.hidden_channels, self.num_heads)
        else:
            self.vec_edge_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        group_cg = getattr(self, "group_cg", None)
        if group_cg is not None:
            group_cg.reset_parameters()
        if isinstance(getattr(self, "vec_edge_proj", None), Linear):
            _xavier_linear(self.vec_edge_proj)

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
        logits = (query_i * key_j * filter_key).sum(dim=-1)
        if self.attention_enhancer != "none":
            sender = vector.index_select(0, idx_j)
            # Sum over the polynomial components. Each degree is an invariant
            # contraction, so the channel vector is O(3) invariant.
            dots = (sender * direction.unsqueeze(-1)).sum(dim=1)
            if self.attention_enhancer == "max":
                extra = dots.amax(dim=-1, keepdim=True)
            else:
                extra = self.vec_edge_proj(dots)
            logits = logits + extra
        return F.silu(logits) * window.reshape(-1, 1)

    def _vector_residual(
        self,
        vector_gate: Tensor,
        out_scale: Tensor,
        aggregated_vector: Tensor,
    ) -> Tensor:
        gated = vector_gate * out_scale.unsqueeze(1)
        mixed = self.group_cg(aggregated_vector, gated)
        return mixed + aggregated_vector
