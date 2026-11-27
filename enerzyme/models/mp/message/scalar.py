"""Invariant continuous-filter messages (SchNet, PhysNet, SpookyNet s-channel, PaiNN scalar path)."""

from __future__ import annotations

from typing import Callable, Optional

from torch import Tensor, nn

from ..block import BlockState
from ..graph import GraphContext


class FilterMessage(nn.Module):
    """``m_ij = pre(x)[sender] * (filter(rbf_ij) * envelope(d_ij))``.

    ``filter`` maps the radial basis to feature space, ``pre`` transforms the
    sender features before gathering, and ``envelope(dist)`` multiplies the
    filter row-wise. The filter is formed before the product with the gathered
    features so the association order matches PyG's CFConv.
    """

    def __init__(
        self,
        filter: nn.Module,
        pre: Optional[nn.Module] = None,
        envelope: Optional[Callable[[Tensor], Tensor]] = None,
        edge_envelope: bool = False,
    ) -> None:
        super().__init__()
        self.filter = filter
        self.pre = pre
        if edge_envelope and envelope is not None:
            raise ValueError("choose one source for the edge envelope")
        self.edge_envelope = edge_envelope
        self.envelope = envelope

    def forward(self, state: BlockState, ctx: GraphContext) -> Tensor:
        weight = self.filter(ctx.rbf)
        if self.edge_envelope:
            weight = weight * state.edge
        elif self.envelope is not None:
            weight = weight * self.envelope(ctx.dist).view(-1, 1)
        x = state.scalar if self.pre is None else self.pre(state.scalar)
        # index_select backpropagates through index_add_; x[sender] would use the much slower accumulating index_put_ on CPU.
        return x.index_select(0, ctx.sender) * weight
