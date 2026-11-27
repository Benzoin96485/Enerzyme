"""Node updates: ``update(state, aggregated, ctx) -> BlockState``."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .block import BlockState
from .graph import GraphContext


class ResidualUpdate(nn.Module):
    """``scalar + net(aggregated)``; ``net=None`` adds the aggregate directly."""

    def __init__(self, net: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.net = net

    def forward(self, state: BlockState, aggregated: Tensor, ctx: GraphContext) -> BlockState:
        delta = aggregated if self.net is None else self.net(aggregated)
        return state._replace(scalar=state.scalar + delta)
