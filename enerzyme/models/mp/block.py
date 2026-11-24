"""Block protocol for message passing.

A Core is ``embed → blocks → head``. Every block maps ``(BlockState,
GraphContext) → BlockState``:

* ``scalar``: invariant node features, usually ``(N, F)``
* ``equi``: equivariant node features in the Core's layout (``IrrepsFlat``
  ``(N, dim)``, ``SphArray`` ``(N, (L+1)^2, C)``, Cartesian ``(N, 3, F)``,
  rank-2 tensors ``(N, F, 3, 3)``, or a tuple/dict of those)
* ``edge``: per-edge state carried between blocks (DPA3, ALIGNN, eSEN, ...)
* ``aux``: anything else a later block or the head reads (per-block energies,
  attention caches, non-local summaries)

``InteractionBlock`` is the generic ``message → aggregate → [body order] →
update`` composition; blocks whose numerics require a fused kernel implement
``forward`` directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional

from torch import Tensor, nn

from .graph import GraphContext
from .aggregate import Aggregate


class BlockState(NamedTuple):
    scalar: Optional[Tensor] = None
    equi: Any = None
    edge: Any = None
    aux: Optional[Dict[str, Any]] = None

    def with_aux(self, **items: Any) -> "BlockState":
        aux = dict(self.aux) if self.aux else {}
        aux.update(items)
        return self._replace(aux=aux)

    def get(self, key: str, default: Any = None) -> Any:
        return self.aux.get(key, default) if self.aux else default


class Block(nn.Module):
    """Base class for message-passing blocks; subclasses implement ``forward(state, ctx)``."""

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:  # pragma: no cover - interface
        raise NotImplementedError


class InteractionBlock(Block):
    """``update(state, body_order(aggregate(message(state, ctx))))``.

    ``message(state, ctx)`` returns per-edge messages; ``aggregate(messages,
    ctx)`` reduces them to nodes (default: plain receiver sum);
    ``body_order(aggregated, state, ctx)`` raises the correlation order
    (MACE / ACE products); ``update(state, aggregated, ctx)`` returns the new
    ``BlockState``.
    """

    def __init__(
        self,
        message: nn.Module,
        update: nn.Module,
        aggregate: Optional[nn.Module] = None,
        body_order: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.message = message
        self.aggregate = aggregate if aggregate is not None else Aggregate()
        self.body_order = body_order
        self.update = update

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        messages = self.message(state, ctx)
        aggregated = self.aggregate(messages, ctx)
        if self.body_order is not None:
            aggregated = self.body_order(aggregated, state, ctx)
        return self.update(state, aggregated, ctx)


class FnBlock(Block):
    """Wrap ``fn(module, state, ctx) -> BlockState`` around an existing module.

    Lets a Core reuse an interaction module whose ``forward`` takes plain
    tensors, keeping its parameter names under ``.module``.
    """

    def __init__(self, module: nn.Module, fn: Callable[[nn.Module, BlockState, GraphContext], BlockState]) -> None:
        super().__init__()
        self.module = module
        self._fn = fn

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        return self._fn(self.module, state, ctx)
