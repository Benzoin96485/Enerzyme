"""Per-edge attention scores for receiver queries and sender keys."""

from typing import Callable, Optional

from torch import Tensor


def dot_product_attention_logits(
    query: Tensor, key: Tensor, edge_filter: Optional[Tensor] = None, *,
    scale: Optional[float] = None, activation: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    product = query * key
    if edge_filter is not None:
        product = product * edge_filter
    logits = product.sum(dim=-1)
    if scale is not None:
        logits = logits * scale
    return logits if activation is None else activation(logits)
