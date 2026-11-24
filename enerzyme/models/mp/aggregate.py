"""Receiver-wise aggregation of scalar, vector, and spherical messages."""

from typing import Optional

from torch import Tensor, nn

from .graph import GraphContext
from .ops import scatter_count, scatter_sum


class Aggregate(nn.Module):
    """Sum, mean, fixed rescaling, or normalization by learned edge density."""

    def __init__(self, divisor: Optional[float] = None, mode: str = "sum") -> None:
        super().__init__()
        if mode not in ("sum", "mean", "density"):
            raise ValueError(f"unknown aggregation mode {mode!r}")
        self.divisor = None if divisor is None else float(divisor)
        self.mode = mode

    def normalizer(
        self,
        index: Tensor,
        num_nodes: int,
        message: Tensor,
        density: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self.mode == "density":
            if density is None:
                raise ValueError("density aggregation requires one density per edge")
            if density.shape != index.shape and density.shape != (*index.shape, 1):
                raise ValueError("density must have shape [edges] or [edges, 1]")
            return scatter_sum(density.reshape(-1), index, num_nodes) + 1
        if self.mode == "mean":
            return scatter_count(index, num_nodes, message.dtype).clamp_min(1)
        return None

    def reduce(
        self,
        message: Tensor,
        index: Tensor,
        num_nodes: int,
        density: Optional[Tensor] = None,
    ) -> Tensor:
        norm = self.normalizer(index, num_nodes, message, density)
        return self.reduce_with_normalizer(message, index, num_nodes, norm)

    def reduce_with_normalizer(
        self, message: Tensor, index: Tensor, num_nodes: int, norm: Optional[Tensor]
    ) -> Tensor:
        """Reuse the same receiver density for multiple feature streams."""
        out = scatter_sum(message, index, num_nodes)
        if norm is not None:
            out = out / norm.view((num_nodes,) + (1,) * (out.ndim - 1))
        if self.divisor is not None:
            out = out / self.divisor
        return out

    def forward(
        self,
        message: Tensor,
        ctx: GraphContext,
        index: Optional[Tensor] = None,
        density: Optional[Tensor] = None,
    ) -> Tensor:
        index = ctx.receiver if index is None else index
        return self.reduce(message, index, ctx.num_nodes, density)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, divisor={self.divisor}"
