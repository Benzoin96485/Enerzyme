"""Index scatter / segment reductions shared by every Core.

``scatter_sum`` is the single summation primitive. On CPU the native
``index_add_`` and ``torch_scatter.scatter_sum`` accumulate in the same order and
agree bitwise, so the backend is picked per call for speed only: the
``torch_scatter`` kernel is faster for small reductions, ``index_add_`` for large
ones. CUDA uses a strided index view with native ``scatter_add_`` to avoid
materializing expanded indices during force backpropagation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

try:
    import torch_scatter as _torch_scatter
except ImportError:  # pragma: no cover - optional dependency
    _torch_scatter = None

TORCH_SCATTER_MAX_NUMEL = 4 * 2**20


def batch_size_from_counts(counts) -> Optional[int]:
    """Number of structures described by the batch's per-structure atom counts."""
    if isinstance(counts, (tuple, list)) or getattr(counts, "ndim", 0) > 0:
        return len(counts)
    # A scalar can describe the total atom count across a prebatched input.
    return None


def _dim_size(index: Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return int(dim_size)
    if torch.compiler.is_compiling():
        # AOTAutograd saves the output size itself for the compiled backward.
        if statically_known_true(index.numel() == 0):
            return 0
        if not statically_known_true(index.numel() > 0):
            # A mask can select any number of entries, including none.
            index = index.long()
            index = torch.cat((index.reshape(-1), index.new_full((1,), -1)))
        return int((index.max().long() + 1).item())
    return int(index.max().item()) + 1 if index.numel() else 0


def _use_torch_scatter(src: Tensor) -> bool:
    if _torch_scatter is None or torch.compiler.is_compiling():
        return False
    return src.numel() <= TORCH_SCATTER_MAX_NUMEL


def scatter_sum(src: Tensor, index: Tensor, dim_size: Optional[int] = None, dim: int = 0) -> Tensor:
    """Sum slices of ``src`` along ``dim`` into ``dim_size`` rows given by 1-D ``index``."""
    dim = dim + src.dim() if dim < 0 else dim
    size = _dim_size(index, dim_size)
    if torch.compiler.is_compiling() and statically_known_true(index.numel() == 0):
        # Match eager's independent zero: an unused parameter must keep grad=None.
        return src.new_zeros((*src.shape[:dim], size, *src.shape[dim + 1:]))
    if index.dtype != torch.long:
        index = index.long()
    if src.device.type != "cuda" and _use_torch_scatter(src) and index.numel():
        return _torch_scatter.scatter_sum(src, index, dim=dim, dim_size=size)
    # With a single output scalar the CPU Inductor vectorizer collapses an
    # atomic-add destination to a constant index. Retain a spare slot until
    # after reduction so the generated kernel keeps its indexed writes.
    if torch.compiler.is_compiling() and src.device.type == "cpu" and src.ndim == 1:
        if dim_size is not None:
            # A supplied out-of-bounds index must not fall in the spare slot.
            index = index + (index == size)
        padded = src.new_zeros(size + 1)
        return padded.index_add_(dim, index, src).narrow(dim, 0, size)
    shape = list(src.shape)
    shape[dim] = size
    out = src.new_zeros(shape)
    if torch.compiler.is_compiling() or index.numel():
        # CPU Inductor can miscompile index_add_ across a nonleading axis in
        # channel-first messages. The expanded index keeps the destination
        # of each vector lane explicit without allocating a full index tensor.
        compiled_nonleading = dim != 0 and src.device.type == "cpu" and torch.compiler.is_compiling()
        if src.device.type == "cuda" or compiled_nonleading:
            out.scatter_add_(dim, _broadcast_index(index, src, dim), src)
        else:
            out.index_add_(dim, index, src)
    return out


def scatter_count(index: Tensor, dim_size: Optional[int] = None, dtype: torch.dtype = torch.float32) -> Tensor:
    size = _dim_size(index, dim_size)
    ones = torch.ones(index.shape[0], dtype=dtype, device=index.device)
    return scatter_sum(ones, index, size)


def scatter_mean(src: Tensor, index: Tensor, dim_size: Optional[int] = None, dim: int = 0) -> Tensor:
    """Mean over each segment; empty segments give zero."""
    dim = dim + src.dim() if dim < 0 else dim
    size = _dim_size(index, dim_size)
    total = scatter_sum(src, index, size, dim)
    count = scatter_count(index, size, src.dtype).clamp_min(1)
    view = [1] * src.dim()
    view[dim] = size
    return total / count.view(view)


def _broadcast_index(index: Tensor, src: Tensor, dim: int) -> Tensor:
    view = [1] * src.dim()
    view[dim] = -1
    return index.long().view(view).expand_as(src)


def scatter_reduce(
    src: Tensor,
    index: Tensor,
    dim_size: Optional[int] = None,
    dim: int = 0,
    reduce: str = "amax",
    fill: Optional[float] = None,
    include_self: bool = False,
) -> Tensor:
    """``amax`` / ``amin`` / ``prod`` / ``mean`` reductions via ``scatter_reduce``.

    Empty segments take ``fill`` (zero by default) when ``include_self`` is false.
    """
    dim = dim + src.dim() if dim < 0 else dim
    size = _dim_size(index, dim_size)
    shape = list(src.shape)
    shape[dim] = size
    out = src.new_full(shape, 0.0 if fill is None else fill)
    if not index.numel():
        return out
    return out.scatter_reduce(dim, _broadcast_index(index, src, dim), src, reduce=reduce, include_self=include_self)


def scatter_max(src: Tensor, index: Tensor, dim_size: Optional[int] = None, dim: int = 0) -> Tensor:
    return scatter_reduce(src, index, dim_size, dim, reduce="amax")


def scatter_min(src: Tensor, index: Tensor, dim_size: Optional[int] = None, dim: int = 0) -> Tensor:
    return scatter_reduce(src, index, dim_size, dim, reduce="amin")


def segment_sum_csr(src: Tensor, ptr: Tensor) -> Tensor:
    """Sum contiguous segments described by ``ptr`` (length ``num_segments + 1``)."""
    lengths = ptr[1:] - ptr[:-1]
    return torch.segment_reduce(src, "sum", lengths=lengths, axis=0, unsafe=True)


def gather(src: Tensor, index: Tensor) -> Tensor:
    """Row gather; ``gather`` beats fancy indexing on accelerators."""
    if src.device.type == "cpu":
        return src[index]
    flat = src.reshape(src.shape[0], -1)
    out = flat.gather(0, index.view(-1, 1).expand(-1, flat.shape[1]))
    return out.view((index.shape[0],) + tuple(src.shape[1:]))


def gather_sender(src: Tensor, index: Tensor) -> Tensor:
    """Gather sender features while keeping the two endpoint gradients independent."""
    if torch.compiler.is_compiling() and src.device.type == "cpu":
        # Inductor can fuse a row-wise receiver reduction with the sender
        # reduction, reading its zero buffer before the indexed writes.
        columns = src.movedim(0, -1).contiguous()
        return columns.index_select(-1, index).movedim(-1, 0)
    return src.index_select(0, index)
