"""Sparse E2Former attention backed by the shared native/Triton operations."""

from ...mp.equivariant.kernels import (
    make_sparse_v_kernel,
    sparse_qk,
    sparse_qk_pytorch,
    sparse_v_agg,
    sparse_v_agg_pytorch,
    triton_kernels_available,
)

__all__ = [
    "make_sparse_v_kernel", "sparse_qk", "sparse_qk_pytorch", "sparse_v_agg",
    "sparse_v_agg_pytorch", "triton_kernels_available",
]
