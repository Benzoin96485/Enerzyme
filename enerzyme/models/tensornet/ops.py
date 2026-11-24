"""TensorNet channel-last Cartesian algebra."""

from ..layers.self_pairs import unit_direction as unit_vectors
from ..mp.equivariant.cartesian import (
    identity_tensor,
    decompose_tensor,
    compose_tensor,
    frobenius_squared,
    scalar_invariants,
    vector_to_skew_tensor,
    skew_tensor_to_vector,
    symmetric_traceless,
    tensor_matmul,
    tensor_anticommutator,
)

__all__ = [
    "identity_tensor",
    "decompose_tensor",
    "compose_tensor",
    "frobenius_squared",
    "scalar_invariants",
    "vector_to_skew_tensor",
    "skew_tensor_to_vector",
    "symmetric_traceless",
    "tensor_matmul",
    "tensor_anticommutator",
    "unit_vectors",
]
