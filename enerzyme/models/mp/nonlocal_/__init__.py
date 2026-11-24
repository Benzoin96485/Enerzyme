"""Long-range attention for entire molecules and atom-fragment graphs."""

from .performer import Performer
from .fragment import FragmentLR
from .global_attention import GlobalAttention
from .attention import EuclideanFastAttention
from .block import EFABlock, parse_era_iterations
from .hook import apply_efa_if_configured, build_efa_blocks, efa_input_fields
from ..equivariant.s2.lebedev import (
    LEBEDEV_FREQUENCY_LOOKUP,
    available_lebedev_nums,
    lebedev_quadrature,
    lebedev_tensors,
    recommend_max_frequency,
)
from .rope import (
    apply_rotary_position_embedding,
    calculate_rotary_position_embedding,
    frequency_init,
    linear_efa_aggregate,
)

__all__ = [
    "Performer",
    "FragmentLR",
    "GlobalAttention",
    "EuclideanFastAttention",
    "EFABlock",
    "LEBEDEV_FREQUENCY_LOOKUP",
    "apply_efa_if_configured",
    "apply_rotary_position_embedding",
    "available_lebedev_nums",
    "build_efa_blocks",
    "calculate_rotary_position_embedding",
    "efa_input_fields",
    "frequency_init",
    "lebedev_quadrature",
    "lebedev_tensors",
    "linear_efa_aggregate",
    "parse_era_iterations",
    "recommend_max_frequency",
]
