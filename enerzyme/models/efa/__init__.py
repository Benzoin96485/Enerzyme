"""Euclidean Fast Attention (Frank et al., Nat. Mach. Intell. / arXiv:2412.08541).

Architecture-agnostic nonlocal plug-in for Enerzyme Cores:

* **SpookyNet** — optional replacement for geometry-free ``NonlocalInteraction``
  via ``use_efa=True``.
* **So3krates / SO3LR** — optional layers via ``era_use_in_iterations``;
  architectures ``efa`` / ``so3lr_efa``.
* **Other Cores** — call :class:`EFABlock` or :func:`apply_efa_if_configured`
  on invariant ``[N, F]`` features with absolute ``Ra`` and ``batch_seg``.

Lebedev grids are vendored from Google e3x (Apache-2.0); see
``lebedev_grids.npz`` and ``NOTICE``.
"""

from .attention import EuclideanFastAttention
from .block import EFABlock, parse_era_iterations
from .hook import apply_efa_if_configured, build_efa_blocks, efa_input_fields
from .lebedev import (
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
