"""TACE (Tensor Atomic Cluster Expansion) for Enerzyme.

Supports spherical (e3nn CGTP) and Cartesian (ICT / Cartesian-3j) backends via
``tensor_basis``. TECE is not included.
"""

from .core import TACECore, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = ["TACECore", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
