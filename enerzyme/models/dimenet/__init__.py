"""Directional message passing: original DimeNet and DimeNet++.

``DimeNetCore`` is the bilinear SBF model (ICLR 2020). ``DimeNetPPCore`` is
the Hadamard bottleneck (arXiv:2011.14115). Package-level
``DEFAULT_BUILD_PARAMS`` / ``DEFAULT_LAYER_PARAMS`` stay the original DimeNet
stack; DimeNet++ defaults live on :mod:`enerzyme.models.dimenet.core_pp`.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, DimeNetCore
from .core_pp import DimeNetPPCore

__all__ = [
    "DimeNetCore",
    "DimeNetPPCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
