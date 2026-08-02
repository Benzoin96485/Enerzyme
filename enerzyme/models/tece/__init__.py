"""TECE (Tensor Edge Cluster Expansion) for Enerzyme.

Edge Cluster Expansion + Radial Rotary Attention (Xu et al., arXiv:2607.10664),
adapted from https://github.com/xvzemin/tace (MIT). Reuses TACE edge embed /
``CgtpACE`` and shared ``enerzyme.models.so3`` SO(2) primitives.
"""

from .core import TECECore, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = ["TECECore", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
