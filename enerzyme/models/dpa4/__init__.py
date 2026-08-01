"""DPA4 EMFA SO(2) descriptor for Enerzyme (arXiv:2606.02419).

Native PyTorch reimplementation. See core.py for the main Core class.
"""

from .core import DPA4Core, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = ["DPA4Core", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
