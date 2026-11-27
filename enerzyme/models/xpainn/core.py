"""Public constructor for the optional XequiNet preset."""
from ..presets.xpainn import xpainn as XPaiNNWrapper
from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = ["XPaiNNWrapper", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
