"""Native paper eSCN architecture (Passaro & Zitnick, 2023).

Not to be confused with ``enerzyme.models.esen`` (Meta UMA / eSCN-MD wrappers).
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, eSCNCore

__all__ = ["eSCNCore", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
