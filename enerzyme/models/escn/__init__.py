"""Native paper eSCN architecture (Passaro & Zitnick, 2023).

Not to be confused with ``uma_qs`` (UMA wrappers in ``enerzyme.models.esen``)
or with native eSEN (``architecture: esen``, Fu et al. 2025).
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, eSCNCore

__all__ = ["eSCNCore", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"]
