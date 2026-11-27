"""FreeCG: a grouped sparse Clebsch–Gordan product on ViSNet abstract edges.

Shao, Geng, Wang, and Cui, arXiv:2407.02263. The radial basis, the first
species embedding, the cutoff window, and the scalar energy head stay in
the shared layer stack. The neighbor embedding, the edge state, and the
ViS-MP stack with the group CG product are the Core.

:class:`~enerzyme.models.freecg.cg.GroupSparseCG` is the reusable product.
Another attention stack can call it on a permutation-invariant ``[N, 8, C]``
pair without taking the rest of this Core. QuinNet's abstract edges are that
shape; this package does not change QuinNet's forward.
"""

from .cg import GroupSparseCG, shuffle_offset
from .defaults import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MD17_BUILD_PARAMS,
    PUBLISHED_MD17_CORE_PARAMS,
    PUBLISHED_QM9_BUILD_PARAMS,
    PUBLISHED_QM9_CORE_PARAMS,
)
from .interaction import ATTENTION_ENHANCERS, FreeCGInteraction, normalize_attention_enhancer

__all__ = [
    "FreeCGInteraction",
    "GroupSparseCG",
    "ATTENTION_ENHANCERS",
    "normalize_attention_enhancer",
    "shuffle_offset",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_MD17_BUILD_PARAMS",
    "PUBLISHED_MD17_CORE_PARAMS",
    "PUBLISHED_QM9_BUILD_PARAMS",
    "PUBLISHED_QM9_CORE_PARAMS",
]
