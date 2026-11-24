"""PaiNN: equivariant message passing for tensorial properties.

Schütt, Unke, and Gastegger, ICML 2021 (arXiv:2102.03150). The Core is the
message-passing stack only. Tensor heads live in
:mod:`enerzyme.models.layers.tensorial`.
"""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
