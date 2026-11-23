"""Vector rejection shared by Geodite and ViSNet.

``features`` is ``[E, M, C]`` and ``direction`` is ``[E, M]``. The direction
is not re-normalized. For ``l = 1`` it is a unit vector. ViSNet's runtime
geometry uses the same projection on the polynomial spherical channels.
"""

from __future__ import annotations

from torch import Tensor


def vector_rejection(features: Tensor, direction: Tensor) -> Tensor:
    """Component of ``features`` orthogonal to ``direction``."""
    direction = direction.unsqueeze(-1)
    projected = (features * direction).sum(dim=1, keepdim=True)
    return features - projected * direction
