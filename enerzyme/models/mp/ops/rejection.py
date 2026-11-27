"""Vector rejection shared by Geodite, ViSNet, and QuinNet.

``features`` is ``[E, M, C]``. ``direction`` is ``[E, M]`` or, for QuinNet's
node-level improper and five-body terms, the same shape as ``features``.
The direction is not re-normalized and is not divided by its squared norm.
For ``l = 1`` a vector ``[E, M]`` is a unit vector. ViSNet and QuinNet use
this projection on the polynomial spherical channels. A channel-matched
direction rejects each feature channel against its own angular vector.
"""

from __future__ import annotations

from torch import Tensor


def vector_rejection(features: Tensor, direction: Tensor) -> Tensor:
    """Component of ``features`` orthogonal to ``direction``."""
    if direction.ndim == features.ndim - 1:
        if direction.shape != features.shape[:-1]:
            raise ValueError(
                "direction must have shape "
                f"{tuple(features.shape[:-1])} or {tuple(features.shape)}, "
                f"got {tuple(direction.shape)}"
            )
        direction = direction.unsqueeze(-1)
    elif direction.shape != features.shape:
        raise ValueError(
            "direction must have shape "
            f"{tuple(features.shape[:-1])} or {tuple(features.shape)}, "
            f"got {tuple(direction.shape)}"
        )
    projected = (features * direction).sum(dim=1, keepdim=True)
    return features - projected * direction
