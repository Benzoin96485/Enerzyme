"""Small tensor helpers for GRACE equivariant blocks.

Features are lists indexed by angular momentum ``l``. Entry ``l`` has shape
``[N, channels, 2l+1]`` in the e3nn real-spherical-harmonic ``m`` order
(``m = -l ... l``, channel-major). That is the layout produced by
:func:`enerzyme.models.so3.spherical_harmonics.spherical_harmonics` with
``layout="e3nn"`` and by :func:`e3nn.o3.wigner_3j`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor


def rms_norm(features: Tensor, epsilon: float) -> Tensor:
    """Per-atom RMS over channels and ``m``. Zeros stay zero."""
    moment = features.square().mean(dim=(1, 2), keepdim=True)
    return features * torch.rsqrt(moment + epsilon)


def scale_features(blocks: Sequence[Tensor], factor: Tensor) -> List[Tensor]:
    scale = factor.to(dtype=blocks[0].dtype, device=blocks[0].device)
    while scale.ndim < blocks[0].ndim:
        scale = scale.view(*scale.shape, *([1] * (blocks[0].ndim - scale.ndim)))
    return [block * scale for block in blocks]


def truncate_features(blocks: Sequence[Tensor], lmax: int) -> List[Tensor]:
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    if lmax + 1 > len(blocks):
        raise ValueError(
            f"cannot keep l up to {lmax} from a feature with lmax {len(blocks) - 1}"
        )
    return list(blocks[: lmax + 1])


def scatter_blocks(
    blocks: Sequence[Tensor], index: Tensor, n_atoms: int
) -> List[Tensor]:
    """Sum edge features onto receiver atoms. ``index`` has one entry per edge."""
    scattered: List[Tensor] = []
    for block in blocks:
        acc = block.new_zeros((n_atoms, block.shape[1], block.shape[2]))
        if block.shape[0] > 0:
            acc = acc.index_add(0, index, block)
        scattered.append(acc)
    return scattered


def split_spherical_harmonics(harmonics: Tensor, lmax: int) -> List[Tensor]:
    """Split a packed ``[E, (lmax+1)^2]`` e3nn harmonic into per-degree blocks."""
    expected = (lmax + 1) ** 2
    if harmonics.shape[-1] != expected:
        raise ValueError(
            f"spherical harmonics width {harmonics.shape[-1]} != {(lmax + 1) ** 2}"
        )
    blocks: List[Tensor] = []
    offset = 0
    for degree in range(lmax + 1):
        width = 2 * degree + 1
        blocks.append(harmonics[:, offset : offset + width])
        offset += width
    return blocks


def pack_features(blocks: Sequence[Tensor]) -> Tensor:
    """Pack ``[N, C, 2l+1]`` blocks into the e3nn channel-major vector."""
    return torch.cat([block.reshape(block.shape[0], -1) for block in blocks], dim=-1)


def zeros_like_degrees(
    reference: Tensor, n_atoms: int, channels: int, lmax: int
) -> List[Tensor]:
    return [
        reference.new_zeros((n_atoms, channels, 2 * degree + 1))
        for degree in range(lmax + 1)
    ]


def natural_parity(degree: int) -> int:
    """Parity of a polar spherical harmonic, ``(-1)^l``."""
    return 1 if degree % 2 == 0 else -1


def cg_scale(degree: int) -> float:
    """Component-normalization factor used with Wigner-3j contractions."""
    return math.sqrt(2 * degree + 1)


def clip_lmax(lmax: int, limit: Optional[int]) -> int:
    if limit is None:
        return int(lmax)
    return min(int(lmax), int(limit))
