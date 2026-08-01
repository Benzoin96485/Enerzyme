"""Euclidean Fast Attention module (invariant L=0 path)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from ..so3.lebedev import lebedev_tensors, recommend_max_frequency
from .rope import frequency_init, linear_efa_aggregate


class EuclideanFastAttention(Module):
    """Linear-scaling geometry-aware nonlocal attention (scalar features).

    Contract (Enerzyme / cross-architecture)::

        features:   [N, F_in]
        positions:  [N, 3]   (absolute ``Ra``; must allow force autograd)
        batch_seg:  [N]      (molecule id)

        -> [N, F_v]   (or ``num_features_v``; default ``F_in``)

    Does **not** apply residual MLPs; see :class:`EFABlock` for the full plug-in.
    """

    def __init__(
        self,
        dim_features: int,
        *,
        num_features_qk: Optional[int] = None,
        num_features_v: Optional[int] = None,
        lebedev_num: int = 146,
        max_frequency: Optional[float] = None,
        max_length: float = 10.0,
        parametrized: bool = True,
        frequencies_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.dim_features = int(dim_features)
        self.num_features_qk = (
            self.dim_features if num_features_qk is None else int(num_features_qk)
        )
        self.num_features_v = (
            self.dim_features if num_features_v is None else int(num_features_v)
        )
        if self.num_features_qk % 2 != 0:
            raise ValueError(
                f"num_features_qk must be even, got {self.num_features_qk}"
            )
        self.lebedev_num = int(lebedev_num)
        self.max_length = float(max_length)
        if max_frequency is None:
            max_frequency = recommend_max_frequency(self.lebedev_num)
        self.max_frequency = float(max_frequency)
        self.parametrized = bool(parametrized)
        self.frequencies_trainable = bool(frequencies_trainable)

        if self.parametrized:
            self.proj_q = Linear(self.dim_features, self.num_features_qk, bias=False)
            self.proj_k = Linear(self.dim_features, self.num_features_qk, bias=False)
            self.proj_v = Linear(self.dim_features, self.num_features_v, bias=False)
        elif (
            num_features_qk is not None and num_features_qk != self.dim_features
        ) or (num_features_v is not None and num_features_v != self.dim_features):
            raise ValueError(
                "Down-projections require parametrized=True"
            )

        freqs = frequency_init(
            self.num_features_qk, self.max_frequency, self.max_length
        )
        if self.frequencies_trainable:
            self.frequencies = torch.nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs, persistent=True)

        # Cache empty; filled lazily per device/dtype.
        self.register_buffer("_grid_u", torch.empty(0), persistent=False)
        self.register_buffer("_grid_w", torch.empty(0), persistent=False)
        self._grid_ready = False

    def _ensure_grid(self, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._grid_ready
            and self._grid_u.device == device
            and self._grid_u.dtype == dtype
        ):
            return
        grid_u, grid_w = lebedev_tensors(
            self.lebedev_num, device=device, dtype=dtype
        )
        self._grid_u = grid_u
        self._grid_w = grid_w
        self._grid_ready = True

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        batch_seg: Tensor,
    ) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"EFA expects scalar features [N, F], got {tuple(features.shape)}"
            )
        n = features.shape[0]
        if positions.shape[0] != n or batch_seg.shape[0] != n:
            raise ValueError(
                "features, positions, and batch_seg must share atom dimension N"
            )
        self._ensure_grid(features.device, features.dtype)
        if self.parametrized:
            q = self.proj_q(features)
            k = self.proj_k(features)
            v = self.proj_v(features)
        else:
            q = k = v = features
        return linear_efa_aggregate(
            q,
            k,
            v,
            positions.to(dtype=features.dtype),
            self.frequencies.to(dtype=features.dtype),
            self._grid_u,
            self._grid_w,
            batch_seg,
        )
