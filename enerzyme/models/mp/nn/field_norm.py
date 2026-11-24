"""Equivariant norms for split scalar, vector, and higher-degree fields."""

from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .norm import balanced_degree_moment, degree_index, inverse_rms

VECNORM_TYPES = ("none", "rms", "max_min")


def normalize_vecnorm_type(value: object) -> str:
    """Return ``none``, ``rms``, or ``max_min``."""
    key = str(value).strip().lower().replace("-", "_")
    if key not in VECNORM_TYPES:
        choices = ", ".join(VECNORM_TYPES)
        raise ValueError(f"vecnorm_type must be one of {choices}, got {value!r}")
    return key


class VecLayerNorm(nn.Module):
    """Equivariant vector norm from ``visnet.models.utils.VecLayerNorm``.

    ``vec`` is ``[N, 3, F]`` or ``[N, 8, F]``. ``l = 1`` and ``l = 2`` are
    normalized separately. ``none`` leaves the direction unchanged and still
    applies the per-channel weight. A non-trainable weight is a buffer of ones.
    """

    def __init__(self, hidden_channels: int, trainable: bool, norm_type: str) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.norm_type = normalize_vecnorm_type(norm_type)
        self.trainable = bool(trainable)
        self.eps = 1e-12
        weight = torch.ones(self.hidden_channels)
        if self.trainable:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)
        if self.norm_type == "rms":
            self._norm = self.rms_norm
        elif self.norm_type == "max_min":
            self._norm = self.max_min_norm
        else:
            self._norm = self.none_norm
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.copy_(torch.ones(self.hidden_channels))

    def none_norm(self, vec: Tensor) -> Tensor:
        return vec

    def rms_norm(self, vec: Tensor) -> Tensor:
        dist = torch.linalg.vector_norm(vec, dim=1)
        if bool((dist == 0).all()):
            return torch.zeros_like(vec)
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist.square(), dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.linalg.vector_norm(vec, dim=1, keepdim=True)
        if bool((dist == 0).all()):
            return torch.zeros_like(vec)
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).reshape(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.reshape(-1, 1, 1)) / delta.reshape(-1, 1, 1)
        return F.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        if vec.shape[1] == 3:
            normalized = self._norm(vec)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            normalized = torch.cat((self._norm(vec1), self._norm(vec2)), dim=1)
        else:
            raise ValueError(
                "VecLayerNorm expects 3 or 8 angular channels, " f"got {vec.shape[1]}"
            )
        return normalized * self.weight.view(1, 1, -1)


class CoupledRMSNorm(nn.Module):
    """Per-channel RMS norm that keeps a finite tensor scale when ``T = 0``.

    The scalar mean square ``ν`` is added to every tensor-channel denominator
    (Eq. 4), so a symmetric environment with vanishing angular features still
    has a normalization scale.
    """

    def __init__(
        self,
        scalar_dim: int,
        channels: Sequence[int],
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"rms epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.scalar_gain = nn.Parameter(torch.ones(scalar_dim))
        self.tensor_gain = nn.ParameterList(
            [nn.Parameter(torch.ones(int(width))) for width in channels]
        )

    def forward(
        self, scalars: Tensor, tensors: Sequence[Tensor]
    ) -> Tuple[Tensor, List[Tensor]]:
        variance = scalars.square().mean(dim=-1, keepdim=True)
        scalar_out = self.scalar_gain * scalars * inverse_rms(variance, self.epsilon)
        tensor_out: List[Tensor] = []
        for features, gain in zip(tensors, self.tensor_gain):
            # features: [N, 2ℓ+1, C]; ω is the mean square over m.
            angular = features.square().mean(dim=1)
            scale = inverse_rms(variance + angular, self.epsilon)
            tensor_out.append(gain * features * scale.unsqueeze(1))
        return scalar_out, tensor_out


class TensorLayerNorm(nn.Module):
    """One RMS over every degree, with a per-degree affine gain.

    The degree weight ``1 / (2l + 1)`` gives each ``l`` the same vote. The
    statistic is then averaged over channels, so a symmetric geometry where
    one degree is zero does not make its own norm singular.
    """

    def __init__(self, lmax: int, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.channels = int(num_channels)
        self.split_sizes = [2 * l + 1 for l in range(self.lmax + 1)]
        self.register_buffer("degree_index", degree_index(self.lmax), persistent=False)
        balance = [1.0 / width for width in self.split_sizes for _ in range(width)]
        self.register_buffer(
            "balance", torch.tensor(balance).view(1, -1, 1), persistent=False
        )
        self.affine = nn.Parameter(torch.ones(self.lmax + 1, self.channels))

    def forward(self, scalars: Tensor, steerable: Tensor) -> tuple[Tensor, Tensor]:
        packed = torch.cat([scalars.unsqueeze(1), steerable], dim=1)
        scale = inverse_rms(
            balanced_degree_moment(packed, self.balance, False), self.eps
        )
        gain = self.affine.index_select(0, self.degree_index).unsqueeze(0)
        normalized = packed * scale * gain
        return normalized[:, 0, :], normalized[:, 1:, :]
