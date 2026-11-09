"""One FACTO interaction block.

Scalar and tensor states are normalized together, then four atom-centered
maps update them (Liu et al., arXiv:2611.03276, Eq. 4–11):

* factorized scalar attention
* tensor-to-scalar feedback
* geometric tensor injection
* optional tensor transport

Dense channel mixing stays on atoms. Edge work is a radial weighting, an
angular contraction, and a neighbor sum. The stored state is updated with
the fixed residual scale ``α = 40^{-1/2}``; the block itself sees the
normalized copies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..functional import segment_sum_coo
from .geometry import FACTOEdgeGeometry

RESIDUAL_SCALE = 40.0 ** -0.5


def scatter_neighbors(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` onto receivers. An empty neighbor list yields zeros."""
    if index.numel() == 0:
        return src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    return segment_sum_coo(src, index, dim_size=dim_size)


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
        scalar_out = self.scalar_gain * scalars / torch.sqrt(variance + self.epsilon)
        tensor_out: List[Tensor] = []
        for features, gain in zip(tensors, self.tensor_gain):
            # features: [N, 2ℓ+1, C]; ω is the mean square over m.
            angular = features.square().mean(dim=1)
            denom = torch.sqrt(variance + angular + self.epsilon)
            tensor_out.append(gain * features / denom.unsqueeze(1))
        return scalar_out, tensor_out


class FactorizedScalarAttention(nn.Module):
    """Group-wise scalar attention with the edge sum pulled outside the gates.

    Queries, keys, and values are atom-wise. The key–value product ``s_j`` and
    the query gate are applied once per atom (Eq. 5), so each edge only
    multiplies ``s_j`` by the precomputed radial filter and sums onto ``i``.
    """

    def __init__(self, scalar_dim: int, num_groups: int) -> None:
        super().__init__()
        if num_groups < 1 or scalar_dim % num_groups != 0:
            raise ValueError(
                f"scalar_dim={scalar_dim} must be divisible by num_groups={num_groups}"
            )
        self.num_groups = int(num_groups)
        self.group_dim = scalar_dim // num_groups
        # One fused query map is the concatenation of the per-group maps W^g_q.
        self.query = nn.Linear(scalar_dim, scalar_dim, bias=True)
        self.key = nn.Linear(scalar_dim, self.group_dim, bias=True)
        self.value = nn.Linear(scalar_dim, self.group_dim, bias=True)
        self.mix = nn.Linear(scalar_dim, scalar_dim, bias=True)

    def forward(self, scalars: Tensor, rho: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        n_atom = scalars.shape[0]
        query = self.query(scalars).view(n_atom, self.num_groups, self.group_dim)
        source = F.silu(self.key(scalars)) * self.value(scalars)
        if idx_i.numel() == 0:
            aggregated = scalars.new_zeros((n_atom, self.num_groups, self.group_dim))
        else:
            radial = rho.view(-1, self.num_groups, self.group_dim)
            messages = radial * source[idx_j].unsqueeze(1)
            aggregated = scatter_neighbors(messages, idx_i, n_atom)
        mixed = (F.silu(query) * aggregated).reshape(n_atom, -1)
        return F.silu(self.mix(mixed))


class TensorToScalarFeedback(nn.Module):
    """Rotationally invariant tensor dots, used as a gate on the scalar update.

    Channel maps ``U`` and ``V`` are bias-free: a bias would not transform as
    a tensor. The contraction over ``m`` (Eq. 6) therefore stays zero when the
    tensor state is zero.
    """

    def __init__(self, scalar_dim: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.left = nn.ModuleList(
            [nn.Linear(int(width), int(width), bias=False) for width in channels]
        )
        self.right = nn.ModuleList(
            [nn.Linear(int(width), int(width), bias=False) for width in channels]
        )
        self.mix = nn.Linear(sum(int(width) for width in channels), scalar_dim, bias=True)
        self.to_gate = nn.Linear(scalar_dim, scalar_dim, bias=True)
        self.to_residual = nn.Linear(scalar_dim, scalar_dim, bias=True)

    def forward(self, tensors: Sequence[Tensor], scalar_intermediate: Tensor) -> Tensor:
        invariants = []
        for features, left, right in zip(tensors, self.left, self.right):
            projected_left = left(features)
            projected_right = right(features)
            invariants.append((projected_left * projected_right).sum(dim=1))
        gate = self.mix(torch.cat(invariants, dim=-1))
        return gate * self.to_gate(scalar_intermediate) + self.to_residual(scalar_intermediate)


class GeometricTensorInjection(nn.Module):
    """Create degree-``ℓ`` features from the scalar intermediate and ``Y_ℓm``.

    ``p_j`` is an atom-wise map of the scalar intermediate. The edge only
    multiplies by the precomputed filter ``ξ`` and ``Y_ℓm`` (Eq. 7).
    """

    def __init__(self, scalar_dim: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.project = nn.ModuleList(
            [nn.Linear(scalar_dim, int(width), bias=True) for width in channels]
        )

    def forward(
        self,
        scalar_intermediate: Tensor,
        xi: Sequence[Tensor],
        harmonics: Sequence[Tensor],
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> List[Tensor]:
        n_atom = scalar_intermediate.shape[0]
        injected: List[Tensor] = []
        for project, radial, angular in zip(self.project, xi, harmonics):
            source = project(scalar_intermediate)
            if idx_i.numel() == 0:
                injected.append(source.new_zeros((n_atom, angular.shape[-1], source.shape[-1])))
                continue
            message = source[idx_j] * radial
            weighted = message.unsqueeze(1) * angular.unsqueeze(-1)
            injected.append(scatter_neighbors(weighted, idx_i, n_atom))
        return injected


class TensorTransport(nn.Module):
    """Move existing tensor features with an invariant alignment gate.

    Alignment and the transported message are projected on atoms (Eq. 9–10).
    The edge contracts the projected tensor with ``Y_ℓm``, applies the scalar
    gate and the pair window, and sums onto the receiver. Alignment, message,
    and the tensor state are bias-free so the map stays equivariant.
    """

    def __init__(self, scalar_dim: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.align = nn.ModuleList(
            [nn.Linear(int(width), int(width), bias=False) for width in channels]
        )
        self.message = nn.ModuleList(
            [nn.Linear(int(width), int(width), bias=False) for width in channels]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(scalar_dim, int(width), bias=True) for width in channels]
        )

    def forward(
        self,
        tensors: Sequence[Tensor],
        scalar_intermediate: Tensor,
        chi: Tensor,
        harmonics: Sequence[Tensor],
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> List[Tensor]:
        n_atom = scalar_intermediate.shape[0]
        transported: List[Tensor] = []
        for features, align, message, gate, angular in zip(
            tensors, self.align, self.message, self.gate, harmonics
        ):
            if idx_i.numel() == 0:
                transported.append(
                    features.new_zeros((n_atom, features.shape[1], features.shape[2]))
                )
                continue
            aligned = align(features)
            # Σ_m A_{j,m} Y_m is invariant and has one value per channel.
            score = (aligned[idx_j] * angular.unsqueeze(-1)).sum(dim=1)
            payload = message(features)
            gated = chi.unsqueeze(-1) * gate(scalar_intermediate)[idx_j] * score
            weighted = gated.unsqueeze(1) * payload[idx_j]
            transported.append(scatter_neighbors(weighted, idx_i, n_atom))
        return transported


class FACTOInteraction(nn.Module):
    """Coupled scalar/tensor update. The last block skips the tensor increment."""

    def __init__(
        self,
        scalar_dim: int,
        channels: Sequence[int],
        num_groups: int,
        *,
        update_tensors: bool,
        tensor_transport: bool,
        residual_scale: float = RESIDUAL_SCALE,
        rms_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.update_tensors = bool(update_tensors)
        self.residual_scale = float(residual_scale)
        self.norm = CoupledRMSNorm(scalar_dim, channels, epsilon=rms_epsilon)
        self.attention = FactorizedScalarAttention(scalar_dim, num_groups)
        self.feedback = TensorToScalarFeedback(scalar_dim, channels)
        self.injection: Optional[GeometricTensorInjection]
        self.transport: Optional[TensorTransport]
        if self.update_tensors:
            self.injection = GeometricTensorInjection(scalar_dim, channels)
            self.transport = (
                TensorTransport(scalar_dim, channels) if tensor_transport else None
            )
        else:
            self.injection = None
            self.transport = None

    def forward(
        self,
        scalars: Tensor,
        tensors: Sequence[Tensor],
        geometry: FACTOEdgeGeometry,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        normalized_scalars, normalized_tensors = self.norm(scalars, tensors)
        intermediate = self.attention(normalized_scalars, geometry.rho, idx_i, idx_j)
        delta = self.feedback(normalized_tensors, intermediate)
        scalars_out = scalars + self.residual_scale * delta
        if not self.update_tensors or self.injection is None:
            return scalars_out, list(tensors)
        injected = self.injection(
            intermediate, geometry.xi, geometry.harmonics, idx_i, idx_j
        )
        if self.transport is None:
            transported = [torch.zeros_like(term) for term in injected]
        else:
            transported = self.transport(
                normalized_tensors,
                intermediate,
                geometry.chi,
                geometry.harmonics,
                idx_i,
                idx_j,
            )
        tensors_out = [
            stored + self.residual_scale * (new + carried)
            for stored, new, carried in zip(tensors, injected, transported)
        ]
        return scalars_out, tensors_out
