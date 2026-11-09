"""Shared FACTO edge geometry: Gaussian basis, pair window, spherical harmonics.

Radial filters and real spherical harmonics are computed once per forward and
reused by every interaction block (Liu et al., arXiv:2611.03276, Eq. 1–3).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..cutoff import facto_pair_window
from ..so3.spherical_harmonics import spherical_harmonics

# Paper: s_r = 2.5 / (N_r - 1), centers μ_n = (n - 1) / (N_r - 1).
GAUSSIAN_WIDTH_NUMERATOR = 2.5


class FACTOEdgeGeometry(NamedTuple):
    """Edge tensors consumed by every FACTO interaction block."""

    chi: Tensor
    rho: Tensor
    xi: Tuple[Tensor, ...]
    harmonics: Tuple[Tensor, ...]


class FACTOGeometry(nn.Module):
    """Distance smearing, scalar/tensor filters, and per-degree real harmonics.

    ``φ = χ u`` already includes the pair window. The scalar filter is
    ``ρ = χ SiLU(W_ρ φ)`` so a bias in ``W_ρ`` cannot leak past the cutoff.
    Tensor filters ``ξ^ℓ = W^ℓ_ξ φ`` are bias-free, so they vanish with ``φ``.
    """

    def __init__(
        self,
        scalar_dim: int,
        n_radial: int,
        channels: Sequence[int],
        cutoff: float,
        r_short: float,
    ) -> None:
        super().__init__()
        if n_radial < 2:
            raise ValueError(f"num_radial must be >= 2, got {n_radial}")
        self.scalar_dim = int(scalar_dim)
        self.n_radial = int(n_radial)
        self.channels = tuple(int(c) for c in channels)
        self.cutoff = float(cutoff)
        self.r_short = float(r_short)
        self.lmax = len(self.channels)
        self.width = GAUSSIAN_WIDTH_NUMERATOR / float(self.n_radial - 1)
        centers = torch.arange(self.n_radial, dtype=torch.float32) / float(
            self.n_radial - 1
        )
        self.register_buffer("centers", centers, persistent=False)
        self.scalar_filter = nn.Linear(self.n_radial, self.scalar_dim, bias=True)
        self.tensor_filters = nn.ModuleList(
            [nn.Linear(self.n_radial, channels, bias=False) for channels in self.channels]
        )
        self.degrees = tuple(range(1, self.lmax + 1))

    def pair_window(self, distance: Tensor) -> Tensor:
        return facto_pair_window(distance, self.cutoff, self.r_short)

    def gaussian_basis(self, distance: Tensor) -> Tensor:
        """``u_n = exp(-(r/Rc - μ_n)^2 / (2 s_r^2))``, shape ``(E, N_r)``."""
        scaled = distance.unsqueeze(-1) / self.cutoff
        centers = self.centers.to(dtype=distance.dtype, device=distance.device)
        delta = scaled - centers
        return torch.exp(-0.5 * (delta / self.width) ** 2)

    def forward(self, distance: Tensor, displacement: Tensor) -> FACTOEdgeGeometry:
        """Build filters for one neighbor list.

        Parameters
        ----------
        distance:
            Pair distances ``(E,)``.
        displacement:
            Vectors from receiver ``i`` to sender ``j``, ``(E, 3)``.
            This is Enerzyme's ``vij = R_j - R_i``.
        """
        if distance.ndim != 1:
            raise ValueError(f"distance must be rank 1, got {tuple(distance.shape)}")
        if displacement.shape != (distance.shape[0], 3):
            raise ValueError(
                "displacement must be (E, 3) aligned with distance, "
                f"got {tuple(displacement.shape)} vs E={distance.shape[0]}"
            )
        chi = self.pair_window(distance)
        basis = self.gaussian_basis(distance)
        phi = chi.unsqueeze(-1) * basis
        rho = chi.unsqueeze(-1) * F.silu(self.scalar_filter(phi))
        xi = tuple(layer(phi) for layer in self.tensor_filters)
        harmonics = self._harmonics(displacement)
        return FACTOEdgeGeometry(chi=chi, rho=rho, xi=xi, harmonics=harmonics)

    def _harmonics(self, displacement: Tensor) -> Tuple[Tensor, ...]:
        n_edge = displacement.shape[0]
        widths = [2 * degree + 1 for degree in self.degrees]
        if n_edge == 0:
            return tuple(
                displacement.new_zeros((0, width)) for width in widths
            )
        length = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        # A zero-length image has an undefined direction. Those pairs are
        # excluded by the paper; if one appears, χ is 0 and the direction is
        # only needed to keep spherical harmonics finite.
        direction = displacement / length.clamp_min(1e-12)
        fallback = displacement.new_zeros((n_edge, 3))
        fallback[:, 2] = 1.0
        direction = torch.where(length > 0, direction, fallback)
        packed = spherical_harmonics(
            direction,
            self.degrees,
            layout="e3nn",
            normalization="integral",
            normalize_input=False,
        )
        return tuple(torch.split(packed, widths, dim=-1))
