"""S² grid transforms for spherical harmonic nonlinearities.

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
"""

from __future__ import annotations

import torch
from e3nn.o3 import FromS2Grid, ToS2Grid

from .coefficient_mapping import CoefficientMapping


class SO3_Grid(torch.nn.Module):
    """Convert between spherical harmonic coefficients and an S² grid."""

    def __init__(self, lmax: int, mmax: int, resolution: int | None = None) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution:
            self.long_resolution = resolution
            self.lat_resolution = resolution

        self.initialized = False

    def _initialize(self, device: torch.device) -> None:
        if self.initialized is True:
            return
        self.mapping = CoefficientMapping([self.lmax], [self.lmax], device)

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization="integral",
            device=device,
        )

        self.to_grid_mat = torch.einsum(
            "mbi,am->bai", to_grid.shb, to_grid.sha
        ).detach()
        self.to_grid_mat = self.to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization="integral",
            device=device,
        )

        self.from_grid_mat = torch.einsum(
            "am,mbi->bai", from_grid.sha, from_grid.shb
        ).detach()
        self.from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        self.initialized = True

    def get_to_grid_mat(self, device: torch.device):
        self._initialize(device)
        return self.to_grid_mat

    def get_from_grid_mat(self, device: torch.device):
        self._initialize(device)
        return self.from_grid_mat

    def to_grid(self, embedding: torch.Tensor, lmax: int, mmax: int) -> torch.Tensor:
        self._initialize(embedding.device)
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai,zic->zbac", to_grid_mat, embedding)

    def from_grid(self, grid: torch.Tensor, lmax: int, mmax: int) -> torch.Tensor:
        self._initialize(grid.device)
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return torch.einsum("bai,zbac->zic", from_grid_mat, grid)
