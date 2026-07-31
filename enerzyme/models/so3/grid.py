"""S² grid transforms for spherical harmonic nonlinearities.

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
``normalization`` / ``rescale_by_mmax`` options support EquiformerV2 grids.
"""

from __future__ import annotations

import math

import torch
from e3nn.o3 import FromS2Grid, ToS2Grid

from .coefficient_mapping import CoefficientMapping


class SO3_Grid(torch.nn.Module):
    """Convert between spherical harmonic coefficients and an S² grid."""

    def __init__(
        self,
        lmax: int,
        mmax: int,
        resolution: int | None = None,
        normalization: str = "integral",
        rescale_by_mmax: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.normalization = normalization
        self.rescale_by_mmax = rescale_by_mmax
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
            normalization=self.normalization,
            device=device,
        )

        self.to_grid_mat = torch.einsum(
            "mbi,am->bai", to_grid.shb, to_grid.sha
        ).detach()
        if self.rescale_by_mmax and self.lmax != self.mmax:
            for lval in range(self.lmax + 1):
                if lval <= self.mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * self.mmax + 1))
                self.to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    self.to_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        self.to_grid_mat = self.to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=self.normalization,
            device=device,
        )

        self.from_grid_mat = torch.einsum(
            "am,mbi->bai", from_grid.sha, from_grid.shb
        ).detach()
        if self.rescale_by_mmax and self.lmax != self.mmax:
            for lval in range(self.lmax + 1):
                if lval <= self.mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * self.mmax + 1))
                self.from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    self.from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
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
