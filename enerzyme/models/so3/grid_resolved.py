"""S² grid with asymmetric resolution / m-primary layout (EquiformerV3).

Distinct from ``SO3_Grid`` (eSCN / EquiformerV2), which uses a single
resolution and lazy init.
"""

from __future__ import annotations

import copy
import math

import torch
from e3nn.o3 import FromS2Grid, ToS2Grid

from .rotation_fused import CoefficientMappingModule


class SO3GridResolved(torch.nn.Module):
    """Project between SH coeffs and an S² grid (EquiformerV3 API)."""

    def __init__(
        self,
        lmax,
        mmax,
        normalization="component",
        resolution_list=None,
        use_m_primary=False,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=False
        )
        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
            device="cpu",
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
            device="cpu",
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        from_grid_mat = from_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.permute(1, 0)

        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum("ai, ji -> aj", to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum("ia, ji -> ja", from_grid_mat, temp.to_m)

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat

    def to_grid(self, embedding):
        return torch.einsum("aj, njc -> nac", self.to_grid_mat, embedding)

    def from_grid(self, grid):
        return torch.einsum("ja, nac -> njc", self.from_grid_mat, grid)

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, "
            f"lat_resolution={self.lat_resolution}, "
            f"long_resolution={self.long_resolution}, "
            f"use_m_primary={self.use_m_primary}"
        )


# Upstream alias
SO3Grid = SO3GridResolved
