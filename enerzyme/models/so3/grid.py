"""Unified e3nn lat–long S² grid projector (flat layout).

Replaces the former ``SO3_Grid`` (eSCN / EquiformerV2, lazy ``[B,A,I]``) and
``SO3GridResolved`` (EquiformerV3, eager flat) with one class that always
registers flat buffers ``(G, D)`` / ``(D, G)`` and implements
:class:`~enerzyme.models.so3.s2_projector.S2GridProjector`.
"""

from __future__ import annotations

import copy
import math

import torch
from e3nn.o3 import FromS2Grid, ToS2Grid
from torch import Tensor, nn

from .coefficient_mapping import CoefficientMapping
from .rotation_fused import CoefficientMappingModule


class SO3Grid(nn.Module):
    """Project between SH coeffs and an e3nn lat–long S² grid (flat).

    Args:
        lmax: Maximum spherical degree.
        mmax: Maximum order retained after coefficient masking.
        normalization: e3nn ``ToS2Grid`` / ``FromS2Grid`` normalization.
        resolution: If set, isotropic ``lat = long = resolution``.
        resolution_list: ``[lat, long]``; wins over ``resolution``.
        rescale_by_mmax: Rescale high-``l`` columns when ``lmax != mmax``.
            ``None`` means enable whenever ``lmax != mmax``.
        use_m_primary: Remap coeff axis with EquiformerV3 ``to_m``.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        *,
        normalization: str = "component",
        resolution: int | None = None,
        resolution_list: list[int] | None = None,
        rescale_by_mmax: bool | None = None,
        use_m_primary: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.normalization = normalization
        self.use_m_primary = bool(use_m_primary)
        if rescale_by_mmax is None:
            rescale_by_mmax = self.lmax != self.mmax
        self.rescale_by_mmax = bool(rescale_by_mmax)

        self.lat_resolution = 2 * (self.lmax + 1)
        if self.lmax == self.mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * self.mmax + 1
        if resolution is not None:
            self.lat_resolution = int(resolution)
            self.long_resolution = int(resolution)
        if resolution_list is not None:
            resolution_list = copy.deepcopy(resolution_list)
            if len(resolution_list) != 2:
                raise ValueError(
                    f"resolution_list must be [lat, long], got {resolution_list}"
                )
            self.lat_resolution = int(resolution_list[0])
            self.long_resolution = int(resolution_list[1])

        # Used by multi-resolution embedding.to_grid / _from_grid coeff slicing.
        self.mapping = CoefficientMapping(
            [self.lmax], [self.lmax], torch.device("cpu")
        )

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=False
        )
        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
            device="cpu",
        )
        to_grid_mat = torch.einsum(
            "mbi,am->bai", to_grid.shb, to_grid.sha
        ).detach()
        if self.rescale_by_mmax and self.lmax != self.mmax:
            for lval in range(self.lmax + 1):
                if lval <= self.mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * self.mmax + 1))
                to_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor
        to_grid_mat = to_grid_mat[
            :, :, mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
            device="cpu",
        )
        from_grid_mat = torch.einsum(
            "am,mbi->bai", from_grid.sha, from_grid.shb
        ).detach()
        if self.rescale_by_mmax and self.lmax != self.mmax:
            for lval in range(self.lmax + 1):
                if lval <= self.mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * self.mmax + 1))
                from_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor
        from_grid_mat = from_grid_mat[
            :, :, mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1).permute(1, 0)

        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum("ai,ji->aj", to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum("ia,ji->ja", from_grid_mat, temp.to_m)

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)
        self.grid_size = int(to_grid_mat.shape[0])
        self.n_coeff = int(to_grid_mat.shape[1])

    def get_to_grid_mat(self, device: torch.device | None = None) -> Tensor:
        mat = self.to_grid_mat
        return mat if device is None else mat.to(device=device)

    def get_from_grid_mat(self, device: torch.device | None = None) -> Tensor:
        mat = self.from_grid_mat
        return mat if device is None else mat.to(device=device)

    def to_grid(self, embedding: Tensor) -> Tensor:
        """``(N, D, C)`` → ``(N, G, C)``."""
        return torch.einsum(
            "aj,njc->nac",
            self.to_grid_mat.to(dtype=embedding.dtype),
            embedding,
        )

    def from_grid(self, grid: Tensor) -> Tensor:
        """``(N, G, C)`` → ``(N, D, C)``."""
        return torch.einsum(
            "ja,nac->njc",
            self.from_grid_mat.to(dtype=grid.dtype),
            grid,
        )

    def extra_repr(self) -> str:
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, "
            f"lat_resolution={self.lat_resolution}, "
            f"long_resolution={self.long_resolution}, "
            f"normalization={self.normalization!r}, "
            f"rescale_by_mmax={self.rescale_by_mmax}, "
            f"use_m_primary={self.use_m_primary}"
        )


# Legacy aliases (same class).
SO3_Grid = SO3Grid
SO3GridResolved = SO3Grid


def build_so3_grid_table(
    lmax_max: int,
    *,
    normalization: str = "component",
    resolution: int | None = None,
    resolution_list: list[int] | None = None,
    rescale_by_mmax: bool | None = None,
    use_m_primary: bool = False,
) -> nn.ModuleList:
    """Square ``ModuleList[l][m]`` of :class:`SO3Grid` for eSCN / EquiformerV2."""
    table = nn.ModuleList()
    for lval in range(int(lmax_max) + 1):
        row = nn.ModuleList()
        for mval in range(int(lmax_max) + 1):
            row.append(
                SO3Grid(
                    lval,
                    mval,
                    normalization=normalization,
                    resolution=resolution,
                    resolution_list=resolution_list,
                    rescale_by_mmax=rescale_by_mmax,
                    use_m_primary=use_m_primary,
                )
            )
        table.append(row)
    return table
