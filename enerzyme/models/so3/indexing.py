"""Packed SO(3) index helpers shared by equivariant Cores.

Includes DPA4 / EMFA m-major truncation utilities (Li et al., arXiv:2606.02419;
deepmd-kit ``dpa4_nn.indexing`` concepts).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from torch import Tensor


def get_so3_dim_of_lmax(lmax: int) -> int:
    return int((int(lmax) + 1) ** 2)


get_so3_dim = get_so3_dim_of_lmax


def map_degree_idx(lmax: int) -> np.ndarray:
    """Degree ``l`` for each packed ``(l, m)`` row: shape ``((lmax+1)^2,)``."""
    lmax = int(lmax)
    counts = np.array([2 * degree + 1 for degree in range(lmax + 1)], dtype=np.int64)
    return np.repeat(np.arange(lmax + 1, dtype=np.int64), counts)


def so3_packed_index(degree: int, m: int) -> int:
    """Packed index ``l^2 + l + m`` (m ordered ``-l..+l`` within each ``l``)."""
    degree = int(degree)
    m = int(m)
    return degree * degree + degree + m


def build_gie_zonal_index(lmax: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices for GIE zonal (m=0) coupling of non-scalar packed rows."""
    lmax_i = int(lmax)
    ebed_dim = get_so3_dim_of_lmax(lmax_i)
    if lmax_i == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    packed_degree_by_row = map_degree_idx(lmax_i)
    node_row_index = np.arange(1, ebed_dim, dtype=np.int64)
    node_degree_by_row = packed_degree_by_row[1:]
    node_zonal_m0_col_index = node_degree_by_row * (node_degree_by_row + 1)
    node_radial_l_index = node_degree_by_row - 1
    return node_row_index, node_zonal_m0_col_index, node_radial_l_index


def build_m_major_index(lmax: int, mmax: int) -> np.ndarray:
    """Coefficient indices for m-major layout truncated by ``mmax``."""
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")
    indices: list[int] = []
    for degree in range(lmax_i + 1):
        indices.append(so3_packed_index(degree, 0))
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, -m))
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, m))
    return np.array(indices, dtype=np.int64)


def build_m_major_l_index(lmax: int, mmax: int) -> np.ndarray:
    """Degree index aligned with :func:`build_m_major_index`."""
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")
    degrees: list[int] = []
    for degree in range(lmax_i + 1):
        degrees.append(degree)
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)
    return np.array(degrees, dtype=np.int64)


def build_rotate_inv_rescale(
    lmax: int, mmax: int, degree_index: np.ndarray
) -> np.ndarray:
    """Inverse-rotation amplitude rescale when ``mmax < lmax``."""
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    degrees = np.asarray(degree_index, dtype=np.int64)
    rescale = np.ones(degrees.shape[0], dtype=np.float64)
    if mmax_i == lmax_i:
        return rescale
    mask = degrees > mmax_i
    if mask.any():
        denom = float(2 * mmax_i + 1)
        degree_values = degrees[mask].astype(np.float64)
        rescale[mask] = np.sqrt((2.0 * degree_values + 1.0) / denom)
    return rescale


def project_D_to_m(
    D_full: Tensor,
    coeff_index_m: Tensor,
    ebed_dim_full: int,
    cache: Optional[Dict[str, Tensor]] = None,
    key_lmax: int = 0,
    key_mmax: int = 0,
) -> Tensor:
    """Row-project Wigner-D to the m-major truncated layout."""
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = D_block.index_select(1, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def project_Dt_from_m(
    Dt_full: Tensor,
    coeff_index_m: Tensor,
    ebed_dim_full: int,
    cache: Optional[Dict[str, Tensor]] = None,
    key_lmax: int = 0,
    key_mmax: int = 0,
) -> Tensor:
    """Column-project Wigner-Dᵀ from the m-major truncated layout."""
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = Dt_block.index_select(2, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def m_major_reduced_dim(lmax: int, mmax: int) -> int:
    return int(build_m_major_index(lmax, mmax).shape[0])
