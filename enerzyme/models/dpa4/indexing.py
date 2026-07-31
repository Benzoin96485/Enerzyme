"""Packed SO(3) index helpers — re-exported from shared ``enerzyme.models.so3.indexing``."""

from ..so3.indexing import (  # noqa: F401
    build_gie_zonal_index,
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim,
    get_so3_dim_of_lmax,
    m_major_reduced_dim,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
    so3_packed_index,
)

__all__ = [
    "build_gie_zonal_index",
    "build_m_major_index",
    "build_m_major_l_index",
    "build_rotate_inv_rescale",
    "get_so3_dim",
    "get_so3_dim_of_lmax",
    "m_major_reduced_dim",
    "map_degree_idx",
    "project_D_to_m",
    "project_Dt_from_m",
    "so3_packed_index",
]
