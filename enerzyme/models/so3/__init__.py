"""Shared SO(3) / SO(2) primitives for equivariant GNNs (eSCN and future SO(2) ports).

These modules are adapted from fairchem v1 eSCN (Passaro & Zitnick, *Reducing SO(3)
Convolutions to SO(2)*, 2023). They are **not** used by the Meta UMA wrappers under
``enerzyme.models.esen``, which keep the fairchem ``eSCNMD*`` checkpoint path.
"""

from .coefficient_mapping import CoefficientMapping
from .embedding import SO3_Embedding
from .grid import SO3_Grid
from .rotation import SO3_Rotation, init_edge_rot_mat
from .so2_conv import SO2Block, SO2Conv

__all__ = [
    "CoefficientMapping",
    "SO3_Embedding",
    "SO3_Grid",
    "SO3_Rotation",
    "SO2Block",
    "SO2Conv",
    "init_edge_rot_mat",
]
