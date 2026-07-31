"""Shared SO(3) / SO(2) primitives for equivariant GNNs (eSCN, EquiformerV2, …).

These modules are adapted from fairchem v1 eSCN (Passaro & Zitnick, *Reducing SO(3)
Convolutions to SO(2)*, 2023) with EquiformerV2 extensions (``get_rotate_inv_rescale``,
grid kwargs, ``SO3_LinearV2``). They are **not** used by the Meta UMA wrappers under
``enerzyme.models.esen``, which keep the fairchem ``eSCNMD*`` checkpoint path.
"""

from .coefficient_mapping import CoefficientMapping
from .embedding import SO3_Embedding
from .grid import SO3_Grid
from .linear import SO3_LinearV2
from .rotation import SO3_Rotation, init_edge_rot_mat
from .so2_conv import SO2Block, SO2Conv

__all__ = [
    "CoefficientMapping",
    "SO3_Embedding",
    "SO3_Grid",
    "SO3_LinearV2",
    "SO3_Rotation",
    "SO2Block",
    "SO2Conv",
    "init_edge_rot_mat",
]
