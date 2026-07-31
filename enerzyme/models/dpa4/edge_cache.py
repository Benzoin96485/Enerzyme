"""Edge cache for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
    safe_norm,
)


@dataclass
class EdgeCache:
    """Per-edge geometry cache created once per forward pass."""
    src: Tensor
    dst: Tensor
    edge_vec: Tensor
    edge_rbf: Tensor
    edge_env: Tensor
    deg: Tensor
    inv_sqrt_deg: Tensor
    D_full: Optional[Tensor] = None
    Dt_full: Optional[Tensor] = None
    edge_quat: Optional[Tensor] = None


def build_edge_cache(
    *,
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    n_nodes: int,
    radial_basis: torch.nn.Module,
    envelope: torch.nn.Module,
    wigner_calc: WignerDCalculator,
    random_gamma: bool = False,
    deg_norm_floor: float = 1e-7,
) -> EdgeCache:
    """Build the edge cache from sparse edge list.

    Args:
        idx_i: source (neighbor) indices (E,)
        idx_j: destination (center) indices (E,)
        vij: edge vectors (E, 3) from j->i
        n_nodes: total atoms N
        radial_basis: RadialBasis module
        envelope: C3CutoffEnvelope module
        wigner_calc: WignerDCalculator module
        random_gamma: whether to apply random Z-roll (training only)
        deg_norm_floor: floor for degree normalization
    """
    device = vij.device
    dtype = vij.dtype
    src = idx_i.long()
    dst = idx_j.long()

    edge_len = safe_norm(vij)  # (E, 1)
    edge_env = envelope(edge_len)  # (E, 1)
    edge_rbf = radial_basis(edge_len)  # (E, n_radial)

    # Edge quaternion -> Wigner-D
    edge_quat = build_edge_quaternion(vij, edge_len)
    if random_gamma and wigner_calc.training:
        gamma = torch.rand(src.shape[0], device=device, dtype=dtype) * (2 * math.pi)
        edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)
    D_full, Dt_full = wigner_calc(edge_quat)

    # Smooth degree: sum(env^2) over incoming edges per node
    deg = torch.zeros(n_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, dst, (edge_env.squeeze(-1) ** 2))
    inv_sqrt_deg = (1.0 / torch.sqrt(deg + deg_norm_floor)).reshape(n_nodes, 1, 1)

    return EdgeCache(
        src=src,
        dst=dst,
        edge_vec=vij,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_quat=edge_quat,
    )
