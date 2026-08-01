"""Quaternion edge frames for DPA4 / SeZM, with shared e3nn/Jd Wigner-D.

Edge frames stay quaternion-based (``build_edge_quaternion``, optional Z-roll).
Packed Wigner-D matrices come from
:func:`~enerzyme.models.so3.wigner_jd.wigner_from_rotation_matrix` after
``quaternion_to_rotation_matrix``, supporting any ``lmax`` packaged in ``Jd.pt``.

DPA4's historical l=1 Wigner block used a signed permutation of the Cartesian
rotation (``A R Aᵀ`` with ``A`` mapping ``(x,y,z) → (-y,-z,x)``). The shared
e3nn/Jd backend returns ``D¹(R) = R``, so :class:`WignerDCalculator` evaluates
Wigner-D on ``A R(q) Aᵀ`` to keep that DPA4 layout for every ``l`` (and restore
SO(3) scalar invariance of the edge-frame stack).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .wigner_jd import max_wigner_lmax, wigner_from_rotation_matrix

# Signed permutation: rows are -e_y, -e_z, +e_x (historical DPA4 l=1 basis).
_DPA4_CARTESIAN_BASIS = torch.tensor(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)


def safe_norm(x: Tensor, eps: float = 1e-7) -> Tensor:
    return torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps * eps)


def quaternion_normalize(q: Tensor, eps: float = 1e-7) -> Tensor:
    return q / safe_norm(q, eps)


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    w, x, y, z = q.unbind(-1)
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return torch.stack([
        torch.stack([1-2*(y2+z2), 2*(xy-wz), 2*(xz+wy)], -1),
        torch.stack([2*(xy+wz), 1-2*(x2+z2), 2*(yz-wx)], -1),
        torch.stack([2*(xz-wy), 2*(yz+wx), 1-2*(x2+y2)], -1),
    ], dim=-2)


def quaternion_z_rotation(gamma: Tensor) -> Tensor:
    hg = 0.5 * gamma
    w = torch.cos(hg)
    z = torch.sin(hg)
    zeros = torch.zeros_like(gamma)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def _smooth_step(x: Tensor) -> Tensor:
    x_c = torch.clamp(x, 0.0, 1.0)
    eps = torch.finfo(x_c.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x_c, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x_c, min=eps))
    interior = left / (left + right)
    return torch.where(x_c <= 0, torch.zeros_like(x_c),
           torch.where(x_c >= 1, torch.ones_like(x_c), interior))


def build_edge_quaternion(edge_vec: Tensor, edge_len: Tensor | None = None,
                          eps: float = 1e-7) -> Tensor:
    """Stable edge quaternion for global->local +Z convention."""
    if edge_len is None:
        edge_len = safe_norm(edge_vec, eps)
    else:
        edge_len = torch.sqrt(edge_len * edge_len + eps * eps)
    eu = edge_vec / edge_len
    x, y, z = eu[..., 0], eu[..., 1], eu[..., 2]

    # +Z chart
    q_pos = torch.stack([1.0 + z, y, -x, torch.zeros_like(x)], dim=-1)
    q_pos = quaternion_normalize(q_pos, eps)
    # -Z chart
    q_neg = torch.stack([-x, torch.zeros_like(x), 1.0 - z, y], dim=-1)
    q_neg = quaternion_normalize(q_neg, eps)

    blend = _smooth_step(0.5 * (z + 1.0))
    # nlerp
    dot = (q_pos * q_neg).sum(-1, keepdim=True)
    q_neg_aligned = torch.where(dot < 0, -q_neg, q_neg)
    blended = (1.0 - blend.unsqueeze(-1)) * q_neg_aligned + blend.unsqueeze(-1) * q_pos
    return quaternion_normalize(blended, eps)


class WignerDCalculator(nn.Module):
    """Packed e3nn Wigner-D from unit quaternions (shared Jd backend).

    Keeps DPA4's quaternion edge frame while supporting any ``lmax`` packaged
    in ``Jd.pt`` (see :func:`~enerzyme.models.so3.wigner_jd.max_wigner_lmax`).
    Evaluates Wigner-D on ``A R(q) Aᵀ`` so the packed layout matches DPA4's
    historical Cartesian / SO(2) basis (not raw e3nn ``D(R) = R`` for ``l=1``).
    """

    def __init__(self, lmax: int, eps: float = 1e-7) -> None:
        super().__init__()
        lmax = int(lmax)
        if lmax < 0:
            raise ValueError(f"`lmax` must be non-negative, got {lmax}")
        if lmax > max_wigner_lmax():
            raise NotImplementedError(
                f"wigner D maximum l implemented is {max_wigner_lmax()}, got lmax={lmax}"
            )
        self.lmax = lmax
        self.eps = eps
        self.dim_full = (lmax + 1) ** 2
        self.register_buffer("_basis_A", _DPA4_CARTESIAN_BASIS.clone(), persistent=False)

    def forward(self, edge_quaternion: Tensor) -> tuple[Tensor, Tensor]:
        """Build block-diagonal Wigner-D from quaternions.

        Args:
            edge_quaternion: (E, 4) unit quaternions in (w,x,y,z) order.

        Returns:
            (D_full, Dt_full) each with shape (E, D, D) where D=(lmax+1)^2.
        """
        q = quaternion_normalize(edge_quaternion, self.eps)
        rot = quaternion_to_rotation_matrix(q)
        A = self._basis_A.to(device=rot.device, dtype=rot.dtype)
        # Conjugate into DPA4's Cartesian basis before the shared Jd backend.
        rot = A @ rot @ A.transpose(-2, -1)
        D_full = wigner_from_rotation_matrix(rot, end_lmax=self.lmax, start_lmax=0)
        Dt_full = D_full.transpose(-2, -1)
        return D_full, Dt_full

    def forward_zonal(self, edge_quaternion: Tensor, lmin: int = 1) -> Tensor:
        """Build local m=0 to global zonal coupling for GIE."""
        D_full, Dt_full = self.forward(edge_quaternion)
        zonal_parts = []
        for l in range(max(lmin, 1), self.lmax + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            m0_col = l  # m=0 column within the l-block
            zonal_parts.append(Dt_full[:, start:end, start + m0_col])
        if not zonal_parts:
            return torch.zeros(edge_quaternion.shape[0], 0, device=edge_quaternion.device)
        return torch.cat(zonal_parts, dim=-1)
