"""Quaternion-based Wigner-D edge frames (DPA4 / SeZM).

Reimplemented in PyTorch from DPA4 concepts (Li et al., arXiv:2606.02419).
Only the essential ``lmax <= 2`` path is implemented for the water-mini
default; the generic polynomial path for higher ``l`` is omitted for v1
but the API is forward-compatible.

Complementary to :class:`~enerzyme.models.so3.rotation.SO3_Rotation`, which
builds Wigner-D from 3×3 rotation matrices via e3nn Euler angles and ``Jd.pt``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


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
    """Wigner-D calculator for lmax <= 2 using direct quaternion formulas.

    For lmax=2 (water-mini default), uses:
    - l=0: identity (1x1)
    - l=1: rotation matrix reordered to packed (l,m) basis
    - l=2: degree-4 quaternion tensor contraction
    """

    def __init__(self, lmax: int, eps: float = 1e-7) -> None:
        super().__init__()
        self.lmax = lmax
        self.eps = eps
        self.dim_full = (lmax + 1) ** 2
        # l=1 reorder: packed basis is (m=-1, m=0, m=+1) -> (y, z, x)
        self.register_buffer("l1_perm", torch.tensor([1, 2, 0], dtype=torch.long))
        l1_sign = torch.tensor([-1.0, -1.0, 1.0])
        self.register_buffer("l1_sign_outer", torch.outer(l1_sign, l1_sign))

        if lmax >= 2:
            s2 = math.sqrt(2.0)
            s6 = math.sqrt(6.0)
            basis = torch.zeros(5, 3, 3)
            basis[0, 0, 1] = basis[0, 1, 0] = 1.0 / s2
            basis[1, 1, 2] = basis[1, 2, 1] = 1.0 / s2
            basis[2, 0, 0] = basis[2, 1, 1] = -1.0 / s6
            basis[2, 2, 2] = 2.0 / s6
            basis[3, 0, 2] = basis[3, 2, 0] = 1.0 / s2
            basis[4, 0, 0] = 1.0 / s2
            basis[4, 1, 1] = -1.0 / s2
            self.register_buffer("l2_basis", basis)

    def forward(self, edge_quaternion: Tensor) -> tuple[Tensor, Tensor]:
        """Build block-diagonal Wigner-D from quaternions.

        Args:
            edge_quaternion: (E, 4) unit quaternions in (w,x,y,z) order.

        Returns:
            (D_full, Dt_full) each with shape (E, D, D) where D=(lmax+1)^2.
        """
        q = quaternion_normalize(edge_quaternion, self.eps)
        n_edge = q.shape[0]
        device = q.device
        dtype = q.dtype
        dim = self.dim_full

        # Build block-diagonal D out-of-place to preserve gradient graph
        blocks = []

        # l=0 block: (E, 1, 1) identity
        l0_block = torch.ones(n_edge, 1, 1, device=device, dtype=dtype)
        blocks.append(l0_block)

        if self.lmax >= 1:
            rot = quaternion_to_rotation_matrix(q)  # (E, 3, 3)
            perm = self.l1_perm
            D_l1 = rot[:, perm][:, :, perm] * self.l1_sign_outer.unsqueeze(0)
            blocks.append(D_l1)  # (E, 3, 3)

        if self.lmax >= 2:
            basis = self.l2_basis.to(dtype=dtype)
            rotated_basis = torch.einsum("eik,bkl,ejl->ebij", rot, basis, rot)
            D_l2 = torch.einsum("aij,ebij->eab", basis, rotated_basis)
            blocks.append(D_l2)  # (E, 5, 5)

        # Construct block-diagonal without in-place ops
        D_full = torch.zeros(n_edge, dim, dim, device=device, dtype=dtype)
        offset = 0
        for blk in blocks:
            sz = blk.shape[1]
            # Create a sparse mask and add
            pad_before = offset
            pad_after = dim - offset - sz
            # Pad each block to (E, dim, dim) with zeros
            blk_padded = torch.nn.functional.pad(
                blk, (pad_before, pad_after, pad_before, pad_after)
            )  # (E, dim, dim)
            D_full = D_full + blk_padded
            offset += sz

        Dt_full = D_full.transpose(-2, -1)
        return D_full, Dt_full

    def forward_zonal(self, edge_quaternion: Tensor, lmin: int = 1) -> Tensor:
        """Build local m=0 to global zonal coupling for GIE."""
        D_full, Dt_full = self.forward(edge_quaternion)
        # Extract m=0 column for each l >= lmin
        zonal_parts = []
        for l in range(max(lmin, 1), self.lmax + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            m0_col = l  # m=0 column within the l-block
            zonal_parts.append(Dt_full[:, start:end, start + m0_col])
        if not zonal_parts:
            return torch.zeros(edge_quaternion.shape[0], 0, device=edge_quaternion.device)
        return torch.cat(zonal_parts, dim=-1)
