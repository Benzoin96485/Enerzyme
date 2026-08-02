"""Recursive / direct Cartesian Wigner-D for TECE SO(2) edge frames.

Adapted from https://github.com/xvzemin/tace (MIT). Prefer ``wigner_type='recursive'``.
Uses Enerzyme ``tace.cartnn.ICTD`` for the direct path; no ``opt_einsum_fx`` required.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3

from ..tace.cartnn import ICTD
from .rotation_fused import CoefficientMappingModule


def _norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps * eps)


def _quaternion_normalize(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return q / _norm(q, eps)


def _smooth_step_cinf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    eps = torch.finfo(x.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x, min=eps))
    interior = left / (left + right)
    return torch.where(
        x <= 0.0,
        torch.zeros_like(x),
        torch.where(x >= 1.0, torch.ones_like(x), interior),
    )


def _quaternion_nlerp(
    q0: torch.Tensor, q1: torch.Tensor, weight: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight.unsqueeze(-1)) * q0 + weight.unsqueeze(-1) * q1
    return _quaternion_normalize(blended, eps)


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1
            ),
            torch.stack(
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)], dim=-1
            ),
            torch.stack(
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)], dim=-1
            ),
        ],
        dim=-2,
    )


def init_edge_rot_mat_quaternion(
    edge_distance_vec: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Hopf-fibration quaternion frame aligning the edge with the y-axis."""
    edge_unit = edge_distance_vec / _norm(edge_distance_vec, eps)
    x, y, z = edge_unit.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack([1.0 + y, -z, torch.zeros_like(x), x], dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack([-z, 1.0 - y, x, torch.zeros_like(x)], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (y + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return _quaternion_to_rotation_matrix(quaternion)


class WignerD(torch.nn.Module):
    """Edge-aligned Wigner-D with recursive CG or direct ICT construction."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        rotation_type: str = "quaternion",
        wigner_type: str = "recursive",
    ):
        super().__init__()
        if rotation_type not in {"quaternion"}:
            raise ValueError(
                f"Unknown rotation_type={rotation_type!r}; expected 'quaternion'."
            )
        if wigner_type not in {"recursive", "direct"}:
            raise ValueError(
                f"Unknown wigner_type={wigner_type!r}; expected 'recursive' or 'direct'."
            )
        self.mmax = mmax
        self.lmax = lmax
        self.rotation_type = rotation_type
        self.wigner_type = wigner_type

        for l in range(2, self.lmax + 1):
            self.register_buffer(
                f"CG_{l}", o3.wigner_3j(1, l - 1, l), persistent=False
            )
        for l in range(self.lmax + 1):
            _, _, C, _ = ICTD(l, l, False)
            self.register_buffer(f"C_{l}", C[0], persistent=False)

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=True
        )
        wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
        wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)
        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.mmax, use_rotate_inv_rescale=False
        )
        to_m = mapping.to_m
        wigner_inv_rescale = torch.einsum("nia, ba -> nib", wigner_inv_rescale, to_m)
        wigner_index_to_m_array = torch.zeros(to_m.shape[0], (self.lmax + 1) ** 2)
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer("wigner_index_to_m_array", wigner_index_to_m_array)
        self.register_buffer("wigner_inv_rescale", wigner_inv_rescale)

    def get_wigner(self, edge_vector) -> tuple[torch.Tensor, torch.Tensor]:
        rot_mat3x3 = init_edge_rot_mat_quaternion(edge_vector)
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        wigner = torch.einsum("mi, nij -> nmj", self.wigner_index_to_m_array, wigner)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        return wigner, wigner_inv

    def _rotation_to_wigner_matrix(self, edge_rot_mat, start_lmax, end_lmax):
        if self.wigner_type == "direct":
            return self._rotation_to_wigner_matrix_direct(
                edge_rot_mat, start_lmax, end_lmax
            )
        return self._rotation_to_wigner_matrix_recursive(
            edge_rot_mat, start_lmax, end_lmax
        )

    def _rotate_cartesian_tensor(
        self, cartesian_basis: torch.Tensor, rotation: torch.Tensor, degree: int
    ) -> torch.Tensor:
        out = cartesian_basis
        for axis in range(1, degree + 1):
            out = out.movedim(axis, -1)
            out = torch.einsum("bij,b...j->b...i", rotation, out)
            out = out.movedim(-1, axis)
        return out

    def _rotation_to_wigner_matrix_direct(
        self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
        blocks = []
        batch = edge_rot_mat.shape[0]
        for degree in range(start_lmax, end_lmax + 1):
            C = getattr(self, f"C_{degree}")
            width = 2 * degree + 1
            cartesian_basis = C.unsqueeze(0).expand(batch, -1, -1)
            cartesian_basis = cartesian_basis.reshape(batch, *([3] * degree), width)
            rotated_basis = self._rotate_cartesian_tensor(
                cartesian_basis, edge_rot_mat, degree
            )
            rotated_basis = rotated_basis.reshape(batch, 3**degree, width)
            blocks.append(torch.einsum("pi,bpj->bij", C, rotated_basis))

        size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
        wigner = edge_rot_mat.new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            wigner[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return wigner

    def _compute_one_wigner(
        self, degree: int, d1: torch.Tensor, d_prev: torch.Tensor, cg: torch.Tensor
    ) -> torch.Tensor:
        left = torch.einsum("abm,eac->ebmc", cg, d1)
        left = torch.einsum("ebmc,ebd->emcd", left, d_prev)
        return torch.einsum("emcd,cdn->emn", left, cg)

    def _rotation_to_wigner_matrix_recursive(
        self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
        batch = edge_rot_mat.shape[0]
        all_blocks = [edge_rot_mat.new_ones(batch, 1, 1)]
        if end_lmax >= 1:
            all_blocks.append(edge_rot_mat)
        for degree in range(2, end_lmax + 1):
            cg = getattr(self, f"CG_{degree}")
            block = self._compute_one_wigner(
                degree, all_blocks[1], all_blocks[degree - 1], cg
            )
            all_blocks.append(block * (2 * degree + 1))
        blocks = all_blocks[start_lmax : end_lmax + 1]
        size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
        wigner = edge_rot_mat.new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            wigner[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return wigner

    def extra_repr(self):
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, "
            f"rotation={self.rotation_type}, wigner={self.wigner_type}"
        )
