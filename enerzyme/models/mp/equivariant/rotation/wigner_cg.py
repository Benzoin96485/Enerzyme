"""Recursive / direct Cartesian Wigner-D for TECE SO(2) edge frames.

Adapted from https://github.com/xvzemin/tace (MIT). Prefer ``wigner_type='recursive'``.
Uses Enerzyme ``tace.cartnn.ICTD`` for the direct path; no ``opt_einsum_fx`` required.
"""

from __future__ import annotations

import torch
from ..cg import wigner_3j
from .fused import OrderProjection
from .wigner_quaternion import quaternion_to_rotation_matrix, quaternion_normalize, safe_norm


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
    return quaternion_normalize(blended, eps)


def init_edge_rot_mat_quaternion(
    edge_distance_vec: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Hopf-fibration quaternion frame aligning the edge with the y-axis."""
    edge_unit = edge_distance_vec / safe_norm(edge_distance_vec, eps)
    x, y, z = edge_unit.unbind(dim=-1)
    q_pos = quaternion_normalize(
        torch.stack([1.0 + y, -z, torch.zeros_like(x), x], dim=-1), eps
    )
    q_neg = quaternion_normalize(
        torch.stack([-z, 1.0 - y, x, torch.zeros_like(x)], dim=-1), eps
    )
    blend = _smooth_step_cinf(0.5 * (y + 1.0))
    quaternion = _quaternion_nlerp(q_neg, q_pos, blend, eps)
    return quaternion_to_rotation_matrix(quaternion)


class WignerD(OrderProjection):
    """Edge-aligned Wigner-D with recursive CG or direct ICT construction."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        rotation_type: str = "quaternion",
        wigner_type: str = "recursive",
        dtype=None,
    ):
        super().__init__(lmax, mmax)
        if rotation_type not in {"quaternion"}:
            raise ValueError(
                f"Unknown rotation_type={rotation_type!r}; expected 'quaternion'."
            )
        if wigner_type not in {"recursive", "direct"}:
            raise ValueError(
                f"Unknown wigner_type={wigner_type!r}; expected 'recursive' or 'direct'."
            )
        self.rotation_type = rotation_type
        self.wigner_type = wigner_type

        if wigner_type == "recursive":
            for l in range(2, lmax + 1):
                self.register_buffer(f"CG_{l}", wigner_3j(1, l - 1, l, dtype=dtype), persistent=False)
        else:
            from ....tace.cartnn import ICTD

            for l in range(lmax + 1):
                _, _, C, _ = ICTD(l, l, False)
                self.register_buffer(f"C_{l}", C[0], persistent=False)

    def get_wigner(self, edge_vector):
        frame = init_edge_rot_mat_quaternion(edge_vector)
        return self.project(self._rotation_to_wigner_matrix(frame, 0, self.lmax))

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

    def _rotation_to_wigner_matrix_recursive(self, edge_rot_mat, start_lmax, end_lmax):
        return wigner_from_cg(edge_rot_mat, end_lmax, start_lmax,
                              coefficients=lambda l: getattr(self, f"CG_{l}"))

    def extra_repr(self):
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, "
            f"rotation={self.rotation_type}, wigner={self.wigner_type}"
        )


def wigner_from_cg(frame, lmax, start_lmax=0, coefficients=None):
    """Build e3nn degree blocks from D¹(R) = R and the highest-spin CG coupling."""
    batch = frame.shape[0]
    blocks = [frame.new_ones(batch, 1, 1)]
    if lmax:
        blocks.append(frame)
    for l in range(2, lmax + 1):
        cg = (wigner_3j(1, l - 1, l, dtype=frame.dtype, device=frame.device)
              if coefficients is None else coefficients(l))
        left = torch.einsum("abm,eac->ebmc", cg, frame)
        left = torch.einsum("ebmc,ebd->emcd", left, blocks[-1])
        blocks.append(torch.einsum("emcd,cdn->emn", left, cg) * (2 * l + 1))
    size = (lmax + 1) ** 2 - start_lmax ** 2
    wigner = frame.new_zeros(batch, size, size)
    offset = 0
    for block in blocks[start_lmax:]:
        width = block.shape[-1]
        wigner[:, offset:offset + width, offset:offset + width] = block
        offset += width
    return wigner
