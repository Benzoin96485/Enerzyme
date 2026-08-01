"""Wigner-D rotations for spherical harmonic embeddings.

Uses the shared e3nn / Jd backend in :mod:`enerzyme.models.so3.wigner_jd`.
Adapted from fairchem v1 eSCN / e3nn 0.4.0 (MIT license).

For quaternion edge frames (DPA4 / EMFA), see
:mod:`enerzyme.models.so3.wigner_quaternion`.
"""

from __future__ import annotations

import torch

from .coefficient_mapping import CoefficientMapping
from .wigner_jd import (
    max_wigner_lmax,
    wigner_D,
    wigner_from_rotation_matrix,
    z_rot_mat,
)


class SO3_Rotation:
    """Build Wigner-D matrices from per-edge 3x3 rotation matrices and apply them."""

    def __init__(
        self,
        rot_mat3x3: torch.Tensor,
        lmax: int,
        apply_rotate_inv_rescale: bool = False,
    ) -> None:
        self.device = rot_mat3x3.device
        self.dtype = rot_mat3x3.dtype
        # EquiformerV2 needs mmax-truncation rescale on rotate-back; paper eSCN
        # (fairchem v1) does not — keep False for escn backward compatibility.
        self.apply_rotate_inv_rescale = apply_rotate_inv_rescale

        # Keep the Wigner graph so EnergyReduce+Force can form Fa = -dE/dRa
        # through edge frames (fairchem v1 detached here for direct ForceBlock).
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()

        self.set_lmax(lmax)

    def set_lmax(self, lmax: int) -> None:
        self.lmax = lmax
        self.mapping = CoefficientMapping([self.lmax], [self.lmax], self.device)

    def rotate(self, embedding: torch.Tensor, out_lmax: int, out_mmax: int) -> torch.Tensor:
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    def rotate_inv(self, embedding: torch.Tensor, in_lmax: int, in_mmax: int) -> torch.Tensor:
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]
        if self.apply_rotate_inv_rescale and in_mmax < in_lmax:
            wigner_inv = wigner_inv * self.mapping.get_rotate_inv_rescale(
                in_lmax, in_mmax
            )
        return torch.bmm(wigner_inv, embedding)

    def RotationToWignerDMatrix(
        self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
        return wigner_from_rotation_matrix(
            edge_rot_mat, end_lmax=end_lmax, start_lmax=start_lmax
        )

    def wigner_D(self, lval: int, alpha, beta, gamma) -> torch.Tensor:
        return wigner_D(lval, alpha, beta, gamma)

    def _z_rot_mat(self, angle: torch.Tensor, lv: int) -> torch.Tensor:
        return z_rot_mat(angle, lv)


def init_edge_rot_mat(edge_distance_vec: torch.Tensor) -> torch.Tensor:
    """Build a deterministic orthonormal edge frame from displacement vectors.

    Unlike the original fairchem eSCN random completion, this uses a stable
    axis choice so rotations of the system transform the frame consistently.
    The returned matrix stays in the autograd graph so default EnergyReduce+Force
    stacks obtain the angular contribution to Fa = -dE/dRa (fairchem v1 detached
    the frame for its direct ForceBlock path).
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.linalg.norm(edge_vec_0, dim=1).clamp(min=1e-8)
    norm_x = edge_vec_0 / edge_vec_0_distance.view(-1, 1)

    # Deterministic helper axis: pick the Cartesian axis least aligned with norm_x
    abs_x = torch.abs(norm_x)
    min_axis = torch.argmin(abs_x, dim=1)
    helper = torch.zeros_like(norm_x)
    helper.scatter_(1, min_axis.unsqueeze(1), 1.0)

    norm_z = torch.cross(norm_x, helper, dim=1)
    norm_z = norm_z / torch.linalg.norm(norm_z, dim=1, keepdim=True).clamp(min=1e-8)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.linalg.norm(norm_y, dim=1, keepdim=True).clamp(min=1e-8)

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)
    return edge_rot_mat


__all__ = [
    "SO3_Rotation",
    "init_edge_rot_mat",
    "max_wigner_lmax",
]
