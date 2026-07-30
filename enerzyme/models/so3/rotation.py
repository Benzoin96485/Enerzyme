"""Wigner-D rotations for spherical harmonic embeddings.

Adapted from fairchem v1 eSCN / e3nn 0.4.0 (MIT license).
"""

from __future__ import annotations

import os

import torch
from e3nn import o3

from .coefficient_mapping import CoefficientMapping

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
_Jd = torch.load(
    os.path.join(os.path.dirname(__file__), "Jd.pt"),
    map_location="cpu",
    weights_only=False,
)


class SO3_Rotation:
    """Build Wigner-D matrices from per-edge 3x3 rotation matrices and apply them."""

    def __init__(self, rot_mat3x3: torch.Tensor, lmax: int) -> None:
        self.device = rot_mat3x3.device
        self.dtype = rot_mat3x3.dtype

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
        return torch.bmm(wigner_inv, embedding)

    def RotationToWignerDMatrix(
        self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int
    ) -> torch.Tensor:
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(
            len(alpha), size, size, device=self.device, dtype=self.dtype
        )
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = self.wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner

    def wigner_D(self, lval: int, alpha, beta, gamma) -> torch.Tensor:
        if not lval < len(_Jd):
            raise NotImplementedError(
                f"wigner D maximum l implemented is {len(_Jd) - 1}"
            )

        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = _Jd[lval].to(dtype=alpha.dtype, device=alpha.device)
        Xa = self._z_rot_mat(alpha, lval)
        Xb = self._z_rot_mat(beta, lval)
        Xc = self._z_rot_mat(gamma, lval)
        return Xa @ J @ Xb @ J @ Xc

    def _z_rot_mat(self, angle: torch.Tensor, lv: int) -> torch.Tensor:
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * lv + 1, 2 * lv + 1))
        inds = torch.arange(0, 2 * lv + 1, 1, device=device)
        reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
        frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M


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
