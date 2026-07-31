"""Fused Wigner-D rotation with m-primary layout (EquiformerV3).

Merges ``._m_primary()`` into the Wigner-D matrices so SO(2) linears see
layout ``(0,...), (1,...)`` without an extra permute. Distinct from
``SO3_Rotation`` used by eSCN / EquiformerV2.
"""

from __future__ import annotations

import math
import os

import torch
from e3nn import o3

# Same Jd tables as so3.rotation / EquiformerV3
_Jd = torch.load(
    os.path.join(os.path.dirname(__file__), "Jd.pt"),
    map_location="cpu",
    weights_only=False,
)

_ROTATION_MASK_THRESHOLD = 0.999999


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}"
        )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


class CoefficientMappingModule(torch.nn.Module):
    """l/m coefficient helpers for EquiformerV3 fused rotation / grids."""

    def __init__(self, lmax, mmax, use_rotate_inv_rescale=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        l_harmonic = []
        m_harmonic = []
        m_complex = []
        for l in range(0, self.lmax + 1):
            mmax_l = min(self.mmax, l)
            m = torch.arange(-mmax_l, mmax_l + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(m_complex, dim=0)
        m_harmonic = torch.cat(m_harmonic, dim=0)
        l_harmonic = torch.cat(l_harmonic, dim=0)

        num_m_coefficients = len(l_harmonic)
        to_m = torch.zeros([num_m_coefficients, num_m_coefficients])
        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)
        to_m = to_m.detach()

        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("to_m", to_m)

        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = self.lmax
        indices = torch.arange(len(l_harmonic))
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)
        mask_idx_i = torch.tensor([]).long()
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)
        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer(f"coefficient_idx_l{l}_m{m}", mask_indices)

    def prepare_coefficient_idx(self):
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, f"coefficient_idx_l{l}_m{m}", None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            return torch.masked_select(indices, mask)
        return self.prepare_coefficient_idx()[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones((1, int((l + 1) ** 2), int((l + 1) ** 2)))
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub ** 2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[
                        :, start_idx : (start_idx + length), start_idx : (start_idx + length)
                    ] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer(f"rotate_inv_rescale_l{l}_m{m}", rotate_inv_rescale)

    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, f"rotate_inv_rescale_l{l}_m{m}", None))
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list

    def get_rotate_inv_rescale(self, lmax, mmax):
        return self.prepare_rotate_inv_rescale()[lmax][mmax]

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, mmax={self.mmax})"


class SO3RotationFused(torch.nn.Module):
    """EquiformerV3 SO(3) rotation with fused m-primary permute."""

    def __init__(self, lmax, mmax, use_rotation_mask=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotation_mask = use_rotation_mask

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
        wigner_index_to_m_array = torch.zeros(to_m.shape[0], ((self.lmax + 1) ** 2))
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer("wigner_index_to_m_array", wigner_index_to_m_array)
        self.register_buffer("wigner_inv_rescale", wigner_inv_rescale)

    def set_wigner(self, rot_mat3x3):
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        index = self.wigner_index_to_m_array.to(
            device=wigner.device, dtype=wigner.dtype
        )
        wigner = torch.einsum("mi, nij -> nmj", index, wigner)
        if torch.is_autocast_enabled():
            wigner = wigner.to(torch.float16)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        rescale = self.wigner_inv_rescale.to(device=wigner.device, dtype=wigner.dtype)
        wigner_inv = wigner_inv * rescale
        if torch.is_autocast_enabled():
            wigner_inv = wigner_inv.to(torch.float16)
        self.wigner = wigner
        self.wigner_inv = wigner_inv

    def rotate(self, inputs):
        return torch.bmm(self.wigner, inputs)

    def rotate_inv(self, inputs):
        return torch.bmm(self.wigner_inv, inputs)

    def _rotation_to_wigner_matrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat[:, :, 1]
        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        R = torch.bmm(R, edge_rot_mat)
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        if self.use_rotation_mask:
            yprod = (x @ x.new_tensor([0, 1, 0])).detach()
            backprop_mask = (yprod > -_ROTATION_MASK_THRESHOLD) & (
                yprod < _ROTATION_MASK_THRESHOLD
            )
            alpha_detach = alpha[(~backprop_mask)].clone().detach()
            gamma_detach = gamma[(~backprop_mask)].clone().detach()
            beta_detach = beta.clone().detach()
            beta_detach[yprod > _ROTATION_MASK_THRESHOLD] = 0.0
            beta_detach[yprod < -_ROTATION_MASK_THRESHOLD] = math.pi
            beta_detach = beta_detach[(~backprop_mask)]

        size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
        wigner = torch.zeros(
            len(alpha), size, size, device=edge_rot_mat.device, dtype=edge_rot_mat.dtype
        )
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            if self.use_rotation_mask:
                block = wigner_D(
                    lmax, alpha[backprop_mask], beta[backprop_mask], gamma[backprop_mask]
                )
                block_detach = wigner_D(lmax, alpha_detach, beta_detach, gamma_detach)
                end = start + block.size()[1]
                wigner[backprop_mask, start:end, start:end] = block
                wigner[(~backprop_mask), start:end, start:end] = block_detach
            else:
                block = wigner_D(lmax, alpha, beta, gamma)
                end = start + block.size()[1]
                wigner[:, start:end, start:end] = block
            start = end
        if self.use_rotation_mask:
            return wigner
        # Keep Wigner in the autograd graph so EnergyReduce+Force can form
        # Fa = -dE/dRa through edge frames (same contract as SO3_Rotation /
        # EquiformerV2). Upstream EquiformerV3 detaches here for direct force
        # heads; Enerzyme's default stack uses energy gradients instead.
        return wigner

    def extra_repr(self):
        return f"lmax={self.lmax}, mmax={self.mmax}"


# Alias matching upstream class name for fixtures / docs
SO3Rotation = SO3RotationFused
