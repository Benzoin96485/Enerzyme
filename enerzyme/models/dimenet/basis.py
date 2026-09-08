"""Spherical Fourier–Bessel basis for DimeNet (Gasteiger et al., ICLR 2020).

Evaluates the joint 2D representation of incoming distances ``d_kj`` and
angles ``α_(kj,ji)`` without sympy, matching the official
``SphericalBasisLayer`` (normalized spherical Bessel × ``Y_l^0`` × DimeNet
envelope ``u(x)/x``).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from ..so3.envelope import DimeNetEnvelope


def _spherical_jn_numpy(order: int, r: np.ndarray) -> np.ndarray:
    """Spherical Bessel ``j_n`` via recurrence from ``j_0``, ``j_1``."""
    r = np.asarray(r, dtype=np.float64)
    small = np.abs(r) < 1e-12
    j0 = np.where(small, 1.0, np.sin(r) / r)
    if order == 0:
        return j0
    j1 = np.where(small, r / 3.0, np.sin(r) / (r * r) - np.cos(r) / r)
    if order == 1:
        return j1
    jm2, jm1 = j0, j1
    for ell in range(1, order):
        jl = (2 * ell + 1) / r * jm1 - jm2
        jm2, jm1 = jm1, jl
    return jm1


def _bisection_root(fn, left: float, right: float, tol: float = 1e-12, maxiter: int = 200) -> float:
    f_left, f_right = fn(left), fn(right)
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left * f_right > 0.0:
        raise ValueError(f"no sign change of spherical Bessel in ({left}, {right})")
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        f_mid = fn(mid)
        if abs(f_mid) < tol or (right - left) < tol:
            return float(mid)
        if f_left * f_mid < 0.0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    return float(0.5 * (left + right))


def spherical_bessel_zeros(n_order: int, n_radial: int) -> np.ndarray:
    """First ``n_radial`` zeros of ``j_0 … j_{n_order-1}``.

    Matches ``gasteigerjo/dimenet`` ``Jn_zeros`` (interlacing + bisection).
    ``j_0`` zeros are ``n π``.
    """
    zeros = np.zeros((n_order, n_radial), dtype=np.float64)
    zeros[0] = np.arange(1, n_radial + 1, dtype=np.float64) * np.pi
    if n_order == 1:
        return zeros
    points = np.arange(1, n_radial + n_order, dtype=np.float64) * np.pi
    for order in range(1, n_order):
        n_search = n_radial + n_order - 1 - order
        roots = np.empty(n_search, dtype=np.float64)
        fn = lambda x, ell=order: float(_spherical_jn_numpy(ell, np.array([x]))[0])
        for j in range(n_search):
            roots[j] = _bisection_root(fn, float(points[j]), float(points[j + 1]))
        points = roots
        zeros[order, :n_radial] = roots[:n_radial]
    return zeros


def spherical_bessel_normalizer(zeros: np.ndarray) -> np.ndarray:
    """``1 / sqrt(0.5 j_{l+1}(z_{ln})^2)`` as in official ``bessel_basis``."""
    n_order, n_radial = zeros.shape
    normalizer = np.empty_like(zeros)
    for order in range(n_order):
        for n in range(n_radial):
            j_next = _spherical_jn_numpy(order + 1, np.array([zeros[order, n]]))[0]
            normalizer[order, n] = math.sqrt(2.0) / abs(j_next)
    return normalizer


def spherical_jn(order: int, z: Tensor) -> Tensor:
    """Torch spherical Bessel ``j_n(z)`` via recurrence."""
    small = z.abs() < 1e-8
    j0 = torch.where(small, torch.ones_like(z), torch.sin(z) / z)
    if order == 0:
        return j0
    j1 = torch.where(
        small,
        z / 3.0,
        torch.sin(z) / (z * z) - torch.cos(z) / z,
    )
    if order == 1:
        return j1
    jm2, jm1 = j0, j1
    for ell in range(1, order):
        jl = (2 * ell + 1) / z.clamp(min=1e-12) * jm1 - jm2
        jm2, jm1 = jm1, jl
    return jm1


def real_sph_harm_l0(degree: int, angle: Tensor) -> Tensor:
    """Real ``Y_l^0(θ) = sqrt((2l+1)/4π) P_l(cos θ)`` (DimeNet m=0)."""
    x = torch.cos(angle)
    if degree == 0:
        p = torch.ones_like(x)
    elif degree == 1:
        p = x
    else:
        p_nm2 = torch.ones_like(x)
        p_nm1 = x
        for ell in range(1, degree):
            p = ((2 * ell + 1) * x * p_nm1 - ell * p_nm2) / (ell + 1)
            p_nm2, p_nm1 = p_nm1, p
    pref = math.sqrt((2 * degree + 1) / (4.0 * math.pi))
    return pref * p


class SphericalFourierBesselBasis(Module):
    """Joint SBF ``a_{ln}(d, α)`` on triplets, gathered from pairwise distances.

    Args:
        num_spherical: ``NSHBF`` (``l = 0 … NSHBF-1``).
        num_radial: ``NSRBF`` (``n = 1 … NSRBF``).
        cutoff: short-range cutoff ``c``.
        envelope_exponent: DimeNet envelope exponent (official default 5).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__()
        if num_spherical < 1:
            raise ValueError("num_spherical must be >= 1")
        if num_radial < 1:
            raise ValueError("num_radial must be >= 1")
        if num_radial > 64:
            raise ValueError("num_radial > 64 is not supported (official DimeNet limit)")
        self.num_spherical = int(num_spherical)
        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.envelope = DimeNetEnvelope(envelope_exponent)
        zeros = spherical_bessel_zeros(self.num_spherical, self.num_radial)
        normalizer = spherical_bessel_normalizer(zeros)
        self.register_buffer("zeros", torch.tensor(zeros, dtype=torch.float64))
        self.register_buffer("normalizer", torch.tensor(normalizer, dtype=torch.float64))

    @property
    def out_dim(self) -> int:
        return self.num_spherical * self.num_radial

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        """Evaluate SBF on triplets.

        Args:
            dist: pairwise distances ``d_ji`` aligned with directed edges ``(E,)``.
            angle: triplet angles ``α_(kj,ji)`` ``(T,)``.
            idx_kj: edge index of incoming ``k→j`` for each triplet ``(T,)``.

        Returns:
            ``(T, num_spherical * num_radial)`` joint basis.
        """
        d_scaled = dist / self.cutoff
        env = self.envelope(d_scaled)
        zeros = self.zeros.to(dtype=dist.dtype, device=dist.device)
        normalizer = self.normalizer.to(dtype=dist.dtype, device=dist.device)
        radial = []
        for ell in range(self.num_spherical):
            for n in range(self.num_radial):
                arg = zeros[ell, n] * d_scaled
                radial.append(env * normalizer[ell, n] * spherical_jn(ell, arg))
        rbf = torch.stack(radial, dim=-1)[idx_kj]
        angular = torch.stack(
            [real_sph_harm_l0(ell, angle) for ell in range(self.num_spherical)],
            dim=-1,
        )
        n_sph, n_rad = self.num_spherical, self.num_radial
        return (rbf.view(-1, n_sph, n_rad) * angular.view(-1, n_sph, 1)).reshape(
            -1, n_sph * n_rad
        )


def sbf_shapes(num_spherical: int, num_radial: int) -> Tuple[int, int]:
    return num_spherical, num_radial
