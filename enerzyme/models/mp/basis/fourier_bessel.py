"""Spherical Fourier–Bessel basis for DimeNet (Gasteiger et al., ICLR 2020).

Evaluates the joint 2D representation of incoming distances ``d_kj`` and
angles ``α_(kj,ji)``, matching the official ``SphericalBasisLayer``
(normalized spherical Bessel × ``Y_l^0`` × DimeNet envelope ``u(x)/x``).

Spherical-Bessel zeros and normalizers use :mod:`scipy.special` /
:func:`scipy.optimize.brentq` (same recipe as ``gasteigerjo/dimenet``).
The forward pass keeps a torch recurrence so forces stay differentiable.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn as scipy_spherical_jn
from torch import Tensor
from torch.nn import Module

from .envelope import DimeNetEnvelope, polynomial_envelope


def spherical_bessel_zeros(n_order: int, n_radial: int) -> np.ndarray:
    """First ``n_radial`` zeros of ``j_0 … j_{n_order-1}``.

    Matches ``gasteigerjo/dimenet`` ``Jn_zeros`` (interlacing + Brent).
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
        for j in range(n_search):
            roots[j] = brentq(
                lambda x, ell=order: float(scipy_spherical_jn(ell, x)),
                float(points[j]),
                float(points[j + 1]),
            )
        points = roots
        zeros[order, :n_radial] = roots[:n_radial]
    return zeros


def spherical_bessel_normalizer(zeros: np.ndarray) -> np.ndarray:
    """``1 / sqrt(0.5 j_{l+1}(z_{ln})^2)`` as in official ``bessel_basis``."""
    orders = np.arange(zeros.shape[0], dtype=np.int64)[:, None] + 1
    j_next = scipy_spherical_jn(orders, zeros)
    return np.sqrt(2.0) / np.abs(j_next)


def spherical_jn(order: int, z: Tensor) -> Tensor:
    """Differentiable spherical Bessel ``j_n(z)`` (recurrence, same as scipy)."""
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


def normalized_spherical_bessel(scaled: Tensor, zeros: Tensor, normalizer: Tensor) -> Tensor:
    """Evaluate every radial frequency together, returning (edge, degree, frequency)."""
    zeros = zeros.to(scaled)
    normalizer = normalizer.to(scaled)
    scaled = scaled.reshape(-1, 1)
    return torch.stack([
        spherical_jn(ell, scaled * zeros[ell]) * normalizer[ell]
        for ell in range(zeros.shape[0])
    ], dim=1)


def register_bessel_tables(module: Module, degrees: int, frequencies: int) -> None:
    zeros = spherical_bessel_zeros(degrees, frequencies)
    module.register_buffer("zeros", torch.from_numpy(zeros))
    module.register_buffer("normalizer", torch.from_numpy(spherical_bessel_normalizer(zeros)))


def spherical_bessel_expansion(dist: Tensor, zeros: Tensor, normalizer: Tensor, cutoff: float) -> Tensor:
    """M3GNet/MatterSim length-normalized basis, flattened in degree-major order."""
    scaled = dist.clamp(max=cutoff) / cutoff
    radial = normalized_spherical_bessel(scaled, zeros, normalizer)
    return (radial * cutoff ** -1.5).flatten(1)


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
        register_bessel_tables(self, self.num_spherical, self.num_radial)

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
        radial = normalized_spherical_bessel(d_scaled, self.zeros, self.normalizer)
        rbf = (env[:, None, None] * radial).index_select(0, idx_kj)
        angular = torch.stack(
            [real_sph_harm_l0(ell, angle) for ell in range(self.num_spherical)],
            dim=-1,
        )
        n_sph, n_rad = self.num_spherical, self.num_radial
        return (rbf.view(-1, n_sph, n_rad) * angular.view(-1, n_sph, 1)).reshape(
            -1, n_sph * n_rad
        )


class SphericalBesselRadial(Module):
    """Enveloped spherical Bessel radial factor ``(E, L, N)`` or ``(E, L², N)``.

    ``repeat_m`` repeats degree ``l`` across the ``2l+1`` real harmonics so the
    radial tensor lines up with :func:`real_spherical_harmonics`.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        repeat_m: bool = False,
    ) -> None:
        super().__init__()
        if num_spherical < 1:
            raise ValueError("num_spherical must be >= 1")
        if num_radial < 1 or num_radial > 64:
            raise ValueError("num_radial must lie in 1..64")
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.num_spherical = int(num_spherical)
        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.envelope_exponent = int(envelope_exponent)
        self.repeat_m = bool(repeat_m)
        self.norm_const = self.cutoff ** -1.5
        register_bessel_tables(self, self.num_spherical, self.num_radial)
        if self.repeat_m:
            repeats = torch.arange(self.num_spherical, dtype=torch.long) * 2 + 1
            self.register_buffer("degree_repeats", repeats)

    @property
    def angular_dim(self) -> int:
        if self.repeat_m:
            return self.num_spherical * self.num_spherical
        return self.num_spherical

    def forward(self, dist: Tensor) -> Tensor:
        if dist.numel() == 0:
            return dist.new_empty(0, self.angular_dim, self.num_radial)
        scaled = dist / self.cutoff
        env = polynomial_envelope(dist, self.cutoff, self.envelope_exponent)
        radial = normalized_spherical_bessel(scaled, self.zeros, self.normalizer)
        radial = radial * (env * self.norm_const).unsqueeze(-1).unsqueeze(-1)
        if not self.repeat_m:
            return radial
        repeats = self.degree_repeats.to(device=dist.device)
        return torch.repeat_interleave(radial, repeats, dim=1)
