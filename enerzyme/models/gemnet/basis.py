"""Fourier–Bessel bases for GemNet angles and dihedrals.

The radial factor is the normalized spherical Bessel function times the
polynomial envelope ``u_p(x)`` and ``c^{-3/2}``. Triplet angles use ``Y_l^0``.
Dihedrals use the real spherical harmonics ``Y_l^m`` with ``m = -l … l``,
packed at index ``l² + (m + l)``. Zeros and ``Y_l^0`` are the shared DimeNet
helpers; the envelope power and the ``c^{-3/2}`` factor are GemNet's, not
DimeNet's ``u(x)/x`` basis.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module

from ..cutoff import polynomial_envelope
from ..dimenet.basis import (
    real_sph_harm_l0,
    spherical_bessel_normalizer,
    spherical_bessel_zeros,
    spherical_jn,
)


def y_lm0(angle: Tensor, num_spherical: int) -> Tensor:
    """``Y_l^0(angle)`` for ``l = 0 … num_spherical-1``, shape ``(T, L)``."""
    if angle.numel() == 0:
        return angle.new_empty(0, num_spherical)
    return torch.stack(
        [real_sph_harm_l0(ell, angle) for ell in range(num_spherical)],
        dim=-1,
    )


def real_spherical_harmonics(theta: Tensor, phi: Tensor, num_spherical: int) -> Tensor:
    """Real ``Y_l^m(θ, φ)`` up to ``l = num_spherical - 1``.

    Channel ``l² + (m + l)`` stores ``m``. The ``m > 0`` functions carry the
    Condon–Shortley factor ``(-1)^m``, and ``m = 0`` matches
    :func:`enerzyme.models.dimenet.basis.real_sph_harm_l0`.
    """
    if num_spherical < 1:
        raise ValueError("num_spherical must be >= 1")
    n_out = num_spherical * num_spherical
    if theta.numel() == 0:
        return theta.new_empty(0, n_out)
    if theta.shape != phi.shape:
        raise ValueError(
            f"theta and phi shapes differ: {tuple(theta.shape)} vs {tuple(phi.shape)}"
        )

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    legendre: list[list[Tensor | None]] = [
        [None] * num_spherical for _ in range(num_spherical)
    ]
    diagonal = torch.ones_like(cos_t)
    legendre[0][0] = diagonal
    for order in range(1, num_spherical):
        diagonal = -(2 * order - 1) * sin_t * diagonal
        legendre[order][order] = diagonal
    for order in range(num_spherical - 1):
        previous = legendre[order][order]
        assert previous is not None
        legendre[order + 1][order] = (2 * order + 1) * cos_t * previous
    for degree in range(2, num_spherical):
        for order in range(degree - 1):
            lower = legendre[degree - 1][order]
            lower2 = legendre[degree - 2][order]
            assert lower is not None and lower2 is not None
            legendre[degree][order] = (
                (2 * degree - 1) * cos_t * lower - (degree + order - 1) * lower2
            ) / (degree - order)

    out = theta.new_empty(theta.shape[0], n_out)
    for degree in range(num_spherical):
        zonal = legendre[degree][0]
        assert zonal is not None
        norm0 = math.sqrt((2 * degree + 1) / (4.0 * math.pi))
        out[:, degree * degree + degree] = norm0 * zonal
        for order in range(1, degree + 1):
            associated = legendre[degree][order]
            assert associated is not None
            norm = math.sqrt(
                (2 * degree + 1)
                / (4.0 * math.pi)
                * math.factorial(degree - order)
                / math.factorial(degree + order)
            )
            base = math.sqrt(2.0) * ((-1) ** order) * norm * associated
            out[:, degree * degree + degree + order] = base * torch.cos(order * phi)
            out[:, degree * degree + degree - order] = base * torch.sin(order * phi)
    return out


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
        zeros = spherical_bessel_zeros(self.num_spherical, self.num_radial)
        normalizer = spherical_bessel_normalizer(zeros)
        self.register_buffer("zeros", torch.tensor(zeros, dtype=torch.float64))
        self.register_buffer("normalizer", torch.tensor(normalizer, dtype=torch.float64))
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
        zeros = self.zeros.to(dtype=dist.dtype, device=dist.device)
        normalizer = self.normalizer.to(dtype=dist.dtype, device=dist.device)
        radial = dist.new_empty(dist.shape[0], self.num_spherical, self.num_radial)
        scaled_col = scaled.unsqueeze(-1)
        for ell in range(self.num_spherical):
            arg = zeros[ell].unsqueeze(0) * scaled_col
            radial[:, ell, :] = spherical_jn(ell, arg) * normalizer[ell].unsqueeze(0)
        radial = radial * (env * self.norm_const).unsqueeze(-1).unsqueeze(-1)
        if not self.repeat_m:
            return radial
        repeats = self.degree_repeats.to(device=dist.device)
        return torch.repeat_interleave(radial, repeats, dim=1)


def joint_zonal(radial: Tensor, angular: Tensor) -> Tensor:
    """``R_{ln}(d) Y_l^0(α)`` flattened to ``(T, L · N)``."""
    if radial.shape[:2] != (angular.shape[0], angular.shape[1]):
        raise ValueError(
            "joint zonal basis needs radial (T, L, N) and angular (T, L), "
            f"got {tuple(radial.shape)} and {tuple(angular.shape)}"
        )
    return (radial * angular.unsqueeze(-1)).reshape(radial.shape[0], -1)
