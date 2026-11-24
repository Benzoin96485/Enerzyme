"""Fourier–Bessel bases for GemNet angles and dihedrals.

The radial factor is the normalized spherical Bessel function times the
polynomial envelope ``u_p(x)`` and ``c^{-3/2}``. Triplet angles use ``Y_l^0``.
Dihedrals use the real spherical harmonics ``Y_l^m`` with ``m = -l … l``,
packed at index ``l² + (m + l)``. Zeros and ``Y_l^0`` are the shared DimeNet
helpers; the envelope power and the ``c^{-3/2}`` factor are GemNet's, not
DimeNet's ``u(x)/x`` basis.
"""

from __future__ import annotations

import torch
from torch import Tensor
from ..mp.basis.angular import spherical_harmonics
from ..mp.basis.fourier_bessel import SphericalBesselRadial, real_sph_harm_l0


def legendre_outer(angle_cab: Tensor, dihedral: Tensor, num_spherical: int) -> Tensor:
    """Outer product of two zonal harmonics, flattened to ``(Q, L²)``.

    GemNet-OC's ``legendre_outer`` spherical basis. Channel
    ``l_cab * L + l_dihedral`` stores ``Y_{l_cab}^0(angle_cab) Y_{l_dihedral}^0(φ)``.
    Both angles are the unsigned angles in ``[0, π]`` already used by GemNet-Q.
    This is not the real ``Y_l^m`` basis of GemNet-Q.
    """
    left = y_lm0(angle_cab, num_spherical)
    right = y_lm0(dihedral, num_spherical)
    if left.numel() == 0:
        return left.new_empty(0, num_spherical * num_spherical)
    return (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(left.shape[0], -1)


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
    if theta.shape != phi.shape:
        raise ValueError(f"theta and phi shapes differ: {tuple(theta.shape)} vs {tuple(phi.shape)}")
    sin_theta = torch.sin(theta)
    direction = torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)), dim=-1)
    return spherical_harmonics(direction, lmax=num_spherical - 1, layout="gemnet", normalize_input=False)


def joint_zonal(radial: Tensor, angular: Tensor) -> Tensor:
    """``R_{ln}(d) Y_l^0(α)`` flattened to ``(T, L · N)``."""
    if radial.shape[:2] != (angular.shape[0], angular.shape[1]):
        raise ValueError(
            "joint zonal basis needs radial (T, L, N) and angular (T, L), "
            f"got {tuple(radial.shape)} and {tuple(angular.shape)}"
        )
    return (radial * angular.unsqueeze(-1)).reshape(radial.shape[0], -1)
