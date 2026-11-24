"""Fourier–Bessel and zonal bases used by DimeNet and DimeNet++."""

from typing import Tuple

from ..mp.basis.fourier_bessel import (
    SphericalFourierBesselBasis,
    real_sph_harm_l0,
    spherical_bessel_normalizer,
    spherical_bessel_zeros,
    spherical_jn,
)


def sbf_shapes(num_spherical: int, num_radial: int) -> Tuple[int, int]:
    return num_spherical, num_radial
