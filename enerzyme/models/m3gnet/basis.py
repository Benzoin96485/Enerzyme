"""Spherical Bessel features for M3GNet bonds and triplets.

Chen and Ong, Nat. Comput. Sci. 2022. The radial factor is

``j_l(z_{ln} r / c) * sqrt(2) / (c^{3/2} |j_{l+1}(z_{ln})|)``

with ``l`` major and ``n`` minor. Channels vanish at ``r = c``. There is no
polynomial envelope on this expansion; the three-body weight is applied later.
Zeros and ``Y_l^0`` are the shared DimeNet helpers. ``l = 0`` reduces to
``sqrt(2/c) sin(n π r / c) / r``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..layers.rbf import BaseRBF


def _dimenet_basis():
    from ..dimenet.basis import (
        real_sph_harm_l0,
        spherical_bessel_normalizer,
        spherical_bessel_zeros,
        spherical_jn,
    )

    return (
        real_sph_harm_l0,
        spherical_bessel_normalizer,
        spherical_bessel_zeros,
        spherical_jn,
    )


def spherical_bessel_expansion(
    dist: Tensor,
    zeros: Tensor,
    normalizer: Tensor,
    cutoff: float,
) -> Tensor:
    """``(E, max_l * num_radial)`` spherical Bessel expansion.

    ``zeros`` and ``normalizer`` have shape ``(max_l, num_radial)``.
    ``normalizer`` is ``sqrt(2) / |j_{l+1}(z)|``; this function multiplies
    ``c^{-3/2}``.
    """
    max_l, num_radial = zeros.shape
    degree = int(max_l * num_radial)
    if dist.numel() == 0:
        return dist.new_empty(0, degree)
    _, _, _, spherical_jn = _dimenet_basis()
    scaled = dist.clamp(max=float(cutoff)) / float(cutoff)
    scale = float(cutoff) ** -1.5
    zeros = zeros.to(dtype=dist.dtype, device=dist.device)
    normalizer = normalizer.to(dtype=dist.dtype, device=dist.device)
    radial = dist.new_empty(dist.shape[0], max_l, num_radial)
    scaled_col = scaled.unsqueeze(-1)
    for ell in range(max_l):
        arg = zeros[ell].unsqueeze(0) * scaled_col
        radial[:, ell, :] = spherical_jn(ell, arg) * normalizer[ell].unsqueeze(0)
    return (radial * scale).reshape(dist.shape[0], degree)


def joint_zonal_basis(
    rbf: Tensor,
    angle: Tensor,
    idx_ik: Tensor,
    max_l: int,
    num_radial: int,
) -> Tensor:
    """``j_l(r_ik) Y_l^0(θ_jik)`` flattened to ``(T, max_l * num_radial)``.

    ``rbf`` is the bond expansion of every edge. The radial factor is taken
    from the other bond ``i→k``, which is the distance that multiplies the
    spherical harmonic in the M3GNet three-body term.
    """
    degree = int(max_l) * int(num_radial)
    if idx_ik.numel() == 0:
        return rbf.new_empty(0, degree)
    if rbf.shape[-1] != degree:
        raise ValueError(
            f"radial basis width {rbf.shape[-1]} != max_l * num_radial ({degree})"
        )
    if angle.shape[0] != idx_ik.shape[0]:
        raise ValueError(
            f"angle rows {angle.shape[0]} != triplet rows {idx_ik.shape[0]}"
        )
    real_sph_harm_l0, _, _, _ = _dimenet_basis()
    radial = rbf.index_select(0, idx_ik).view(-1, max_l, num_radial)
    if angle.numel() == 0:
        return rbf.new_empty(0, degree)
    angular = torch.stack(
        [real_sph_harm_l0(ell, angle) for ell in range(max_l)],
        dim=-1,
    )
    return (radial * angular.unsqueeze(-1)).reshape(-1, degree)


class SphericalBesselRBFLayer(BaseRBF):
    """M3GNet bond expansion. YAML name: ``SphericalBesselRBF``.

    ``num_rbf`` is the radial count ``max_n``. ``max_l`` is the number of
    spherical orders ``l = 0 … max_l - 1``. The output width is
    ``max_l * num_rbf``. ``apply_cutoff_fn`` stays off: a cutoff envelope on
    this basis is not the M3GNet three-body weight.
    """

    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        max_l: int = 3,
        cutoff_fn: Optional[str] = None,
        apply_cutoff_fn: bool = False,
    ) -> None:
        if apply_cutoff_fn:
            raise ValueError(
                "SphericalBesselRBF does not apply a cutoff envelope. "
                "M3GNetCore multiplies the three-body term by a polynomial cutoff."
            )
        if int(max_l) < 1:
            raise ValueError(f"max_l must be >= 1, got {max_l}")
        if int(num_rbf) < 1 or int(num_rbf) > 64:
            raise ValueError(f"num_rbf must lie in 1..64, got {num_rbf}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        super().__init__(
            int(num_rbf),
            float(cutoff_sr),
            cutoff_fn,
            apply_cutoff_fn=False,
        )
        self.max_l = int(max_l)
        self.flavor = "m3gnet"
        self.degree = self.max_l * self.num_rbf
        _, spherical_bessel_normalizer, spherical_bessel_zeros, _ = _dimenet_basis()
        zeros = spherical_bessel_zeros(self.max_l, self.num_rbf)
        normalizer = spherical_bessel_normalizer(zeros)
        self.register_buffer("zeros", torch.tensor(zeros, dtype=torch.float64))
        self.register_buffer("normalizer", torch.tensor(normalizer, dtype=torch.float64))

    def _get_rbf(self, Dij: Tensor) -> Tensor:
        return spherical_bessel_expansion(
            Dij, self.zeros, self.normalizer, self.cutoff_sr
        )
