"""Spherical Bessel features for M3GNet bonds and triplets.

Chen and Ong, Nat. Comput. Sci. 2022. The radial factor is

``j_l(z_{ln} r / c) * sqrt(2) / (c^{3/2} |j_{l+1}(z_{ln})|)``

with ``l`` major and ``n`` minor. Channels vanish at ``r = c``. There is no
polynomial envelope on this expansion; the three-body weight is applied later.
Zeros and ``Y_l^0`` are the shared DimeNet helpers. ``l = 0`` reduces to
``sqrt(2/c) sin(n π r / c) / r``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..layers.rbf import SphericalBesselRBFLayer
from ..mp.basis.fourier_bessel import real_sph_harm_l0, spherical_bessel_expansion


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
    radial = rbf.index_select(0, idx_ik).view(-1, max_l, num_radial)
    if angle.numel() == 0:
        return rbf.new_empty(0, degree)
    angular = torch.stack(
        [real_sph_harm_l0(ell, angle) for ell in range(max_l)],
        dim=-1,
    )
    return (radial * angular.unsqueeze(-1)).reshape(-1, degree)
