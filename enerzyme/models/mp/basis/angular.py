"""Real spherical harmonics in the coordinate and normalization conventions of each model."""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module


SHLayout = Literal["so3krates", "spookynet_d", "e3nn", "dpa4c", "gemnet", "polynomial"]
SHNormalization = Literal["integral", "norm", "component", "none"]


def _tesseral_degree(x: Tensor, y: Tensor, z: Tensor, squared, degree: int) -> Tensor:
    """Low-degree polynomials in increasing m order, with unit block norm on the sphere."""
    if degree == 0:
        return torch.ones_like(x).unsqueeze(-1)
    if degree == 1:
        return torch.stack((y, z, x), dim=-1)
    if degree == 2:
        sqrt_three = math.sqrt(3.0)
        return torch.stack(
            (
                sqrt_three * x * y,
                sqrt_three * y * z,
                0.5 * (3.0 * z * z - squared),
                sqrt_three * x * z,
                0.5 * sqrt_three * (x * x - y * y),
            ),
            dim=-1,
        )
    if degree == 3:
        return torch.stack(
            (
                math.sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y),
                math.sqrt(15.0) * x * y * z,
                math.sqrt(3.0 / 8.0) * y * (5.0 * z * z - squared),
                0.5 * z * (5.0 * z * z - 3.0 * squared),
                math.sqrt(3.0 / 8.0) * x * (5.0 * z * z - squared),
                0.5 * math.sqrt(15.0) * z * (x * x - y * y),
                math.sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y),
            ),
            dim=-1,
        )
    if degree == 4:
        z_squared = z * z
        x2_minus_y2 = x * x - y * y
        return torch.stack(
            (
                0.5 * math.sqrt(35.0) * x * y * x2_minus_y2,
                0.25 * math.sqrt(70.0) * y * z * (3.0 * x * x - y * y),
                0.5 * math.sqrt(5.0) * x * y * (7.0 * z_squared - squared),
                0.25 * math.sqrt(10.0) * y * z * (7.0 * z_squared - 3.0 * squared),
                0.125
                * (
                    35.0 * z_squared * z_squared
                    - 30.0 * z_squared * squared
                    + 3.0 * squared * squared
                ),
                0.25 * math.sqrt(10.0) * x * z * (7.0 * z_squared - 3.0 * squared),
                0.25 * math.sqrt(5.0) * x2_minus_y2 * (7.0 * z_squared - squared),
                0.25 * math.sqrt(70.0) * x * z * (x * x - 3.0 * y * y),
                0.125 * math.sqrt(35.0) * (x**4 - 6.0 * x * x * y * y + y**4),
            ),
            dim=-1,
        )

    raise ValueError(f"low-degree tesseral polynomials require degree <= 4, got {degree}")


def _higher_tesseral_blocks(x: Tensor, y: Tensor, z: Tensor, squared, lmax: int) -> list[Tensor]:
    """Solid-harmonic recurrence for GemNet bases of arbitrary order.

    Re/Im[(x + iy)^m] contain the azimuthal phase and sin(theta)^m.
    The remaining associated Legendre factor is polynomial even at the poles.
    """
    one = torch.ones_like(x)
    real, imag = one, torch.zeros_like(x)
    channels = [[] for _ in range(lmax + 1)]
    diagonal = 1.0
    for m in range(lmax + 1):
        if m:
            real, imag = x * real - y * imag, x * imag + y * real
            diagonal *= 2 * m - 1
        previous, current = torch.zeros_like(x), diagonal * one
        for degree in range(m, lmax + 1):
            if degree > m:
                previous, current = current, (
                    (2 * degree - 1) * z * current
                    - (degree + m - 1) * squared * previous
                ) / (degree - m)
            factor = math.sqrt((2.0 if m else 1.0) * math.factorial(degree - m) / math.factorial(degree + m))
            values = channels[degree]
            if m:
                values.insert(0, factor * current * imag)
            values.append(factor * current * real)
    return [torch.stack(values, dim=-1) for values in channels]


def degree_widths(lmax: int) -> List[int]:
    """Return the multiplicity ``2l + 1`` for each degree from one through ``lmax``."""
    return [2 * degree + 1 for degree in range(1, int(lmax) + 1)]


def angular_dim(lmax: int) -> int:
    """Steerable width ``(lmax + 1)² - 1``."""
    return (int(lmax) + 1) ** 2 - 1


def split_degrees(tensor: Tensor, lmax: int, dim: int = 1) -> tuple[Tensor, ...]:
    return torch.split(tensor, degree_widths(lmax), dim=dim)


def _polynomial_harmonics(unit_vectors: Tensor, lmax: int) -> Tensor:
    """Real harmonics of unit vectors. Shape ``[E, (lmax + 1)² - 1]``."""
    if unit_vectors.ndim != 2 or unit_vectors.shape[-1] != 3:
        raise ValueError(
            f"unit vectors must have shape [E, 3], got {tuple(unit_vectors.shape)}"
        )
    lmax = int(lmax)
    if lmax < 1 or lmax > 4:
        raise ValueError(f"lmax must be in 1..4, got {lmax}")
    x, y, z = unit_vectors.unbind(dim=-1)
    parts = [x, y, z]
    if lmax == 1:
        return torch.stack(parts, dim=-1)

    sqrt3 = math.sqrt(3.0)
    y2 = y.square()
    x2z2 = x.square() + z.square()
    sh20 = sqrt3 * x * z
    sh21 = sqrt3 * x * y
    sh22 = y2 - 0.5 * x2z2
    sh23 = sqrt3 * y * z
    sh24 = (sqrt3 / 2.0) * (z.square() - x.square())
    parts.extend((sh20, sh21, sh22, sh23, sh24))
    if lmax == 2:
        return torch.stack(parts, dim=-1)

    sqrt42 = math.sqrt(42.0)
    sqrt168 = math.sqrt(168.0)
    sqrt7 = math.sqrt(7.0)
    sh30 = (1.0 / 6.0) * sqrt42 * (sh20 * z + sh24 * x)
    sh31 = sqrt7 * sh20 * y
    sh32 = (1.0 / 8.0) * sqrt168 * (4.0 * y2 - x2z2) * x
    sh33 = 0.5 * sqrt7 * y * (2.0 * y2 - 3.0 * x2z2)
    sh34 = (1.0 / 8.0) * sqrt168 * z * (4.0 * y2 - x2z2)
    sh35 = sqrt7 * sh24 * y
    sh36 = (1.0 / 6.0) * sqrt42 * (sh24 * z - sh20 * x)
    parts.extend((sh30, sh31, sh32, sh33, sh34, sh35, sh36))
    if lmax == 3:
        return torch.stack(parts, dim=-1)

    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    sqrt14 = math.sqrt(14.0)
    sqrt21 = math.sqrt(21.0)
    sqrt210 = math.sqrt(210.0)
    sqrt105 = math.sqrt(105.0)
    sqrt70 = math.sqrt(70.0)
    sh40 = 0.75 * sqrt2 * (sh30 * z + sh36 * x)
    sh41 = 0.75 * sh30 * y + (3.0 / 8.0) * sqrt6 * sh31 * z + (3.0 / 8.0) * sqrt6 * sh35 * x
    sh42 = (
        -3.0 / 56.0 * sqrt14 * sh30 * z
        + (3.0 / 14.0) * sqrt21 * sh31 * y
        + (3.0 / 56.0) * sqrt210 * sh32 * z
        + (3.0 / 56.0) * sqrt210 * sh34 * x
        + (3.0 / 56.0) * sqrt14 * sh36 * x
    )
    sh43 = (
        -3.0 / 56.0 * sqrt42 * sh31 * z
        + (3.0 / 28.0) * sqrt105 * sh32 * y
        + (3.0 / 28.0) * sqrt70 * sh33 * x
        + (3.0 / 56.0) * sqrt42 * sh35 * x
    )
    sh44 = (
        -3.0 / 28.0 * sqrt42 * sh32 * x
        + (3.0 / 7.0) * sqrt7 * sh33 * y
        - 3.0 / 28.0 * sqrt42 * sh34 * z
    )
    sh45 = (
        -3.0 / 56.0 * sqrt42 * sh31 * x
        + (3.0 / 28.0) * sqrt70 * sh33 * z
        + (3.0 / 28.0) * sqrt105 * sh34 * y
        - 3.0 / 56.0 * sqrt42 * sh35 * z
    )
    sh46 = (
        -3.0 / 56.0 * sqrt14 * sh30 * x
        - 3.0 / 56.0 * sqrt210 * sh32 * x
        + (3.0 / 56.0) * sqrt210 * sh34 * z
        + (3.0 / 14.0) * sqrt21 * sh35 * y
        - 3.0 / 56.0 * sqrt14 * sh36 * z
    )
    sh47 = (
        -3.0 / 8.0 * sqrt6 * sh31 * x
        + (3.0 / 8.0) * sqrt6 * sh35 * z
        + 0.75 * sh36 * y
    )
    sh48 = 0.75 * sqrt2 * (-sh30 * x + sh36 * z)
    parts.extend((sh40, sh41, sh42, sh43, sh44, sh45, sh46, sh47, sh48))
    return torch.stack(parts, dim=-1)



def _degree_scale(normalization: str, degree: int) -> float:
    if normalization == "norm":
        return 1.0
    if normalization == "component":
        return math.sqrt(2 * degree + 1)
    if normalization == "integral":
        return math.sqrt((2 * degree + 1) / (4 * math.pi))
    raise ValueError(f"unsupported harmonic normalization {normalization!r}")


def spherical_harmonic_degree(l: int, x: Tensor, normalize: bool, normalization: str = "integral") -> Tensor:
    """One e3nn degree in Cartesian order, with no module construction during forward."""
    from e3nn.o3._spherical_harmonics import _spherical_harmonics

    if not 0 <= l <= 11:
        raise ValueError(f"e3nn spherical harmonics require 0 <= l <= 11, got {l}")
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
    sh = _spherical_harmonics(l, x[..., 0], x[..., 1], x[..., 2])[..., l * l : (l + 1) ** 2]
    if normalization == "norm":
        return sh / math.sqrt(2 * l + 1)
    if normalization == "integral":
        return sh / math.sqrt(4 * math.pi)
    if normalization == "component":
        return sh
    raise ValueError(f"unknown e3nn normalization {normalization!r}")


def spherical_harmonics(
    vecs: Tensor, degrees: Sequence[int] | int | None = None, *,
    lmax: Optional[int] = None, layout: SHLayout = "so3krates",
    normalization: Optional[SHNormalization] = None, normalize_input: bool = True,
    edge_sign: float = 1.0,
) -> Tensor:
    """Evaluate harmonics on (..., 3) vectors in the requested degree order.

    A single integer denotes every degree from zero through that maximum.
    The tesseral So3krates and SpookyNet polynomials assume unit directions.
    DPA4C uses their homogeneous solid-harmonic continuation. ViSNet and
    Geodite use the upstream polynomial amplitudes (normalization="none").
    """
    if lmax is not None:
        if degrees is not None:
            raise ValueError("specify degrees or lmax")
        degrees = lmax
    if isinstance(degrees, int):
        degrees = range(degrees + 1)
    if degrees is None or len(degrees) == 0 or min(degrees) < 0:
        raise ValueError("degrees must be a nonempty sequence of nonnegative integers")
    if vecs.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) vectors, got {tuple(vecs.shape)}")
    max_l = max(degrees)
    if edge_sign != 1.0:
        vecs = vecs * edge_sign
    if normalize_input:
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
    if layout == "e3nn":
        from e3nn.o3._spherical_harmonics import _spherical_harmonics

        if max_l > 11:
            raise NotImplementedError("e3nn spherical harmonics support degrees up to 11")
        x, y, z = vecs.unbind(dim=-1)
        packed = _spherical_harmonics(max_l, x, y, z)
        norm = normalization or "component"
        if norm == "norm":
            # Each degree is normalized independently, including repeats.
            return torch.cat([
                packed[..., l * l : (l + 1) ** 2] / math.sqrt(2 * l + 1)
                for l in degrees
            ], dim=-1)
        if list(degrees) != list(range(max_l + 1)):
            packed = torch.cat([packed[..., l * l : (l + 1) ** 2] for l in degrees], dim=-1)
        if norm == "integral":
            return packed / math.sqrt(4 * math.pi)
        if norm == "component":
            return packed
        raise ValueError(f"unknown e3nn normalization {norm!r}")
    if layout == "polynomial":
        if normalization not in (None, "none") or max_l > 4:
            raise ValueError("polynomial harmonics require normalization='none' and lmax <= 4")
        if max_l == 0:
            return torch.ones_like(vecs[..., :1]).expand(*vecs.shape[:-1], len(degrees))
        packed = _polynomial_harmonics(vecs.reshape(-1, 3), max_l).reshape(
            *vecs.shape[:-1], angular_dim(max_l)
        )
        if list(degrees) == list(range(1, max_l + 1)):
            return packed
        blocks = [
            packed[..., ell * ell - 1 : (ell + 1) ** 2 - 1] if ell else torch.ones_like(vecs[..., :1])
            for ell in degrees
        ]
        return torch.cat(blocks, dim=-1)
    if layout not in ("dpa4c", "spookynet_d", "so3krates", "gemnet"):
        raise ValueError(f"unknown spherical harmonic layout {layout!r}")
    if layout == "spookynet_d" and list(degrees) != [2]:
        raise ValueError("SpookyNet chemistry d order requires degrees=[2]")
    if max_l > 4 and layout != "gemnet":
        raise ValueError(f"{layout} polynomials support degrees zero through four")
    norm = normalization or ("integral" if layout in ("so3krates", "gemnet") else "norm")
    x, y, z = vecs.unbind(-1)
    squared = (x * x + y * y + z * z) if layout == "dpa4c" else 1.0
    high = _higher_tesseral_blocks(x, y, z, squared, max_l) if max_l > 4 else None
    blocks = []
    for ell in degrees:
        if layout == "dpa4c" and ell == 1:
            block = vecs
        elif high is not None:
            block = high[ell]
        else:
            block = _tesseral_degree(x, y, z, squared, ell)
        if layout == "spookynet_d":
            block = block[..., [0, 3, 1, 2, 4]]
        scale = _degree_scale(norm, ell)
        blocks.append(block if scale == 1.0 else block * scale)
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)


def polynomial_spherical_harmonics(unit_vectors: Tensor, lmax: int) -> Tensor:
    """Upstream ViSNet/Geodite polynomial features, omitting degree zero."""
    if unit_vectors.ndim != 2 or not 1 <= lmax <= 4:
        raise ValueError(f"expected [E, 3] vectors and lmax in 1..4, got {tuple(unit_vectors.shape)}, {lmax}")
    return spherical_harmonics(
        unit_vectors, range(1, lmax + 1), layout="polynomial", normalize_input=False,
    )


geodite_spherical_harmonics = polynomial_spherical_harmonics


class RealSphericalHarmonics(Module):
    """Real spherical harmonics on unit vectors for degrees in ``[0, 4]``.

    So3krates / mlff tesseral conventions (``layout='so3krates'``). Prefer the
    functional :func:`spherical_harmonics` for other layouts.

    Parameters
    ----------
    degrees:
        Harmonic degrees to evaluate, in order. Output concatenates blocks of
        length ``2l+1`` for each ``l`` in ``degrees``.
    """

    def __init__(self, degrees: Sequence[int]) -> None:
        super().__init__()
        degrees = list(degrees)
        if not degrees:
            raise ValueError("degrees must be non-empty")
        max_l = max(degrees)
        if max_l < 0 or max_l > 4:
            raise ValueError(
                f"This implementation supports l_max in [0, 4], got {max_l}"
            )
        self.degrees: List[int] = degrees
        self.m_tot = sum(2 * l + 1 for l in degrees)

    def forward(self, vecs: Tensor) -> Tensor:
        """Evaluate SH on displacement vectors.

        Parameters
        ----------
        vecs:
            Shape ``[P, 3]``. Normalized internally.

        Returns
        -------
        Tensor
            Shape ``[P, m_tot]``.
        """
        return spherical_harmonics(
            vecs,
            self.degrees,
            layout="so3krates",
            normalize_input=True,
            edge_sign=1.0,
        )
