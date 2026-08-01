"""Real spherical harmonics with a single tesseral core and layout adapters.

Two evaluation backends:

* **tesseral** — classical real (tesseral) polynomials (So3krates / SpookyNet).
* **e3nn** — ``e3nn.o3.spherical_harmonics`` CG basis (Equiformer / MACE / …).

Architecture-specific *m*-order and normalization are registered layouts on top
of these backends. Tesseral and e3nn bases are **not** related by a static
channel permute for ``l >= 2``.

Adapted from So3krates-torch
(https://github.com/TCPUniLU/So3krates-torch, MIT license) for the tesseral
closed forms (mlff / So3krates phase conventions).
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import Module

PI = math.pi
_sqrt = math.sqrt

SHLayout = Literal["so3krates", "spookynet_d", "e3nn"]
SHNormalization = Literal["integral", "norm", "component"]

# So3krates tesseral l=2 order: (xy, yz, z2, xz, x2-y2)
# SpookyNet chemistry d order:  (xy, xz, yz, z2, x2-y2)
_SPOOKYNET_D_FROM_TESSERAL_L2 = (0, 3, 1, 2, 4)
# integral → SpookyNet ``norm`` (‖Y‖=1 on the unit sphere)
_SPOOKYNET_D_FROM_INTEGRAL = 2.0 * _sqrt(PI / 5.0)


def _tesseral_real_sh_xyz(
    x: Tensor, y: Tensor, z: Tensor, degrees: Sequence[int]
) -> Tensor:
    """Evaluate classical tesseral Y_lm on already-normalized Cartesian components.

    Degree blocks use So3krates / mlff ordering, e.g. ``l=1 → (y, z, x)``,
    ``l=2 → (xy, yz, z², xz, x²−y²)``, with integral (4π) normalization.
    """
    degrees = list(degrees)
    n = x.shape[0]
    m_tot = sum(2 * l + 1 for l in degrees)
    out = torch.empty(n, m_tot, dtype=x.dtype, device=x.device)
    idx = 0
    for degree in degrees:
        if degree == 0:
            out[:, idx] = 0.5 * _sqrt(1 / PI)
            idx += 1
        elif degree == 1:
            c1 = _sqrt(3 / (4 * PI))
            out[:, idx] = c1 * y
            out[:, idx + 1] = c1 * z
            out[:, idx + 2] = c1 * x
            idx += 3
        elif degree == 2:
            c2a = 0.5 * _sqrt(15 / PI)
            c2b = 0.25 * _sqrt(5 / PI)
            c2c = 0.25 * _sqrt(15 / PI)
            out[:, idx] = c2a * x * y
            out[:, idx + 1] = c2a * y * z
            out[:, idx + 2] = c2b * (3 * z**2 - 1)
            out[:, idx + 3] = c2a * x * z
            out[:, idx + 4] = c2c * (x**2 - y**2)
            idx += 5
        elif degree == 3:
            c3a = 0.25 * _sqrt(35 / (2 * PI))
            c3b = 0.5 * _sqrt(105 / PI)
            c3c = 0.25 * _sqrt(21 / (2 * PI))
            c3d = 0.25 * _sqrt(7 / PI)
            c3e = 0.25 * _sqrt(105 / PI)
            out[:, idx] = c3a * y * (3 * x**2 - y**2)
            out[:, idx + 1] = c3b * x * y * z
            out[:, idx + 2] = c3c * y * (5 * z**2 - 1)
            out[:, idx + 3] = c3d * (5 * z**3 - 3 * z)
            out[:, idx + 4] = c3c * x * (5 * z**2 - 1)
            out[:, idx + 5] = c3e * (x**2 - y**2) * z
            out[:, idx + 6] = c3a * x * (x**2 - 3 * y**2)
            idx += 7
        elif degree == 4:
            c4a = 0.75 * _sqrt(35 / PI)
            c4b = 0.75 * _sqrt(35 / (2 * PI))
            c4c = 0.75 * _sqrt(5 / PI)
            c4d = 0.75 * _sqrt(5 / (2 * PI))
            c4e = 0.1875 * _sqrt(1 / PI)
            c4f = 0.375 * _sqrt(5 / PI)
            c4g = 0.1875 * _sqrt(35 / PI)
            out[:, idx] = c4a * x * y * (x**2 - y**2)
            out[:, idx + 1] = c4b * y * (3 * x**2 - y**2) * z
            out[:, idx + 2] = c4c * x * y * (7 * z**2 - 1)
            out[:, idx + 3] = c4d * y * (7 * z**3 - 3 * z)
            out[:, idx + 4] = c4e * (35 * z**4 - 30 * z**2 + 3)
            out[:, idx + 5] = c4d * x * (7 * z**3 - 3 * z)
            out[:, idx + 6] = c4f * (x**2 - y**2) * (7 * z**2 - 1)
            out[:, idx + 7] = c4b * x * (x**2 - 3 * y**2) * z
            out[:, idx + 8] = c4g * (
                x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2)
            )
            idx += 9
        else:
            raise ValueError(
                f"Tesseral closed-form supports l in [0, 4], got {degree}"
            )
    return out


def _prepare_unit_vectors(
    vecs: Tensor, *, normalize_input: bool, edge_sign: float
) -> Tuple[Tensor, Tensor, Tensor]:
    if vecs.shape[-1] != 3:
        raise ValueError(f"Input must have shape [..., 3], got {tuple(vecs.shape)}")
    if edge_sign != 1:
        vecs = edge_sign * vecs
    if normalize_input:
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
    return torch.unbind(vecs, dim=-1)


def _tesseral_spherical_harmonics(
    vecs: Tensor,
    degrees: Sequence[int],
    *,
    normalize_input: bool = True,
    edge_sign: float = 1.0,
) -> Tensor:
    x, y, z = _prepare_unit_vectors(
        vecs, normalize_input=normalize_input, edge_sign=edge_sign
    )
    return _tesseral_real_sh_xyz(x, y, z, degrees)


def _apply_spookynet_d_layout(y_tesseral: Tensor, degrees: Sequence[int]) -> Tensor:
    """Map integral tesseral blocks to SpookyNet chemistry-d ``norm`` layout.

    Currently only ``degrees == [2]`` is supported (SpookyNet edge quadrupole).
    """
    degrees = list(degrees)
    if degrees != [2]:
        raise ValueError(
            "layout='spookynet_d' currently supports degrees=[2] only, "
            f"got {degrees}"
        )
    return y_tesseral[:, list(_SPOOKYNET_D_FROM_TESSERAL_L2)] * _SPOOKYNET_D_FROM_INTEGRAL


def _e3nn_spherical_harmonics(
    vecs: Tensor,
    degrees: Sequence[int],
    *,
    normalization: SHNormalization = "component",
    normalize_input: bool = True,
    edge_sign: float = 1.0,
) -> Tensor:
    from e3nn.o3 import spherical_harmonics as e3nn_sh

    if edge_sign != 1:
        vecs = edge_sign * vecs
    degrees = list(degrees)
    # e3nn accepts a single l or a list; list concatenates in given order.
    return e3nn_sh(
        degrees if len(degrees) > 1 else degrees[0],
        vecs,
        normalize=normalize_input,
        normalization=normalization,
    )


_LAYOUT_DEFAULT_NORMALIZATION: Dict[SHLayout, SHNormalization] = {
    "so3krates": "integral",
    "spookynet_d": "norm",
    "e3nn": "component",
}


def spherical_harmonics(
    vecs: Tensor,
    degrees: Sequence[int],
    *,
    layout: SHLayout = "so3krates",
    normalization: Optional[SHNormalization] = None,
    normalize_input: bool = True,
    edge_sign: float = 1.0,
) -> Tensor:
    """Evaluate real spherical harmonics with a named layout.

    Parameters
    ----------
    vecs:
        Displacement / direction vectors ``[..., 3]``.
    degrees:
        Harmonic degrees to evaluate, in order.
    layout:
        * ``so3krates`` — tesseral integral, mlff m-order (default).
        * ``spookynet_d`` — chemistry d-orbital order + ``norm`` (``degrees=[2]``).
        * ``e3nn`` — e3nn CG basis (for Irreps / TP models).
    normalization:
        Overrides the layout default when set. For tesseral layouts other than
        the baked-in integral→norm SpookyNet map, only the native normalization
        is produced (``integral``); requesting another value raises.
    normalize_input:
        L2-normalize ``vecs`` before evaluation.
    edge_sign:
        Multiply vectors by this factor first (So3krates uses ``-1`` for
        ``-vij``).
    """
    degrees = list(degrees)
    if not degrees:
        raise ValueError("degrees must be non-empty")
    norm = normalization or _LAYOUT_DEFAULT_NORMALIZATION[layout]

    if layout == "e3nn":
        return _e3nn_spherical_harmonics(
            vecs,
            degrees,
            normalization=norm,
            normalize_input=normalize_input,
            edge_sign=edge_sign,
        )

    if layout == "so3krates":
        if norm != "integral":
            raise ValueError(
                "layout='so3krates' only provides integral normalization; "
                f"got normalization={norm!r}"
            )
        return _tesseral_spherical_harmonics(
            vecs,
            degrees,
            normalize_input=normalize_input,
            edge_sign=edge_sign,
        )

    if layout == "spookynet_d":
        if norm != "norm":
            raise ValueError(
                "layout='spookynet_d' only provides norm normalization; "
                f"got normalization={norm!r}"
            )
        y = _tesseral_spherical_harmonics(
            vecs,
            degrees,
            normalize_input=normalize_input,
            edge_sign=edge_sign,
        )
        return _apply_spookynet_d_layout(y, degrees)

    raise ValueError(f"Unknown SH layout {layout!r}")


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
