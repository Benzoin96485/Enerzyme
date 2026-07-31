"""Closed-form real spherical harmonics for So3krates-style SPHC features.

Adapted from So3krates-torch
(https://github.com/TCPUniLU/So3krates-torch, MIT license) to match mlff /
So3krates phase conventions (distinct from e3nn defaults).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import Tensor
from torch.nn import Module

PI = math.pi
_sqrt = math.sqrt


class RealSphericalHarmonics(Module):
    """Real spherical harmonics on unit vectors for degrees in ``[0, 4]``.

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
        if vecs.shape[-1] != 3:
            raise ValueError(f"Input must have shape [..., 3], got {tuple(vecs.shape)}")
        x, y, z = torch.unbind(torch.nn.functional.normalize(vecs, dim=-1), dim=-1)
        n = vecs.shape[0]
        out = torch.empty(n, self.m_tot, dtype=vecs.dtype, device=vecs.device)
        idx = 0
        for degree in self.degrees:
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
                raise ValueError(f"Unsupported degree {degree}")
        return out
