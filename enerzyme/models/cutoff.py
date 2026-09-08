"""Radial cutoffs and polynomial envelopes.

Transition functions (:func:`polynomial_transition`, :func:`bump_transition`,
…) map a distance onto ``[0, 1]`` for RBF / physics layers. Envelope modules
(:class:`PolynomialEnvelope`, :class:`DimeNetEnvelope`,
:class:`C3CutoffEnvelope`) are the same family of smooth radial windows used
inside equivariant message passing (EquiformerV3, DimeNet, DPA4).
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from torch import Tensor


def scale(cutoff_fn: Callable[[Tensor, Tensor, Tensor], Tensor]):
    def scaled_transition_fn(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        x_ = (x - cuton) / (cutoff - cuton)
        kernel = cutoff_fn(x_, zeros, ones)
        return torch.where(x_ > 0, torch.where(x_ < 1, kernel, zeros), ones)
    return scaled_transition_fn


@scale
def polynomial_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
    """
    Polynomial cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = 1 with sufficiently many smooth derivatives [1]. 
    For x >= 1, f(x) = 0. 
    For x <= 0, f(x) = 1. 

    Params:
    -----

    x: Only 0<=x<=1 should be used as input.
    
    References:
    -----
    [1] Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003.
    """
    x3 = x_ ** 3
    x4 = x3 * x_
    x5 = x4 * x_
    return 1 - 6 * x5 + 15 * x4 - 10 * x3


@scale
def bump_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
    """
    Smooth cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff [1]. 
    For x >= cutoff, f(x) = 0. 

    Params:
    -----

    x: Only 0<=x<=1 should be used as input.
    
    References:
    -----
    [1] Nat. Commun., 2021, 12, 7273.
    """
    x_ = torch.where((0 < x_) & (x_ < 1), x_, zeros)
    x2 = x_ ** 2
    return torch.exp(-x2 / (1 - x2))


def _smooth_transition(x_: Tensor, ones: Tensor) -> Tensor:
    return torch.exp(-1 / torch.where(x_ > 0, x_, ones))


@scale
def smooth_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
    fp = _smooth_transition(x_, ones)
    fm = _smooth_transition(1 - x_, ones)
    return fm / (fp + fm)


@scale
def cosine_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
    """
    Cosine cutoff on the scaled interval ``[0, 1]`` (So3krates / Eq. 17 style).
    For x <= 0 return 1; for x >= 1 return 0; else ``0.5 * (cos(pi * x) + 1)``.
    """
    return torch.where(
        x_ <= 0,
        ones,
        torch.where(x_ >= 1, zeros, 0.5 * (torch.cos(torch.pi * x_) + 1.0)),
    )


CUTOFF_REGISTER = {
    "polynomial": polynomial_transition,
    # SO3LR / So3krates-torch alias for the PhysNet polynomial envelope.
    "phys": polynomial_transition,
    "bump": bump_transition,
    "smooth": smooth_transition,
    "cosine": cosine_transition,
}
CUTOFF_KEY_TYPE = Literal["polynomial", "phys", "bump", "smooth", "cosine"]


class PolynomialEnvelope(torch.nn.Module):
    """Polynomial envelope that goes smoothly to zero at ``cutoff``.

    Distinct from PhysNet-style :func:`polynomial_transition` (this form is
    ``1 + a x^p + b x^{p+1} + c x^{p+2}`` on ``x = r / cutoff``, without the
    DimeNet ``1/x``).

    Reference:
        https://github.com/facebookresearch/fairchem (eSEN radial envelope)
        EquiformerV3 (Liao et al., 2026)
    """

    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d_scaled = distance / self.cutoff
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        outputs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        return outputs.view(-1, 1)

    def extra_repr(self) -> str:
        return f"cutoff={self.cutoff}, exponent={self.exponent}"


class DimeNetEnvelope(torch.nn.Module):
    """DimeNet polynomial envelope divided by scaled distance (``u(x)/x``).

    Official ``gasteigerjo/dimenet`` / PyG ``Envelope``: ``envelope_exponent=5``
    sets ``p = exponent + 1 = 6`` (paper Eq. 8). On scaled ``x = d / cutoff``::

        u(x)/x = 1/x + a x^{p-1} + b x^p + c x^{p+1}    (x < 1)
        u(x)/x = 0                                       (x >= 1)

    with ``a = -(p+1)(p+2)/2``, ``b = p(p+2)``, ``c = -p(p+1)/2``.
    The ``1/x`` is absorbed into spherical Bessel ``j_0 ~ sin(x)/x`` when this
    envelope multiplies ``sin(n π x)``.
    """

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.exponent = int(exponent)
        self.p = float(self.exponent + 1)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_safe = x.clamp(min=1e-12)
        x_pow_p0 = x_safe.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x_safe
        x_pow_p2 = x_pow_p1 * x_safe
        env = 1.0 / x_safe + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2
        return env * (x < 1.0).to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"exponent={self.exponent}, p={int(self.p)}"


class C3CutoffEnvelope(torch.nn.Module):
    """C³-continuous polynomial cutoff envelope ``E_p(x)``.

    For scaled distance ``x = r / rcut`` and ``u = 1 - x``::

        E_p(x) = u^4 * sum_{k=0}^{p-1} C(k+3, 3) x^k   (x < 1)
        E_p(x) = 0                                        (x >= 1)

    Default ``p=5`` gives ``E_5(x) = u^4 (1 + 4x + 10x^2 + 20x^3 + 35x^4)``.
    """

    def __init__(self, rcut: float, exponent: int = 5) -> None:
        super().__init__()
        if rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.p = int(exponent)
        coeffs = tuple(float(math.comb(k + 3, 3)) for k in range(self.p))
        self.register_buffer(
            "series_coefficients",
            torch.tensor(coeffs, dtype=torch.float64),
            persistent=False,
        )

    def forward(self, dst: Tensor) -> Tensor:
        u = ((self.rcut - dst) / self.rcut).clamp(min=0.0, max=1.0)
        x = 1.0 - u
        coeffs = self.series_coefficients.to(dtype=x.dtype, device=x.device)
        series = torch.full(
            x.shape, float(coeffs[-1].item()), dtype=x.dtype, device=x.device
        )
        for coefficient in reversed(coeffs[:-1].tolist()):
            series = coefficient + x * series
        return (u**4) * series

    def extra_repr(self) -> str:
        return f"rcut={self.rcut}, exponent={self.p}"
