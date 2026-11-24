"""Distance bases shared by message-passing blocks and YAML pipeline layers."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from ...functional import softplus_inverse
from .envelope import C3CutoffEnvelope, CUTOFF_KEY_TYPE, CUTOFF_REGISTER, polynomial_envelope
from .fourier_bessel import spherical_bessel_expansion, register_bessel_tables


class DistanceBasis(nn.Module):
    """A distance expansion with an optional precomputed cutoff."""

    def _init_radial_module(self, apply_cutoff_fn: bool) -> None:
        super().__init__()

    def forward(self, distance: Tensor, cutoff_values: Optional[Tensor] = None) -> Tensor:
        return self.get_rbf(distance, cutoff_values)

    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        cutoff_fn: Optional[CUTOFF_KEY_TYPE] = None,
        apply_cutoff_fn: bool = True,
    ) -> None:
        self._init_radial_module(apply_cutoff_fn)
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr
        self.apply_cutoff_fn = apply_cutoff_fn
        if apply_cutoff_fn:
            if cutoff_fn is None:
                raise TypeError("cutoff_fn is required when apply_cutoff_fn=True")
            self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        else:
            self.cutoff_fn = (
                CUTOFF_REGISTER[cutoff_fn] if cutoff_fn is not None else None
            )

    def get_rbf(self, Dij_sr: Tensor, cutoff_values_sr: Optional[Tensor]=None, **kwargs) -> Tensor:
        rbf = self._get_rbf(Dij_sr)
        if not self.apply_cutoff_fn:
            return rbf
        if cutoff_values_sr is None:
            cuton = getattr(self, "cuton", 0.0)
            cutoff_values_sr = self.cutoff_fn(Dij_sr, self.cutoff_sr, cuton)
        return cutoff_values_sr.view(-1, 1) * rbf

    @abstractmethod
    def _get_rbf(self, Dij: Tensor) -> Tensor:
        ...


def alignn_gaussian_terms(
    vmin: float,
    vmax: float,
    bins: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Centers and coefficient of the ALIGNN Gaussian expansion.

    The released ``RBFExpansion`` uses ``γ = 1/Δ`` with ``Δ`` the spacing of
    ``linspace(vmin, vmax, bins)``, then ``exp(-γ (x-μ)²)``. Passing an
    explicit length scale in that module switches to ``1/Δ²``; ALIGNN does
    not. Both the bond expansion (``0`` to the cutoff) and the bond-angle
    cosine expansion (``-1`` to ``1``) use this default.
    """
    if int(bins) < 2:
        raise ValueError(f"ALIGNN Gaussian expansion needs at least 2 bins, got {bins}")
    if float(vmax) <= float(vmin):
        raise ValueError(f"vmax must exceed vmin, got vmin={vmin}, vmax={vmax}")
    centers = torch.linspace(float(vmin), float(vmax), int(bins), dtype=dtype, device=device)
    delta = (centers[1] - centers[0]).item()
    coeff = torch.tensor(-1.0 / delta, dtype=centers.dtype, device=centers.device)
    return centers, coeff


def spinconv_gaussian_terms(
    cutoff: float,
    bins: int,
    basis_width_scalar: float = 3.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Centers and coefficient of the SpinConv Gaussian expansion.

    Means are ``linspace(0, cutoff, bins)``. The released
    ``GaussianSmearing`` uses ``σ = basis_width_scalar · Δ`` with
    ``Δ = cutoff / (bins - 1)``, then ``exp(-½ (r-μ)² / σ²)``. The OC20
    config sets ``basis_width_scalar`` to 3, which is the paper's
    "three times the distance between Gaussian means". SchNet is the same
    formula with the scalar equal to 1.
    """
    if int(bins) < 2:
        raise ValueError(f"SpinConv Gaussian expansion needs at least 2 bins, got {bins}")
    if float(cutoff) <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if float(basis_width_scalar) <= 0:
        raise ValueError(
            f"basis_width_scalar must be positive, got {basis_width_scalar}"
        )
    centers = torch.linspace(0, float(cutoff), int(bins), dtype=dtype, device=device)
    delta = (centers[1] - centers[0]).item()
    coeff = torch.tensor(
        -0.5 / (float(basis_width_scalar) * delta) ** 2,
        dtype=centers.dtype,
        device=centers.device,
    )
    return centers, coeff


def gaussian_expansion(distance: Tensor, center: Tensor, coeff: Tensor | float) -> Tensor:
    """``exp(coeff · (r - μ)²)`` for every center."""
    return torch.exp(coeff * (distance.reshape(-1, 1) - center) ** 2)


def gaussian_grid(start: float, stop: float, count: int, width: float) -> tuple[Tensor, float]:
    """SchNet/fairchem offsets and Python-precision coefficient on the native grid."""
    if count < 2 or width <= 0 or start == stop:
        raise ValueError("Gaussian smearing requires two distinct centers and a positive width")
    offset = torch.linspace(start, stop, count)
    coeff = -0.5 / (width * (offset[1] - offset[0])).item() ** 2
    return offset, coeff


class GaussianSmearing(nn.Module):
    """Fairchem Gaussian grid, optionally retaining the atom/fragment dimensions."""

    def __init__(
        self, start: float = -5.0, stop: float = 5.0, num_gaussians: int = 50,
        basis_width_scalar: float = 1.0, *, preserve_shape: bool = False,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        self.preserve_shape = preserve_shape
        offset, self.coeff = gaussian_grid(start, stop, num_gaussians, basis_width_scalar)
        self.register_buffer("offset", offset)

    def forward(self, distance: Tensor) -> Tensor:
        values = gaussian_expansion(distance, self.offset, self.coeff)
        if self.preserve_shape:
            return values.reshape(*distance.shape, self.num_output)
        return values


def gemnet_oc_gaussian_terms(
    num_rbf: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Centers and coefficient of the GemNet-OC Gaussian on a scaled distance.

    Centers are ``linspace(0, 1, num_rbf)``. The width is the SchNet width on
    that unit interval, ``coeff = -0.5 / Δ²`` with ``Δ = 1 / (num_rbf - 1)``.
    The caller scales ``r / cutoff`` before the expansion and multiplies the
    polynomial envelope. This is the radial basis in the OC20 GemNet-OC config.
    """
    if int(num_rbf) < 2:
        raise ValueError(f"GemNet-OC Gaussian needs num_rbf >= 2, got {num_rbf}")
    center = torch.linspace(0, 1, int(num_rbf), dtype=dtype, device=device)
    delta = (center[1] - center[0]).item()
    coeff = torch.tensor(-0.5 / delta**2, dtype=center.dtype, device=center.device)
    return center, coeff


def gemnet_oc_gaussian(
    distance: Tensor,
    cutoff: float,
    center: Tensor,
    coeff: Tensor,
    exponent: int,
) -> Tensor:
    """``u_p(r/c) · exp(-½ ((r/c - μ) / Δ)²)`` with centers on ``[0, 1]``.

    ``u_p`` is :func:`enerzyme.models.mp.basis.envelope.polynomial_envelope`. Distances
    at or beyond ``cutoff`` are zero. An empty distance returns ``(0, num_rbf)``.
    """
    from .envelope import polynomial_envelope

    width = int(center.shape[0])
    if distance.numel() == 0:
        return distance.new_empty(0, width)
    if float(cutoff) <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    scaled = distance / float(cutoff)
    basis = gaussian_expansion(
        scaled,
        center.to(dtype=distance.dtype, device=distance.device),
        coeff.to(dtype=distance.dtype, device=distance.device),
    )
    envelope = polynomial_envelope(distance, float(cutoff), int(exponent))
    return envelope.unsqueeze(-1) * basis


class GaussianBasis(DistanceBasis):
    """
    Radial basis functions based on Gaussian functions:

    .. math:: g_i(x) = \\exp(\\mathtt{coeff}\\cdot(x-\\mathtt{center}_i)^2)

    Here, :math:`i` takes values from :math:`0` to :math:`\\mathtt{num\\_rbf}-1`.
    Centers span :math:`[0, \\mathtt{cutoff}]`. ``flavor`` only selects how
    ``coeff`` (Gaussian width) is initialized:

    * ``PhysNet`` (default): ``coeff = -num_rbf / cutoff``
    * ``SchNet``: ``coeff = -0.5 / Δ²`` with ``Δ = cutoff / (num_rbf - 1)``
      (PyG Gaussian smearing convention)
    * ``esen``: fairchem eSEN ``GaussianSmearing`` with ``basis_width_scalar=2``,
      ``coeff = -0.5 / (2Δ)²``. The message envelope lives in the eSEN Core, so
      the published stack sets ``apply_cutoff_fn: false``.
    * ``alignn``: Choudhary and DeCost, ``exp(-(r-μ)² / Δ)`` with
      ``Δ = cutoff / (num_rbf - 1)`` and ``μ`` evenly spaced on ``[0, cutoff]``.
      The published width is ``1/Δ``, not the SchNet ``1/(2Δ²)``. The published
      expansion has no cutoff envelope (``apply_cutoff_fn: false``).
    * ``spinconv``: Shuaibi et al., ``exp(-½ (r-μ)² / σ²)`` with
      ``σ = basis_width_scalar · Δ`` and ``Δ = cutoff / (num_rbf - 1)``.
      The OC20 config uses ``basis_width_scalar: 3``. The scalar is ignored
      by the other flavors. The published expansion has no cutoff envelope
      (``apply_cutoff_fn: false``).
    * ``gemnet-oc``: Gasteiger et al., TMLR 2022. Centers lie on the scaled
      distance ``r / cutoff`` in ``[0, 1]``, with the SchNet width on that
      interval. The polynomial envelope of order ``envelope_exponent``
      (default 5) is inside the expansion, so ``apply_cutoff_fn`` is false.
      :meth:`expansion` evaluates the same basis at another cutoff. GemNet-OC
      uses that for the interaction hierarchy.

    Cutoff envelope application is controlled separately by ``apply_cutoff_fn``.
    """

    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        cutoff_fn: Optional[CUTOFF_KEY_TYPE] = "bump",
        apply_cutoff_fn: bool = True,
        flavor: Literal[
            "PhysNet", "SchNet", "esen", "alignn", "spinconv", "gemnet-oc"
        ] = "PhysNet",
        basis_width_scalar: Optional[float] = None,
        envelope_exponent: int = 5,
    ) -> None:
        if apply_cutoff_fn and cutoff_fn is None:
            cutoff_fn = "bump"
        super().__init__(num_rbf, cutoff_sr, cutoff_fn, apply_cutoff_fn=apply_cutoff_fn)
        self.flavor = flavor
        self.basis_width_scalar = None
        if flavor == "PhysNet":
            # float64 buffers match historical SpookyNet / PhysNet parity.
            self.register_buffer(
                "center",
                torch.linspace(0, cutoff_sr, num_rbf, dtype=torch.float64),
            )
            width = torch.tensor(num_rbf / cutoff_sr, dtype=torch.float64)
            self.register_buffer("width", width)
            self.register_buffer("coeff", -width)
        elif flavor == "SchNet":
            # Default dtype matches PyG GaussianSmearing offsets / coeff.
            center = torch.linspace(0, cutoff_sr, num_rbf)
            self.register_buffer("center", center)
            coeff = torch.tensor(
                -0.5 / (center[1] - center[0]).item() ** 2
            )
            self.register_buffer("coeff", coeff)
            self.register_buffer("width", -coeff)
        elif flavor == "esen":
            if num_rbf < 2:
                raise ValueError(
                    f"esen Gaussian flavor needs num_rbf >= 2, got {num_rbf}"
                )
            center = torch.linspace(0, cutoff_sr, num_rbf)
            self.register_buffer("center", center)
            delta = (center[1] - center[0]).item()
            # fairchem GaussianSmearing(..., basis_width_scalar=2.0)
            coeff = torch.tensor(-0.5 / (2.0 * delta) ** 2)
            self.register_buffer("coeff", coeff)
            self.register_buffer("width", -coeff)
        elif flavor == "alignn":
            center, coeff = alignn_gaussian_terms(0.0, cutoff_sr, num_rbf)
            self.register_buffer("center", center)
            self.register_buffer("coeff", coeff)
            self.register_buffer("width", -coeff)
        elif flavor == "spinconv":
            scalar = 3.0 if basis_width_scalar is None else float(basis_width_scalar)
            center, coeff = spinconv_gaussian_terms(cutoff_sr, num_rbf, scalar)
            self.basis_width_scalar = scalar
            self.register_buffer("center", center)
            self.register_buffer("coeff", coeff)
            self.register_buffer("width", -coeff)
        elif flavor == "gemnet-oc":
            if apply_cutoff_fn:
                raise ValueError(
                    "GaussianRBF flavor 'gemnet-oc' multiplies the polynomial "
                    "envelope inside the expansion. Set apply_cutoff_fn: false."
                )
            if int(envelope_exponent) < 1:
                raise ValueError(
                    f"envelope_exponent must be >= 1, got {envelope_exponent}"
                )
            center, coeff = gemnet_oc_gaussian_terms(num_rbf)
            self.envelope_exponent = int(envelope_exponent)
            self.register_buffer("center", center)
            self.register_buffer("coeff", coeff)
            self.register_buffer("width", -coeff)
        else:
            raise ValueError(
                "Unknown GaussianRBF flavor "
                f"{flavor!r}; expected 'PhysNet', 'SchNet', 'esen', 'alignn', "
                "'spinconv', or 'gemnet-oc'"
            )
        if flavor != "gemnet-oc" and int(envelope_exponent) != 5:
            raise ValueError(
                "envelope_exponent applies to GaussianRBF flavor 'gemnet-oc', "
                f"got flavor {flavor!r}"
            )
        if flavor != "spinconv" and basis_width_scalar is not None:
            raise ValueError(
                "basis_width_scalar is the SpinConv width σ / Δ and applies "
                f"only to flavor 'spinconv', got flavor {flavor!r}"
            )

    def _get_rbf(self, Dij: Tensor) -> Tensor:
        """
        Evaluates radial basis functions given distances.
        N: Number of input values.
        num_basis_functions: Number of radial basis functions.

        Arguments:
            Dij (FloatTensor [N]):
                Input distances.

        Returns:
            rbf (FloatTensor [N, num_basis_functions]):
                Values of the radial basis functions for the distances r.
        """
        if self.flavor == "gemnet-oc":
            return gemnet_oc_gaussian(
                Dij, self.cutoff_sr, self.center, self.coeff, self.envelope_exponent
            )
        return gaussian_expansion(Dij, self.center, self.coeff)

    def expansion(self, distance: Tensor, cutoff: float) -> Tensor:
        """GemNet-OC Gaussian at ``cutoff``, using this layer's centers.

        The centers live on the scaled interval ``[0, 1]``. A shorter
        interaction cutoff therefore uses the same coefficients and a
        different envelope. Only flavor ``gemnet-oc`` implements this.
        """
        if self.flavor != "gemnet-oc":
            raise ValueError(
                "GaussianRBF.expansion is the GemNet-OC scaled Gaussian, "
                f"got flavor {self.flavor!r}"
            )
        return gemnet_oc_gaussian(
            distance, cutoff, self.center, self.coeff, self.envelope_exponent
        )


class BernsteinBasis(DistanceBasis):
    """
    Radial basis functions based on Bernstein polynomials given by:
    b_{v,n}(x) = (n over v) * (x/cutoff)**v * (1-(x/cutoff))**(n-v)
    (see https://en.wikipedia.org/wiki/Bernstein_polynomial)
    Here, n = num_basis_functions-1 and v takes values from 0 to n. The basis
    functions are placed to optimally cover the range x = 0...cutoff.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_rbf: int, cutoff_sr: float, cutoff_fn: CUTOFF_KEY_TYPE = "bump") -> None:
        """ Initializes the BernsteinPolynomials class. """
        super().__init__(num_rbf, cutoff_sr, cutoff_fn)
        # compute values to initialize buffers
        from ...special import get_berstein_coefficient
        v, n, logc = get_berstein_coefficient(self.num_rbf)
        # register buffers and parameters
        self.register_buffer("logc", torch.tensor(logc, dtype=torch.float64))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float64))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float64))

    def _get_rbf(self, r: Tensor) -> Tensor:
        """
        Evaluates radial basis functions given distances and the corresponding
        values of a cutoff function (must be consistent with cutoff value
        passed at initialization).
        N: Number of input values.
        num_basis_functions: Number of radial basis functions.

        Arguments:
            Dij (FloatTensor [N]):
                Input distances.

        Returns:
            rbf (FloatTensor [N, num_basis_functions]):
                Values of the radial basis functions for the distances r.
        """
        x = r.view(-1, 1) / self.cutoff_sr
        x = torch.where(x < 1.0, x, 0.5 * torch.ones_like(x))  # prevent nans
        x = torch.log(x)
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        return torch.exp(x)


class SincBasis(DistanceBasis):
    """
    Radial basis functions based on sinc functions given by:
    g_i(x) = sinc((i+1)*x/cutoff)
    Here, i takes values from 0 to num_basis_functions-1.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_rbf: int, cutoff_sr: float, cutoff_fn: Literal["polynomial", "bump"]="bump") -> None:
        """ Initializes the SincFunctions class. """
        super().__init__(num_rbf, cutoff_sr, cutoff_fn)
        self.register_buffer(
            "factor", torch.linspace(1, num_rbf, num_rbf, dtype=torch.float64) / cutoff_sr,
        )
        try:
            from torch import sinc
        except ImportError:
            from ...special import sinc
        self.sinc = sinc

    def _get_rbf(self, Dij: Tensor) -> Tensor:
        """
        Evaluates radial basis functions given distances and the corresponding
        values of a cutoff function (must be consistent with cutoff value
        passed at initialization).
        N: Number of input values.
        num_basis_functions: Number of radial basis functions.

        Arguments:
            Dij (FloatTensor [N]):
                Input distances.

        Returns:
            rbf (FloatTensor [N, num_basis_functions]):
                Values of the radial basis functions for the distances r.
        """
        x = self.factor * Dij.view(-1, 1)
        return self.sinc(x)


class ExponentialBasis(DistanceBasis):
    def __init__(
        self,
        num_rbf: int,
        no_basis_at_infinity: bool=False,
        init_alpha: float=0.9448630629184640,
        exp_weighting: bool=False,
        learnable_shape: bool=True,
        cutoff_sr: float=float("inf"),
        cutoff_fn: Literal["polynomial", "bump"]="bump"
    ) -> None:
        '''
        The base class of radial basis functions with a general exponential form.
        It entails the physical knowledge that
        bound state wave functions in two-body systems decay exponentially. [1,2,3]

        RBF(r; alpha) = cutoff_fn(r) * exp(inner_fn(r; alpha)) * (exp(-alpha*r) if exp_weighting)

        Params:
        ----------
        num_basis_functions: Number of radial basis functions.

        no_basis_function_at_infinity: If True, no basis function is put at exp(-alpha*x) = 0, i.e.
        x = infinity.

        init_alpha: Initial value for scaling parameter alpha (Default value corresponds
        to 0.5 1/Bohr converted to 1/Angstrom).

        exp_weighting: If `True`, basis functions are weighted with a factor exp(-alpha*r).

        learnable_shape: If `True`, shape parameters of exponentials are learnable.

        cutoff: Short range cutoff threshold for radial base functions.

        cutoff_fn: Short range cutoff function, whose are called by `cutoff_fn(x, cutoff=cutoff)`
        where x is the distance.

        References:
        ----------
        [1] Commun. Math. Phys. 1973, 32, 319−340.

        [2] J. Chem. Theory Comput. 2019, 15, 3678−3693.

        [3] Nat. Chem. 2020, 12, 891–897.

        '''
        super().__init__(num_rbf, cutoff_sr, cutoff_fn)
        self.exp_weighting = exp_weighting
        self.learnable_shape = learnable_shape
        self.no_basis_at_infinity = no_basis_at_infinity
        self.register_parameter(
            "alpha", Parameter(softplus_inverse(torch.tensor(init_alpha, dtype=torch.float64)))
        )

    @abstractmethod
    def inner_fn(self, alphar: Optional[Tensor]=None, expalphar: Optional[Tensor]=None) -> Tensor:
        ...

    def _get_rbf(self, Dij: Tensor) -> Tensor:
        '''
        Evaluate the RBF values

        Params:
        ----------
        r: Float tensor of distances, shape [M]

        cutoff_values: Float tensor of pre-calculated cutoff distances, shape [M].
        If not provided, the cutoff distances are calculated by `cutoff_fn`.

        Returns:
        ----------
        rbf: Float tensor of RBFs, shape [M, `num_basis_functions`]
        '''
        alphar = -F.softplus(self.alpha) * Dij.view(-1, 1)
        expalphar = torch.exp(alphar)
        return torch.exp(self.inner_fn(alphar, expalphar)) * (expalphar if self.exp_weighting else 1)


class ExponentialGaussianBasis(ExponentialBasis):
    def __init__(
        self,
        num_rbf: int,
        no_basis_at_infinity: bool=False,
        init_alpha: float=0.9448630629184640,
        exp_weighting: bool=False,
        learnable_shape: bool=True,
        cutoff_sr: float=float("inf"),
        cutoff_fn: Literal["polynomial", "bump"]="polynomial",
        init_width_flavor: Literal["PhysNet", "SpookyNet"]="PhysNet"
    ) -> None:
        r'''
        Radial basis functions based on exponential Gaussian functions given by:

        g_i(x) = exp(-width_i*(exp(-alpha*x)-center_i)**2)

        Params:
        ----------
        num_basis_functions: Number of radial basis functions.

        dtype: Data type of floating numbers.

        no_basis_function_at_infinity: If True, no basis function is put at exp(-alpha*x) = 0, i.e.
        x = infinity.

        init_alpha: Initial value for scaling parameter alpha (Default value corresponds
        to 0.5 1/Bohr converted to 1/Angstrom).

        init_width_flavor: Initialization flavor for width of the exponentials. Options:

        - `PhysNet`: A constant number (2K^{-1}(1-\exp(-`cutoff`)))^{-2}, where K is `num_basis_functions`
        - `SpookyNet`: A constant number K or K+1 (`no_basis_function_at_infinity=True`)

        exp_weighting: If `True`, basis functions are weighted with a factor exp(-alpha*r).

        learnable_shape: If `True`, centers and widths of exponentials are learnable.

        cutoff: Short range cutoff threshold for radial base functions.

        cutoff_fn: Short range cutoff function, whose are called by `cutoff_fn(x, cutoff=cutoff)`
        where x is the distance.

        init_width_flavor: Initialization flavor for width of the exponentials. Options:

        - `PhysNet`: A constant number (2K^{-1}(1-\exp(-`cutoff`)))^{-2}, where K is `num_basis_functions` [1].
        - `SpookyNet`: A constant number K or K+1 (`no_basis_function_at_infinity=True`).

        References:
        ----------
        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        '''
        super().__init__(
            num_rbf=num_rbf,
            no_basis_at_infinity=no_basis_at_infinity,
            init_alpha=init_alpha,
            exp_weighting=exp_weighting,
            learnable_shape=learnable_shape,
            cutoff_sr=cutoff_sr,
            cutoff_fn=cutoff_fn
        )
        if cutoff_sr == float("inf") and no_basis_at_infinity:
            self.register_parameter(
                "centers", Parameter(
                    softplus_inverse(torch.linspace(
                        1, 0, num_rbf + 1, dtype=torch.float64
                    )[:-1]),
                    requires_grad=self.learnable_shape
                )
            )
        else:
            self.register_parameter(
                "centers", Parameter(
                    softplus_inverse(torch.linspace(
                        1, math.exp(-cutoff_sr), num_rbf, dtype=torch.float64
                    )),
                    requires_grad=self.learnable_shape
                )
            )
        self._init_width(init_width_flavor)

    def _init_width(self, init_width_flavor: Literal["PhysNet", "SpookyNet"]="PhysNet") -> None:
        '''
        Initialize the widths of exponentials based on the chosen flavor.
        '''
        if init_width_flavor == "SpookyNet":
            self.register_parameter(
                "widths", Parameter(
                    softplus_inverse(
                        torch.tensor(
                            1.0 * self.num_rbf + \
                            int(self.no_basis_at_infinity), dtype=torch.float64
                        )
                    ),
                    requires_grad=self.learnable_shape
                )
            )
        elif init_width_flavor == "PhysNet":
            self.register_parameter(
                "widths", Parameter(
                    softplus_inverse(
                        torch.tensor(
                            [(0.5 / ((-math.expm1(-self.cutoff_sr)) / self.num_rbf)) ** 2] * \
                            self.num_rbf, dtype=torch.float64
                        )
                    ),
                    requires_grad=self.learnable_shape
                )
            )

    def inner_fn(self, alphar: Tensor, expalphar: Tensor) -> torch.Tensor:
        return -F.softplus(self.widths) * (expalphar - F.softplus(self.centers)) ** 2


class ExponentialBernsteinBasis(ExponentialBasis):
    def __init__(
        self,
        num_rbf: int,
        no_basis_at_infinity: bool=False,
        init_alpha: float=0.9448630629184640,
        exp_weighting: bool=False,
        learnable_shape: bool=True,
        cutoff_sr: float=float("inf"),
        cutoff_fn: Literal["polynomial", "bump"]="bump",
    ) -> None:
        '''
        Radial basis functions based on exponential Bernstein polynomials given by:
        b_{v,n}(x) = (n over v) * exp(-alpha*x)**v * (1-exp(-alpha*x))**(n-v)
        (see https://en.wikipedia.org/wiki/Bernstein_polynomial)

        For n to infinity, linear combination of b_{v,n}s can approximate
        any continuous function on the interval [0, 1] uniformly [1].

        NOTE: There is a problem for x = 0, as log(-expm1(0)) will be log(0) = -inf.
        This itself is not an issue, but the buffer v contains an entry 0 and
        0*(-inf)=nan. The correct behaviour could be recovered by replacing the nan
        with 0.0, but should not be necessary because issues are only present when
        r = 0, which will not occur with chemically meaningful inputs.

        References:
        ----------
        [1] Commun. Kharkov Math. Soc. 1912, 13, 1.
        '''
        super().__init__(
            num_rbf=num_rbf,
            no_basis_at_infinity=no_basis_at_infinity,
            init_alpha=init_alpha,
            exp_weighting=exp_weighting,
            learnable_shape=learnable_shape,
            cutoff_sr=cutoff_sr,
            cutoff_fn=cutoff_fn
        )
        from ...special import get_berstein_coefficient
        self.num_rbf += int(no_basis_at_infinity)
        v, n, logc = get_berstein_coefficient(self.num_rbf)
        if no_basis_at_infinity:  # remove last basis function at infinity
            v = v[:-1]
            n = n[:-1]
            logc = logc[:-1]
        self.register_buffer("logc", torch.tensor(logc))
        self.register_buffer("n", torch.tensor(n))
        self.register_buffer("v", torch.tensor(v))

    def inner_fn(self, alphar: Tensor, expalphar: Tensor) -> Tensor:
        return self.logc + self.n * alphar + self.v * torch.log(-torch.expm1(alphar))


class BesselBasis(DistanceBasis):
    """Spherical Bessel radial basis (``l = 0``).

    ``flavor="default"`` (TACE / eSCN / EquiformerV3 stacks):

    .. math:: e_n(r) = \\sqrt{2/c}\\,\\sin(n\\pi r / c) / r

    optionally multiplied by :code:`cutoff_fn` (PhysNet polynomial by default).

    ``flavor="dimenet"`` matches Gasteiger et al. / ``gasteigerjo/dimenet``:
    frequencies live on scaled distance ``x = r/c``, are trainable by default,
    and the DimeNet ``u(x)/x`` envelope replaces a separate cutoff::

        e_n(r) = Envelope(x) * sin(ω_n x),   ω_n = n π

    ``flavor="painn"`` is the PaiNN / SchNetPack Bessel basis (no
    :math:`\\sqrt{2/c}` prefactor). The cosine cutoff belongs on the
    continuous-filter, not on this expansion, so ``apply_cutoff_fn`` defaults
    to false::

        e_n(r) = sin(n π r / c) / r

    ``flavor="equflash"`` is the SevenNet / EquFlash basis. Frequencies are
    trainable by default, and the radial window (XPLOR in the published
    stack) multiplies the basis::

        e_n(r) = (2/c) sin(n π r / c) / r

    ``flavor="nequip"`` is the maintained `mir-group/nequip` edge embedding:
    normalized sinc Bessel functions, the DimeNet / NequIP polynomial envelope
    of order ``polynomial_p`` (default 6), and the constant
    :math:`2\\pi/c^2` applied in ``FullNequIPGNNModel``. Frequencies are the
    integers :math:`n = 1\\ldots N` and are trainable when ``trainable`` is
    true. The envelope is inside this layer, so ``apply_cutoff_fn`` is forced
    off::

        x = r / c
        e_n(r) = (2π / c²) · n · sinc(n x) · u_p(x)

    ``flavor="esen"`` is fairchem eSEN ``EnvelopedBesselBasis``: trainable
    frequencies :math:`ω_n = nπ` on scaled distance, times the polynomial
    envelope of order 5 and :math:`\\sqrt{2/c^3}`::

        x = r / c
        e_n(r) = u_5(x) · √(2/c³) · sin(ω_n x) / x

    The envelope is inside this layer (``apply_cutoff_fn`` is forced off).
    eSEN still multiplies a second envelope on the aggregated edge message.

    ``flavor="gemnet"`` is the GemNet radial basis (Gasteiger et al., NeurIPS
    2021). The envelope power is ``envelope_exponent`` itself (default 5),
    not the DimeNet shift ``p = exponent + 1``. ``basis_cutoff`` is the
    envelope cutoff and may be smaller than the neighbor-list ``cutoff_sr``
    (GemNet-Q keeps interaction edges out to ``cutoff_sr`` while the
    embedding basis dies at 5 Å)::

        x = r / c
        e_n(r) = u_p(x) · √(2/c) · sin(n π x) / r

    The limit at ``r = 0`` is ``√(2/c) · nπ / c``.

    ``flavor="mace"`` is the MACE radial embedding (Batatia et al.): the
    default spherical Bessel ``√(2/c) sin(nπr/c) / r`` times the polynomial
    envelope of order ``polynomial_p``. The published MACE cutoff uses
    ``polynomial_p: 5``. The envelope is inside this layer, so
    ``apply_cutoff_fn`` is forced off. At ``r = 0`` the basis is
    ``√(2/c) · nπ / c``.
    """

    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        cutoff_fn: Optional[CUTOFF_KEY_TYPE] = "polynomial",
        trainable: Optional[bool] = None,
        flavor: Literal[
            "default", "dimenet", "painn", "equflash", "nequip", "esen", "gemnet", "mace"
        ] = "default",
        envelope_exponent: int = 5,
        apply_cutoff_fn: Optional[bool] = None,
        cuton: float = 0.0,
        polynomial_p: int = 6,
        basis_cutoff: Optional[float] = None,
    ) -> None:
        self.flavor = flavor
        if flavor not in (
            "default", "dimenet", "painn", "equflash", "nequip", "esen", "gemnet", "mace"
        ):
            raise ValueError(
                "Unknown BesselRBF flavor "
                f"{flavor!r}; expected 'default', 'dimenet', 'painn', "
                "'equflash', 'nequip', 'esen', 'gemnet', or 'mace'"
            )
        if basis_cutoff is not None and flavor != "gemnet":
            raise ValueError(
                "basis_cutoff is only used by BesselRBF flavor 'gemnet' "
                f"(got flavor {flavor!r})"
            )
        if trainable is None:
            trainable = flavor in ("dimenet", "equflash", "esen", "gemnet")
        if apply_cutoff_fn is None:
            # DimeNet, PaiNN, NequIP, eSEN, and GemNet each own their radial window.
            # EquFlash multiplies the XPLOR window onto this basis.
            apply_cutoff_fn = flavor in ("default", "equflash")
        if flavor in ("dimenet", "nequip", "esen", "gemnet", "mace"):
            apply_cutoff_fn = False
            cutoff_fn = None
        if flavor == "nequip" and int(polynomial_p) < 2:
            raise ValueError(
                f"NequIP polynomial_p must be >= 2, got {polynomial_p}"
            )
        self.polynomial_p = int(polynomial_p)
        super().__init__(
            num_rbf, cutoff_sr, cutoff_fn, apply_cutoff_fn=apply_cutoff_fn
        )
        self.cuton = float(cuton)
        self.envelope_exponent = int(envelope_exponent)
        dtype = torch.get_default_dtype()
        if flavor == "dimenet":
            from .envelope import DimeNetEnvelope

            self.envelope = DimeNetEnvelope(self.envelope_exponent)
            frequencies = math.pi * torch.arange(
                1, num_rbf + 1, dtype=dtype
            )
            if trainable:
                self.bessel_weights = torch.nn.Parameter(frequencies)
            else:
                self.register_buffer("bessel_weights", frequencies)
            self.register_buffer("prefactor", torch.tensor(1.0, dtype=dtype))
        elif flavor == "nequip":
            from .envelope import PolynomialEnvelope

            self.envelope = PolynomialEnvelope(cutoff_sr, self.polynomial_p)
            frequencies = torch.arange(1, num_rbf + 1, dtype=dtype)
            if trainable:
                self.bessel_weights = torch.nn.Parameter(frequencies)
            else:
                self.register_buffer("bessel_weights", frequencies)
            self.register_buffer(
                "prefactor",
                torch.tensor(2.0 * math.pi / (cutoff_sr * cutoff_sr), dtype=dtype),
            )
        elif flavor == "esen":
            from .envelope import polynomial_envelope

            self.envelope = None
            frequencies = math.pi * torch.arange(1, num_rbf + 1, dtype=dtype)
            if trainable:
                self.bessel_weights = torch.nn.Parameter(frequencies)
            else:
                self.register_buffer("bessel_weights", frequencies)
            self.register_buffer(
                "prefactor",
                torch.tensor(math.sqrt(2.0 / cutoff_sr**3), dtype=dtype),
            )
            self._esen_envelope = polynomial_envelope
        elif flavor == "gemnet":
            from .envelope import polynomial_envelope

            basis_c = float(cutoff_sr if basis_cutoff is None else basis_cutoff)
            if basis_c <= 0:
                raise ValueError(f"basis_cutoff must be positive, got {basis_c}")
            self.basis_cutoff = basis_c
            self.envelope = None
            frequencies = math.pi * torch.arange(1, num_rbf + 1, dtype=dtype)
            if trainable:
                self.bessel_weights = torch.nn.Parameter(frequencies)
            else:
                self.register_buffer("bessel_weights", frequencies)
            self.register_buffer(
                "prefactor",
                torch.tensor(math.sqrt(2.0 / basis_c), dtype=dtype),
            )
            self._gemnet_envelope = polynomial_envelope
        else:
            self.envelope = None
            bessel_weights = (
                np.pi
                / cutoff_sr
                * torch.linspace(
                    start=1.0,
                    end=num_rbf,
                    steps=num_rbf,
                    dtype=dtype,
                )
            )
            if trainable:
                self.bessel_weights = torch.nn.Parameter(bessel_weights)
            else:
                self.register_buffer("bessel_weights", bessel_weights)
            if flavor == "painn":
                prefactor = 1.0
            elif flavor == "equflash":
                prefactor = 2.0 / cutoff_sr
            else:
                prefactor = np.sqrt(2.0 / cutoff_sr)
            self.register_buffer(
                "prefactor",
                torch.tensor(prefactor, dtype=dtype),
            )

    def _get_rbf(self, x: Tensor) -> Tensor:  # [..., 1]
        dist = x.view(-1, 1)
        if self.flavor == "dimenet":
            d_scaled = dist / self.cutoff_sr
            return self.envelope(d_scaled) * torch.sin(self.bessel_weights * d_scaled)
        if self.flavor == "nequip":
            # x = r/c, e_n = (2π/c²) * n * sinc(n x) * polynomial envelope.
            scaled = dist / self.cutoff_sr
            weights = self.bessel_weights.view(1, -1)
            bessel = torch.sinc(scaled * weights) * weights
            return self.prefactor * bessel * self.envelope(dist)
        if self.flavor == "esen":
            # x = r/c, e_n = u_5(x) * sqrt(2/c^3) * sin(n π x) / x
            # Limit at x = 0 is u(0) * sqrt(2/c^3) * n π.
            scaled = dist / self.cutoff_sr
            env = self._esen_envelope(scaled, 1.0, self.envelope_exponent)
            weights = self.bessel_weights.view(1, -1)
            arg = scaled * weights
            safe = scaled.clamp_min(1e-12)
            quotient = torch.where(scaled == 0, weights, torch.sin(arg) / safe)
            return env * self.prefactor * quotient
        if self.flavor == "mace":
            # √(2/c) sin(nπr/c) / r × polynomial envelope. Limit nπ√(2/c) / c.
            from .envelope import polynomial_envelope

            env = polynomial_envelope(dist, self.cutoff_sr, self.polynomial_p)
            weights = self.bessel_weights.view(1, -1)
            arg = dist * weights
            safe = dist.clamp_min(1e-12)
            quotient = torch.where(dist == 0, weights, torch.sin(arg) / safe)
            return self.prefactor * quotient * env
        if self.flavor == "gemnet":
            # x = r/c, e_n = u_p(x) * sqrt(2/c) * sin(n π x) / r
            # Limit at r = 0 is sqrt(2/c) * n π / c.
            cutoff = self.basis_cutoff
            scaled = dist / cutoff
            env = self._gemnet_envelope(dist, cutoff, self.envelope_exponent)
            weights = self.bessel_weights.view(1, -1)
            arg = scaled * weights
            safe = dist.clamp_min(1e-12)
            quotient = torch.where(
                dist == 0, weights / cutoff, torch.sin(arg) / safe
            )
            return env * self.prefactor * quotient
        if self.flavor in ("painn", "equflash"):
            # SchNetPack BesselRBF: r = 0 contributes 0 (divide by 1).
            # EquFlash uses the same quotient times ``2/c``. At r = 0 the
            # analytic limit is ``prefactor * nπ/c`` so a contact does not NaN.
            flat = dist.view(-1)
            safe = torch.where(flat == 0, torch.ones_like(flat), flat).unsqueeze(-1)
            values = torch.sin(flat.unsqueeze(-1) * self.bessel_weights) / safe
            if self.flavor == "equflash":
                limit = self.prefactor * self.bessel_weights.view(1, -1)
                values = self.prefactor * values
                values = torch.where(flat.view(-1, 1) == 0, limit, values)
            return values
        numerator = torch.sin(self.bessel_weights.view(1, -1) * dist)
        return self.prefactor * (numerator / dist)


class ExpNormalBasis(DistanceBasis):
    """Exponential normal radial basis (TorchMD-Net / Equiformer MD17).

    Includes an internal cosine cutoff; pass raw short-range distances ``Dij_sr``.
    ``alpha`` scales the exponent ``exp(-alpha * r)``. The TorchMD-Net default is
    ``5 / (cutoff - cuton)``. LEFTNet uses ``alpha=1`` (plain ``exp(-r)``). At a
    5 Å cutoff and ``cuton=0`` the two choices coincide.
    """

    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        cuton: float = 0.0,
        trainable: bool = False,
        alpha: Optional[float] = None,
    ) -> None:
        super().__init__(num_rbf, cutoff_sr, apply_cutoff_fn=False)
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr
        self.cuton = cuton
        self.trainable = trainable
        if alpha is None:
            self.alpha = 5.0 / (cutoff_sr - cuton)
        else:
            if float(alpha) <= 0.0:
                raise ValueError(f"alpha must be positive, got {alpha}")
            self.alpha = float(alpha)
        means, betas = self._initial_params()
        if trainable:
            self.means = Parameter(means)
            self.betas = Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.tensor(-self.cutoff_sr + self.cuton))
        means = torch.linspace(start_value.item(), 1.0, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value.item())) ** -2] * self.num_rbf
        )
        return means, betas

    def _cosine_cutoff(self, distances: Tensor) -> Tensor:
        if self.cuton > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cuton)
                        / (self.cutoff_sr - self.cuton)
                        + 1.0
                    )
                )
                + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_sr).to(distances.dtype)
            cutoffs = cutoffs * (distances > self.cuton).to(distances.dtype)
            return cutoffs
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_sr) + 1.0)
        return cutoffs * (distances < self.cutoff_sr).to(distances.dtype)

    def _get_rbf(self, Dij_sr: Tensor) -> Tensor:
        dist = Dij_sr.view(-1, 1)
        return self._cosine_cutoff(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cuton)) - self.means) ** 2
        )


class BesselC3RadialBasis(nn.Module):
    """Bessel or Gaussian radial basis, optionally multiplied by a C³ envelope.

    Distinct from YAML ``BesselRBFLayer`` (PhysNet-style ``sin(wr)/r`` ×
    polynomial/bump cutoff). With ``apply_envelope=True`` this matches DPA4
    EdgeCache radials (``w * sinc(w r / π)`` × :class:`C3CutoffEnvelope`).
    DPA4C sets ``apply_envelope=False`` and multiplies one explicit envelope
    onto the finished radial–chemical amplitude.
    """

    def __init__(
        self,
        rcut: float,
        n_radial: int = 16,
        basis_type: str = "bessel",
        exponent: int = 7,
        apply_envelope: bool = True,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        self.basis_type = str(basis_type).lower()
        self.apply_envelope = bool(apply_envelope)
        if self.basis_type not in ("bessel", "gaussian"):
            raise ValueError("`basis_type` must be 'bessel' or 'gaussian'")
        self.envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=int(exponent))
        if self.basis_type == "bessel":
            freqs = torch.arange(1, self.n_radial + 1, dtype=torch.float32) * (
                math.pi / self.rcut
            )
        else:
            freqs = torch.linspace(0.0, self.rcut, self.n_radial)
        self.freqs = nn.Parameter(freqs.view(1, self.n_radial))
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.gaussian_coeff = -0.5 / (gaussian_width * gaussian_width)

    def forward(self, r: Tensor) -> Tensor:
        """``r`` shape ``(E,)`` or ``(E, 1)`` → ``(E, n_radial)``."""
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        freqs = self.freqs.to(dtype=r.dtype, device=r.device)
        if self.basis_type == "bessel":
            x = r * freqs
            z = x / math.pi
            pz = math.pi * z
            sinc = torch.where(
                z == 0, torch.ones_like(pz), torch.sin(pz) / pz.clamp_min(1e-12)
            )
            raw = freqs * sinc
        else:
            dr = r - freqs
            raw = torch.exp(dr * dr * self.gaussian_coeff)
        if not self.apply_envelope:
            return raw
        return raw * self.envelope(r)


class SphericalBesselBasis(DistanceBasis):
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
        register_bessel_tables(self, self.max_l, self.num_rbf)

    def _get_rbf(self, Dij: Tensor) -> Tensor:
        return spherical_bessel_expansion(
            Dij, self.zeros, self.normalizer, self.cutoff_sr
        )


def chebyshev_first(degree: int, x: Tensor, dim: int = 0) -> Tensor:
    """``T_0, ..., T_degree`` stacked on the requested axis."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    columns = [torch.ones_like(x)]
    if degree == 0:
        return columns[0].unsqueeze(dim)
    columns.append(x)
    for _ in range(2, degree + 1):
        columns.append(2.0 * x * columns[-1] - columns[-2])
    return torch.stack(columns, dim=dim)


class SpeciesPairChebyshev(nn.Module):
    """``B_u(r, zi, zj) = envelope(r) Σ_β c[zi, zj, u, β] T_β(r / cutoff)``.

    ``zi`` and ``zj`` are compact species indices, not atomic numbers. The
    caller owns that table so several layers can share one vocabulary.
    """

    def __init__(
        self,
        n_channels: int,
        n_species: int,
        degree: int,
        cutoff: float,
        envelope_p: int,
    ) -> None:
        super().__init__()
        if int(n_channels) < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if int(n_species) < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        if int(degree) < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")
        if float(cutoff) <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if int(envelope_p) < 1:
            raise ValueError(f"envelope_p must be >= 1, got {envelope_p}")
        self.n_channels = int(n_channels)
        self.n_species = int(n_species)
        self.degree = int(degree)
        self.cutoff = float(cutoff)
        self.envelope_p = int(envelope_p)
        self.coeff = nn.Parameter(
            torch.empty(self.n_species, self.n_species, self.n_channels, self.degree + 1)
        )
        bound = (self.degree + 1) ** -0.5
        nn.init.uniform_(self.coeff, -bound, bound)

    def forward(self, distance: Tensor, zi: Tensor, zj: Tensor) -> Tensor:
        """Return ``[E, n_channels]``. Distances at or beyond the cutoff are 0."""
        if distance.shape != zi.shape or distance.shape != zj.shape:
            raise ValueError(
                "distance, zi, and zj must share a shape, "
                f"got {tuple(distance.shape)}, {tuple(zi.shape)}, {tuple(zj.shape)}"
            )
        basis = chebyshev_first(self.degree, distance / self.cutoff)
        window = polynomial_envelope(distance, self.cutoff, self.envelope_p).to(
            dtype=distance.dtype
        )
        # coeff rows: (E, C, degree + 1). basis: (degree + 1, E).
        pair = self.coeff[zi.long(), zj.long()]
        return torch.einsum("ecd,de->ec", pair, basis * window)


class ChebyshevBasis(DistanceBasis):
    """First-kind polynomials on r/c (CAMP) or 2r/c−1 (GRACE)."""

    def __init__(self, num_rbf: int, cutoff_sr: float, shifted: bool = False) -> None:
        if num_rbf < 1 or cutoff_sr <= 0:
            raise ValueError("Chebyshev expansion needs a positive count and cutoff")
        super().__init__(num_rbf, cutoff_sr, apply_cutoff_fn=False)
        self.shifted = shifted

    def _get_rbf(self, distance: Tensor) -> Tensor:
        argument = distance / self.cutoff_sr
        if self.shifted:
            argument = 2.0 * argument - 1.0
        return chebyshev_first(self.num_rbf - 1, argument, dim=-1)


_BASIS_TYPES = {
    "gaussian": GaussianBasis,
    "bessel": BesselBasis,
    "spherical_bessel": SphericalBesselBasis,
    "chebyshev": ChebyshevBasis,
    "bernstein": BernsteinBasis,
    "exp_gaussian": ExponentialGaussianBasis,
    "exp_bernstein": ExponentialBernsteinBasis,
    "sinc": SincBasis,
    "expnormal": ExpNormalBasis,
}


def RadialBasis(
    kind: str, num_rbf: int, cutoff_sr: float, *, trainable: Optional[bool] = None, **options
) -> DistanceBasis:
    """Build a distance-to-feature module with its kind's native cutoff and parameters.

    Bessel frequencies, Gaussian centers, sinc frequencies and exponential
    shape parameters can be trained. Analytic polynomial and spherical-Bessel
    tables are fixed. Omitting ``trainable`` preserves each flavor's default.
    """
    if kind not in _BASIS_TYPES:
        raise ValueError(f"unknown radial basis {kind!r}; choices are {tuple(_BASIS_TYPES)}")
    if trainable is not None and kind in ("bessel", "expnormal"):
        options["trainable"] = trainable
    if trainable and kind in ("bernstein", "chebyshev", "spherical_bessel"):
        raise ValueError(f"{kind} has no trainable shape parameters")
    basis = _BASIS_TYPES[kind](num_rbf=num_rbf, cutoff_sr=cutoff_sr, **options)
    if trainable and kind in ("gaussian", "sinc"):
        name = "center" if kind == "gaussian" else "factor"
        setattr(basis, name, Parameter(getattr(basis, name)))
    if trainable is not None and kind in ("exp_gaussian", "exp_bernstein"):
        basis.requires_grad_(trainable)
    return basis
