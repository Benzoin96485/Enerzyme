import math
from abc import ABC, abstractmethod
from typing import Literal, Dict, Optional
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from . import BaseFFLayer
from ..cutoff import CUTOFF_REGISTER
from ..functional import softplus_inverse


class BaseRBF(BaseFFLayer):
    def __init__(
        self,
        num_rbf: int,
        cutoff_sr: float,
        cutoff_fn: Literal["polynomial", "bump"]
    ) -> None:
        super().__init__(input_fields={"Dij_sr", "cutoff_sr_values"}, output_fields={"rbf"})
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]

    def get_rbf(self, Dij_sr: Tensor, cutoff_values: Optional[Tensor]=None, **kwargs) -> Tensor:
        if cutoff_values is None:
            cutoff_values = self.cutoff_fn(Dij_sr, cutoff=self.cutoff_sr)
        return cutoff_values.view(-1, 1) * self._get_rbf(Dij_sr)

    @abstractmethod
    def _get_rbf(self, Dij: Tensor) -> Tensor:
        ...


class GaussianRBFLayer(BaseRBF):
    """
    Radial basis functions based on Gaussian functions given by:
    g_i(x) = exp(-width*(x-center_i)**2)
    Here, i takes values from 0 to num_basis_functions-1. The centers are chosen
    to optimally cover the range x = 0...cutoff and the width parameter is
    selected to give optimal overlap between adjacent Gaussian functions.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_rbf: int, cutoff_sr: float, cutoff_fn: Literal["polynomial", "bump"]="bump") -> None:
        """ Initializes the GaussianFunctions class. """
        super().__init__(num_rbf, cutoff_sr, cutoff_fn)
        self.register_buffer(
            "center",
            torch.linspace(0, cutoff_sr, num_rbf, dtype=torch.float64),
        )
        self.register_buffer(
            "width", torch.tensor(num_rbf / cutoff_sr, dtype=torch.float64)
        )

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
        return torch.exp(
            -self.width * (Dij.view(-1, 1) - self.center) ** 2
        )


class BernsteinRBFLayer(BaseRBF):
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

    def __init__(self, num_rbf: int, cutoff_sr: float, cutoff_fn: Literal["polynomial", "bump"]="bump") -> None:
        """ Initializes the BernsteinPolynomials class. """
        super().__init__(num_rbf, cutoff_sr, cutoff_fn)
        # compute values to initialize buffers
        from ..special import get_berstein_coefficient
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


class SincRBFLayer(BaseRBF):
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
            from ..special import sinc
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


class ExponentialRBF(BaseRBF):
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
        -----
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
        -----
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
        -----
        r: Float tensor of distances, shape [M]

        cutoff_values: Float tensor of pre-calculated cutoff distances, shape [M]. 
        If not provided, the cutoff distances are calculated by `cutoff_fn`.

        Returns:
        -----
        rbf: Float tensor of RBFs, shape [M, `num_basis_functions`]
        '''
        alphar = -F.softplus(self.alpha) * Dij.view(-1, 1)
        expalphar = torch.exp(alphar)
        return torch.exp(self.inner_fn(alphar, expalphar)) * (expalphar if self.exp_weighting else 1)


class ExponentialGaussianRBFLayer(ExponentialRBF):
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
        -----
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
        -----
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


class ExponentialBernsteinRBFLayer(ExponentialRBF):
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
        -----
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
        from ..special import get_berstein_coefficient
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