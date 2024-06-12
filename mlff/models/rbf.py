import math
from typing import Callable, Literal
import torch
from torch import nn
import torch.nn.functional as F
from .functional import cutoff_function, softplus_inverse


class ExponentialRBF(nn.Module):
    def __init__(
        self,
        type_basis_function: Literal["Gaussian", "Bernstein"],
        num_basis_functions: int,
        dtype: torch.dtype,
        no_basis_function_at_infinity: bool=False,
        init_alpha: float=0.9448630629184640,
        exp_weighting: bool=False,
        learnable_shape: bool=True,
        cutoff: float=float("inf"),
        cutoff_fn: Callable[[torch.Tensor, float], torch.Tensor]=None
    ) -> None:
        '''
        The base class of radial basis functions with a general exponential form

        Params:
        -----
        num_basis_functions: Number of radial basis functions.

        dtype: Data type of floating numbers.

        no_basis_function_at_infinity: If True, no basis function is put at exp(-alpha*x) = 0, i.e.
        x = infinity.

        init_alpha: Initial value for scaling parameter alpha (Default value corresponds
        to 0.5 1/Bohr converted to 1/Angstrom).
        
        exp_weighting: If `True`, basis functions are weighted with a factor exp(-alpha*r).

        learnable_shape: If `True`, shape parameters of exponentials are learnable.

        cutoff: Short range cutoff threshold for radial base functions.

        cutoff_fn: Short range cutoff function, whose are called by `cutoff_fn(x, cutoff=cutoff)` 
        where x is the distance.
        '''
        super.__init__()
        self.num_basis_functions = num_basis_functions
        self.dtype = dtype
        self.exp_weighting = exp_weighting
        self.learnable_shape = learnable_shape
        self.no_basis_function_at_infinity = no_basis_function_at_infinity
        
        
        
        self.register_parameter(
            "_alpha", nn.Parameter(softplus_inverse(init_alpha).type(dtype))
        )

        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn
        self.inner_fn = None

    def forward(self, r: torch.Tensor, cutoff_values: torch.Tensor=None) -> torch.Tensor:
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
        alphar = -F.softplus(self._alpha) * r.view(-1, 1)
        expalphar = torch.exp(alphar)
        if cutoff_values is None:
            cutoff_values = self.cutoff_fn(r, cutoff=self.cutoff)
        return cutoff_values.view(-1, 1) * self.inner_fn(alphar, expalphar) * (expalphar if self.exp_weighting else 1)


class ExponentialGaussianRBF(ExponentialRBF):
    def __init__(
        self,
        num_basis_functions: int,
        dtype: torch.dtype,
        no_basis_function_at_infinity: bool=False,
        init_alpha: float=0.9448630629184640,
        exp_weighting: bool=False,
        learnable_shape: bool=True,
        cutoff: float=float("inf"),
        cutoff_fn: Callable=lambda x, cutoff: cutoff_function(x, cutoff=cutoff, flavor="poly"),
        init_width_flavor: Literal["PhysNet", "SpookyNet"]="PhysNet"
    ) -> None:
        '''
        Exponential Gaussian radial basis functions

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
        
        - `PhysNet`: A constant number (2K^{-1}(1-\exp(-`cutoff`)))^{-2}, where K is `num_basis_functions`.
        - `SpookyNet`: A constant number K or K+1 (`no_basis_function_at_infinity=True`).
        '''
        super.__init__(
            num_basis_functions=num_basis_functions,
            dtype=dtype,
            no_basis_function_at_infinity=no_basis_function_at_infinity,
            init_alpha=init_alpha,
            exp_weighting=exp_weighting,
            learnable_shape=learnable_shape
        )
        if cutoff == float("inf") and no_basis_function_at_infinity:
            self.register_parameter(
                "_centers", nn.Parameter(
                    softplus_inverse(torch.linspace(
                        1, 0, num_basis_functions + 1, dtype=dtype
                    )[:-1]), 
                    requires_grad=self.learnable_shape
                )
            )
        else:
            self.register_parameter(
                "_centers", nn.Parameter(
                    softplus_inverse(torch.linspace(
                        1, math.exp(-cutoff), num_basis_functions, dtype=dtype
                    )), 
                    requires_grad=self.learnable_shape
                )
            )
        self._init_width(init_width_flavor)
        self.inner_fn = self.exponential_gaussian

    def _init_width(self, init_width_flavor: Literal["PhysNet", "SpookyNet"]="PhysNet") -> None:
        '''
        Initialize the widths of exponentials based on the chosen flavor.
        '''
        if init_width_flavor == "SpookyNet":
            self.register_parameter(
                "_widths", nn.Parameter(
                    softplus_inverse(
                    1.0 * self.num_basis_functions + \
                    int(self.no_basis_function_at_infinity)).type(self.dtype)
                ), 
                requires_grad=self.learnable_width
            )
        elif init_width_flavor == "PhysNet":
            self.register_parameter(
                "_widths", nn.Parameter(
                    torch.tensor(
                        softplus_inverse(
                            [(0.5 / ((-math.expm1(-self.cutoff)) / self.num_basis_functions)) ** 2] * \
                            self.num_basis_functions
                        ).type(self.dtype)
                    ), 
                    requires_grad=self.learnable_width
                )
            )

    def exponential_gaussian(self, alphar, expalphar) -> torch.Tensor:
        return torch.exp(
            -F.softplus(self.width) * (expalphar - F.softplus(self.center)) ** 2
        )

class ExponentialBernsteinRBF(ExponentialRBF):
    pass