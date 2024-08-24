import math
from abc import ABC, abstractmethod
from typing import Literal, Dict, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter


class BaseScaledTemperedActivation(ABC, Module):
    def __init__(self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.0, learnable: bool=False) -> None:
        super().__init__()
        if float(initial_alpha) == 1.0 and float(initial_beta) == 1.0 and not learnable:
            self.simple = True
            self.alpha = 1.0
            self.beta = 1.0
            self._activation_fn = self.simple_activation_fn
        else:
            self.simple = False
            self.register_parameter("alpha", Parameter(torch.full((dim_feature,), initial_alpha), requires_grad=learnable))
            self.register_parameter("beta", Parameter(torch.full((dim_feature,), initial_beta), requires_grad=learnable))
            self._activation_fn = self.activation_fn

    @abstractmethod
    def simple_activation_fn(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def activation_fn(self, x: Tensor) -> Tensor:
        ...

    def forward(self, x: Tensor) -> Tensor:
        return self._activation_fn(x)


class ShiftedSoftplus(BaseScaledTemperedActivation):
    def __init__(self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.0, learnable: bool=False) -> None:
        """
        Shifted softplus activation function with learnable feature-wise parameters:
        f(x) = alpha/beta * (softplus(beta*x) - log(2))
        softplus(x) = log(exp(x) + 1)
        For beta -> 0  : f(x) -> 0.5*alpha*x
        For beta -> inf: f(x) -> max(0, alpha*x)

        Arguments:
            num_features (int):
                Dimensions of feature space.
            initial_alpha (float):
                Initial "scale" alpha of the softplus function.
            initial_beta (float):
                Initial "temperature" beta of the softplus function.
        """
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)
        self.log2 = math.log(2.0)
    
    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.log2
    
    def activation_fn(self, x: Tensor) -> Tensor:
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self.log2) / self.beta,
            0.5 * x
        )


class Swish(BaseScaledTemperedActivation):
    """
    Swish activation function with learnable feature-wise parameters:
    f(x) = alpha*x * sigmoid(beta*x)
    sigmoid(x) = 1/(1 + exp(-x))
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the "linear component".
        initial_beta (float):
            Initial "temperature" of the "sigmoid component". The default value
            of 1.702 has the effect of initializing swish to an approximation
            of the Gaussian Error Linear Unit (GELU) activation function from
            Hendrycks, Dan, and Gimpel, Kevin. "Gaussian error linear units
            (GELUs)."
    """

    def __init__(
        self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.702, learnable: bool=True
    ) -> None:
        """ Initializes the Swish class. """
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)

    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return F.silu(x)

    def activation_fn(self, x: Tensor) -> Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, num_features]):
                Input features.

        Returns:
            y (FloatTensor [:, num_features]):
                Activated features.
        """
        return self.alpha * x * torch.sigmoid(self.beta * x)


ACTIVATION_REGISTER = {
    "shifted_softplus": ShiftedSoftplus,
    "swish": Swish,
}
ACTIVATION_KEY_TYPE = Literal["shifted_softplus", "swish"]
ACTIVATION_PARAM_TYPE = Dict[Literal["dim_feature", "initial_alpha", "initial_beta", "learnable"], Union[int, float, bool]]


def get_activation_fn(
    key: ACTIVATION_KEY_TYPE, 
    activation_params: ACTIVATION_PARAM_TYPE=dict()
) -> BaseScaledTemperedActivation:
    return ACTIVATION_REGISTER[key](**activation_params)