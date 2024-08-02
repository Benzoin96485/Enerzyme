import math
from typing import Literal, Dict
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def shifted_softplus(x: Tensor) -> Tensor:
    return 


class ShiftedSoftplus(nn.Module):
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
        super().__init__()
        self.log2 = math.log(2.0)
        if float(initial_alpha) == 1.0 and float(initial_beta) == 1.0 and not learnable:
            self._shifted_softplus = self.simple_shifted_softplus
            self.alpha = 1.0
            self.beta = 1.0
        else:
            self.register_parameter("alpha", nn.Parameter(torch.full((dim_feature,), initial_alpha), requires_grad=learnable))
            self.register_parameter("beta", nn.Parameter(torch.full((dim_feature,), initial_beta), requires_grad=learnable))
            self._shifted_softplus = self.scaled_shifted_softplus
    
    def simple_shifted_softplus(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.log2
    
    def scaled_shifted_softplus(self, x: Tensor) -> Tensor:
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self.log2) / self.beta,
            0.5 * x
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._shifted_softplus(x)


ACTIVATION_REGISTER = {
    "shifted_softplus": ShiftedSoftplus,
    "swish": ...,
}


def get_activation_fn(key: Literal["shifted_softplus", "swish"], activation_params: Dict=dict()) -> nn.Module:
    return ACTIVATION_REGISTER[key](**activation_params)