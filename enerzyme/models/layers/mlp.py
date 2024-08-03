from typing import Optional, Union, Literal
from numpy import ndarray
from torch import nn, Tensor
from torch.nn import Module, init
import torch.nn.functional as F
import torch
from ..activation import get_activation_fn, ACTIVATION_PARAM_TYPE, ACTIVATION_KEY_TYPE
from .init import semi_orthogonal_glorot_weights


INITIAL_WEIGHT_TYPE = Union[Tensor, ndarray, Literal["semi_orthogonal_glorot", "orthogonal", "zero"]]
INITIAL_BIAS_TYPE = Union[Tensor, ndarray, Literal["zero"]]


class NeuronLayer(Module):
    def __str__(self):
        return "[ " + str(self.dim_feature_in) + " -> " + str(self.dim_feature_out) + " ]"

    def __init__(
        self, dim_feature_in, dim_feature_out, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
    ):
        super().__init__()
        self.dim_feature_in = dim_feature_in
        self.dim_feature_out = dim_feature_out
        if activation_fn is not None:
            self.activation_fn = get_activation_fn(activation_fn, activation_params)
        else:
            self.activation_fn = None


class DenseLayer(NeuronLayer):
    def __str__(self):
         return "Dense layer: " + super().__str__()

    def __init__(
        self, dim_feature_in: int, dim_feature_out: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_bias: INITIAL_BIAS_TYPE="zero",
        use_bias: bool=True
    ):
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)

        if initial_weight == "semi_orthogonal_glorot":
            self.weight = nn.Parameter(semi_orthogonal_glorot_weights(dim_feature_in, dim_feature_out))
        elif initial_weight == "orthogonal":
            self.weight = nn.Parameter(torch.empty(dim_feature_out, dim_feature_in))
            init.orthogonal_(self.weight)
        elif initial_weight == "zero":
            self.weight = nn.Parameter(torch.empty(dim_feature_out, dim_feature_in))
            init.zeros_(self.weight)
        else:
            self.weight = nn.Parameter(torch.tensor(initial_weight))
        if use_bias:
            if initial_bias == "zero":
                self.bias = nn.Parameter(torch.empty(dim_feature_out))
                init.zeros_(self.bias)
            else:
                self.bias = nn.Parameter(torch.tensor(initial_bias))
        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.activation_fn is None:
            return y
        else:
            return self.activation_fn(y)

    def l2loss(self) -> Tensor:
        return F.mse_loss(self.weight, torch.zeros_like(self.weight), reduction="sum") / 2
    

class ResidualLayer(NeuronLayer):
    def __str__(self):
         return "Residual layer: " + super().__str__()

    def __init__(
        self, dim_feature_in: int, dim_feature_out: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight1: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_weight2: INITIAL_WEIGHT_TYPE="zero", 
        initial_bias: INITIAL_BIAS_TYPE="zero",
        dropout_rate: float=0,
        use_bias: bool=True
    ) -> None:
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)
        
        dropout = nn.Dropout(dropout_rate)
        dense1 = DenseLayer(
            dim_feature_in=dim_feature_in, 
            dim_feature_out=dim_feature_out, 
            activation_fn=activation_fn, 
            activation_params=activation_params, 
            initial_weight=initial_weight1, 
            initial_bias=initial_bias, 
            use_bias=use_bias
        )
        dense2 = DenseLayer(
            dim_feature_in=dim_feature_in, 
            dim_feature_out=dim_feature_out, 
            activation_fn=None,
            initial_weight=initial_weight2, 
            initial_bias=initial_bias, 
            use_bias=use_bias
        )
        if activation_fn is not None:
            self.residual = nn.Sequential(self.activation_fn, dropout, dense1, dense2)
        else:
            self.residual = nn.Sequential(dropout, dense1, dense2)
        

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual(x)