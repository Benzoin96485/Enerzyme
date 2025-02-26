from typing import Optional, Union, Literal
from numpy import ndarray
from torch import Tensor
from torch.nn import Module, init, Sequential, Parameter, Dropout
import torch.nn.functional as F
import torch
from ..activation import get_activation_fn, ACTIVATION_PARAM_TYPE, ACTIVATION_KEY_TYPE
from ..init import semi_orthogonal_glorot_weights


INITIAL_WEIGHT_TYPE = Union[Tensor, ndarray, Literal["semi_orthogonal_glorot", "orthogonal", "zero"]]
INITIAL_BIAS_TYPE = Union[Tensor, ndarray, Literal["zero"]]


class NeuronLayer(Module):
    def __str__(self) -> str:
        return "[ " + str(self.dim_feature_in) + " -> " + str(self.dim_feature_out) + " ]"

    def __init__(
        self, dim_feature_in, dim_feature_out, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
    ) -> None:
        super().__init__()
        self.dim_feature_in = dim_feature_in
        self.dim_feature_out = dim_feature_out
        if activation_fn is not None:
            self.activation_fn = get_activation_fn(activation_fn, activation_params)
        else:
            self.activation_fn = None


class DenseLayer(NeuronLayer):
    def __str__(self) -> str:
         return "Dense layer: " + super().__str__()

    def __init__(
        self, dim_feature_in: int, dim_feature_out: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_bias: INITIAL_BIAS_TYPE="zero",
        use_bias: bool=True,
        shallow_ensemble_size: int=1
    ) -> None:
        self.shallow_ensemble_size = shallow_ensemble_size
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)

        if initial_weight == "semi_orthogonal_glorot":
            self.weight = Parameter(semi_orthogonal_glorot_weights(dim_feature_in, dim_feature_out * shallow_ensemble_size))
        elif initial_weight == "orthogonal" or shallow_ensemble_size > 1:
            self.weight = Parameter(torch.empty(dim_feature_out * shallow_ensemble_size, dim_feature_in))
            init.orthogonal_(self.weight)
        elif initial_weight == "zero":
            self.weight = Parameter(torch.empty(dim_feature_out * shallow_ensemble_size, dim_feature_in))
            init.zeros_(self.weight)
        else:
            if not isinstance(initial_weight, Tensor):
                initial_weight = torch.tensor(initial_weight)
            self.weight = Parameter(initial_weight)
        if use_bias:
            if initial_bias == "zero":
                self.bias = Parameter(torch.empty(dim_feature_out * shallow_ensemble_size))
                init.zeros_(self.bias)
            else:
                if not isinstance(initial_bias, Tensor):
                    initial_bias = torch.tensor(initial_bias)
                self.bias = Parameter(initial_bias)
        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.activation_fn is not None:
            y = self.activation_fn(y)
        if self.shallow_ensemble_size > 1:
            return y.view(-1, self.dim_feature_out, self.shallow_ensemble_size)
        else:
            return y

    def l2loss(self) -> Tensor:
        return F.mse_loss(self.weight, torch.zeros_like(self.weight), reduction="sum") / 2 / self.shallow_ensemble_size
    

class ResidualLayer(NeuronLayer):
    def __str__(self) -> str:
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
        
        dropout = Dropout(dropout_rate)
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
            dim_feature_in=dim_feature_out, 
            dim_feature_out=dim_feature_out, 
            activation_fn=None,
            initial_weight=initial_weight2, 
            initial_bias=initial_bias, 
            use_bias=use_bias
        )
        if activation_fn is not None:
            self.residual = Sequential(self.activation_fn, dropout, dense1, dense2)
        else:
            self.residual = Sequential(dropout, dense1, dense2)
        

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual(x)
    

class ResidualStack(NeuronLayer):
    def __str__(self) -> str:
         return f"Residual stack ({self.num_residual} layers): " + super().__str__()

    def __init__(
        self, dim_feature: int, num_residual: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight1: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_weight2: INITIAL_WEIGHT_TYPE="zero", 
        initial_bias: INITIAL_BIAS_TYPE="zero",
        dropout_rate: float=0,
        use_bias: bool=True
    ) -> None:
        super().__init__(dim_feature, dim_feature)
        self.num_residual = num_residual
        self.stack = Sequential(*(ResidualLayer(
            dim_feature, dim_feature,
            activation_fn, activation_params,
            initial_weight1, initial_weight2, initial_bias,
            dropout_rate, use_bias
        ) for _ in range(num_residual)))

    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x)


class ResidualMLP(NeuronLayer):
    def __str__(self) -> str:
         return f"Residual MLP ({self.num_residual} residual layers): " + super().__str__()
    
    def __init__(
        self, dim_feature_in: int, dim_feature_out:int, num_residual: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight1: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_weight2: INITIAL_WEIGHT_TYPE="zero", 
        initial_weight_out: INITIAL_WEIGHT_TYPE="zero",
        initial_bias_residual: INITIAL_BIAS_TYPE="zero",
        initial_bias_out: INITIAL_BIAS_TYPE="zero",
        dropout_rate: float=0,
        use_bias_residual: bool=True,
        use_bias_out: bool=True,
        shallow_ensemble_size: int=1
    ) -> None:
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)
        self.stack = ResidualStack(
            dim_feature_in, num_residual, 
            activation_fn, activation_params,
            initial_weight1, initial_weight2, initial_bias_residual,
            dropout_rate, use_bias_residual
        )
        self.output = DenseLayer(
            dim_feature_in, dim_feature_out,
            initial_weight=initial_weight_out, initial_bias=initial_bias_out,
            use_bias=use_bias_out, 
            shallow_ensemble_size=shallow_ensemble_size
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_fn is None:
            return self.output(self.stack(x))
        else:
            return self.output(self.activation_fn(self.stack(x)))
