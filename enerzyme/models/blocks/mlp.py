import math
from typing import Optional, Union, Literal
from numpy import ndarray
from torch import Tensor
from torch.nn import Module, init, Sequential, Parameter, Dropout
import torch.nn.functional as F
import torch
from ..activation import get_activation_fn, ACTIVATION_PARAM_TYPE, ACTIVATION_KEY_TYPE
from ..init import semi_orthogonal_glorot_weights


INITIAL_WEIGHT_TYPE = Union[Tensor, ndarray, Literal["semi_orthogonal_glorot", "orthogonal", "zero", "xavier_uniform"]]
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
        elif initial_weight == "xavier_uniform":
            self.weight = Parameter(torch.empty(dim_feature_out * shallow_ensemble_size, dim_feature_in))
            init.xavier_uniform_(self.weight)
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
        use_bias: bool=True,
        use_residual: bool=True,
    ) -> None:
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)
        self.use_residual = use_residual
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
        if self.use_residual:
            return x + self.residual(x)
        else:
            return self.residual(x)
    

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
        use_bias: bool=True,
        use_residual: bool=True,
    ) -> None:
        super().__init__(dim_feature, dim_feature)
        self.num_residual = num_residual
        self.stack = Sequential(*(ResidualLayer(
            dim_feature, dim_feature,
            activation_fn, activation_params,
            initial_weight1, initial_weight2, initial_bias,
            dropout_rate, use_bias, use_residual=use_residual
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
        shallow_ensemble_size: int=1,
        use_residual: bool=True
    ) -> None:
        super().__init__(dim_feature_in, dim_feature_out, activation_fn, activation_params)
        self.stack = ResidualStack(
            dim_feature_in, num_residual, 
            activation_fn, activation_params,
            initial_weight1, initial_weight2, initial_bias_residual,
            dropout_rate, use_bias_residual, use_residual=use_residual
        )
        self.output = DenseLayer(
            dim_feature_in, dim_feature_out,
            initial_weight=initial_weight_out, initial_bias=initial_bias_out,
            use_bias=use_bias_out, 
            shallow_ensemble_size=shallow_ensemble_size
        )
        self.num_residual = num_residual

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_fn is None:
            return self.output(self.stack(x))
        else:
            return self.output(self.activation_fn(self.stack(x)))


class _ShiftedSoftplus(Module):
    """``softplus(x) - log(2)``, zero at the origin."""

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - math.log(2.0)


def _variance_nonlinearity(name: Optional[str]) -> Optional[type]:
    if name is None:
        return None
    key = str(name).lower()
    if key in {"none", "null"}:
        return None
    table = {
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "ssp": _ShiftedSoftplus,
    }
    if key not in table:
        known = ", ".join([*sorted(table), "none"])
        raise ValueError(f"unknown nonlinearity {name!r}; expected one of {known}")
    return table[key]


class VariancePreservingMLP(Module):
    """Bias-free MLP whose initial weights keep activation variance near one.

    Matches ``nequip.nn.ScalarMLPFunction`` (uniform weights, ReLU gain on
    every nonlinear layer except the variance-normalized end). ``nn.Linear``
    stores ``(out, in)``; the scale is folded into the initial weight, so the
    model class is the same as NequIP's ``(in, out)`` layout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: Optional[int] = None,
        nonlinearity: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1 or int(output_dim) < 1:
            raise ValueError(
                f"MLP dims must be >= 1, got {input_dim} -> {output_dim}"
            )
        depth = int(hidden_layers_depth)
        if depth < 0:
            raise ValueError(f"hidden_layers_depth must be >= 0, got {depth}")
        if depth == 0:
            dims = [int(input_dim), int(output_dim)]
        else:
            if hidden_layers_width is None or int(hidden_layers_width) < 1:
                raise ValueError(
                    "hidden_layers_width must be >= 1 when "
                    f"hidden_layers_depth={depth}"
                )
            width = int(hidden_layers_width)
            dims = [int(input_dim), *([width] * depth), int(output_dim)]
        act_cls = _variance_nonlinearity(nonlinearity)
        self.dims = dims
        self.is_nonlinear = act_cls is not None and len(dims) > 2
        self.forward_weight_init = bool(forward_weight_init)
        layers: list = []
        n_linear = len(dims) - 1
        for layer, (h_in, h_out) in enumerate(zip(dims, dims[1:])):
            if self.forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if act_cls is None or layer == 0 else math.sqrt(2.0)
            else:
                norm_dim = h_out
                gain = 1.0 if act_cls is None or layer == n_linear - 1 else math.sqrt(2.0)
            linear = torch.nn.Linear(h_in, h_out, bias=bias)
            init.uniform_(linear.weight, -math.sqrt(3.0), math.sqrt(3.0))
            with torch.no_grad():
                linear.weight.mul_(gain / math.sqrt(float(norm_dim)))
            if bias:
                init.zeros_(linear.bias)
            layers.append(linear)
            if act_cls is not None and layer != n_linear - 1:
                layers.append(act_cls())
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
