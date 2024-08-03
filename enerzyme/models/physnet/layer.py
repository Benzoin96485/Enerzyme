from typing import Optional, Union
import numpy as np
import torch
from torch import nn, Tensor
from ..functional import segment_sum
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from ..layers.mlp import DenseLayer as _DenseLayer
from ..layers.mlp import ResidualStack as _ResidualStack
from ..layers.mlp import INITIAL_WEIGHT_TYPE, INITIAL_BIAS_TYPE, NeuronLayer


DEFAULT_TYPE = Optional[Union[Tensor, np.ndarray, str]]


def weight_default(initial_weight: DEFAULT_TYPE=None) -> INITIAL_WEIGHT_TYPE:
    return "semi_orthogonal_glorot" if initial_weight is None else initial_weight


def bias_default(initial_weight: DEFAULT_TYPE=None) -> INITIAL_BIAS_TYPE:
    return "zero" if initial_weight is None else initial_weight


def DenseLayer(
    dim_feature_in: int, 
    dim_feature_out: int, 
    activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
    activation_params: ACTIVATION_PARAM_TYPE=dict(),
    initial_weight: DEFAULT_TYPE=None,
    initial_bias: DEFAULT_TYPE=None,
    use_bias: bool=True
) -> _DenseLayer:
    return _DenseLayer(
        dim_feature_in=dim_feature_in,
        dim_feature_out=dim_feature_out,
        activation_fn=activation_fn,
        activation_params=activation_params,
        initial_weight=weight_default(initial_weight),
        initial_bias=bias_default(initial_bias),
        use_bias=use_bias
    )


# def ResidualLayer(
#     dim_feature_in: int,
#     dim_feature_out: int,
#     activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
#     activation_params: ACTIVATION_PARAM_TYPE=dict(),
#     initial_weight: Optional[Union[Tensor, np.ndarray]]=None,
#     initial_bias: Optional[Union[Tensor, np.ndarray]]=None,
#     use_bias: bool=True,
#     dropout_rate: float=0.0
# ) -> _ResidualLayer:
#     default_initial_weight = weight_default(initial_weight)
#     return _ResidualLayer(
#         dim_feature_in=dim_feature_in,
#         dim_feature_out=dim_feature_out,
#         activation_fn=activation_fn,
#         activation_params=activation_params,
#         initial_weight1=default_initial_weight,
#         initial_weight2=default_initial_weight,
#         initial_bias=bias_default(initial_bias),
#         use_bias=use_bias,
#         dropout_rate=dropout_rate
#     )


def ResidualStack(
    dim_feature: int,
    num_residual: int,
    activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
    activation_params: ACTIVATION_PARAM_TYPE=dict(),
    initial_weight: Optional[Union[Tensor, np.ndarray]]=None,
    initial_bias: Optional[Union[Tensor, np.ndarray]]=None,
    use_bias: bool=True,
    dropout_rate: float=0.0
) -> _ResidualStack:
    default_initial_weight = weight_default(initial_weight)
    return _ResidualStack(
        dim_feature=dim_feature,
        num_residual=num_residual,
        activation_fn=activation_fn,
        activation_params=activation_params,
        initial_weight1=default_initial_weight,
        initial_weight2=default_initial_weight,
        initial_bias=bias_default(initial_bias),
        use_bias=use_bias,
        dropout_rate=dropout_rate
    )


class InteractionLayer(NeuronLayer):
    def __str__(self) -> str:
         return "Interaction layer: " + super().__str__()
    
    def __init__(
        self, num_rbf, dim_embedding, num_residual, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, 
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        dropout_rate=0.0) -> None:
        super().__init__(num_rbf, dim_embedding, activation_fn, activation_params)
        self.dropout = nn.Dropout(dropout_rate)
        #transforms radial basis functions to feature space
        self.k2f = DenseLayer(num_rbf, dim_embedding, initial_weight="zero", use_bias=False)
        #rearrange feature vectors for computing the "message"
        self.dense_i = DenseLayer(dim_embedding, dim_embedding, activation_fn, activation_params) # central atoms
        self.dense_j = DenseLayer(dim_embedding, dim_embedding, activation_fn, activation_params) # neighbouring atoms
        #for performing residual transformation on the "message"
        self.residual_stack = ResidualStack(dim_embedding, num_residual, activation_fn, activation_params, dropout_rate=dropout_rate)
        #for performing the final update to the feature vectors
        self.dense = DenseLayer(dim_embedding, dim_embedding)
        self.u = nn.Parameter(torch.ones([dim_embedding]))

    
    def forward(self, x: Tensor, rbf: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        #pre-activation
        if self.activation_fn is not None: 
            xa = self.dropout(self.activation_fn(x))
        else:
            xa = self.dropout(x)
        #calculate feature mask from radial basis functions
        g = self.k2f(rbf)
        #calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        xj = segment_sum(g * self.dense_j(xa)[idx_j], idx_i)
        #add contributions to get the "message" 
        m = xi + xj 
        m = self.residual_stack(m)
        if self.activation_fn is not None: 
            m = self.activation_fn(m)
        x = self.u * x + self.dense(m)
        return x
    
# class DenseLayer(NeuronLayer):
#     def __str__(self):
#         return "Dense layer: " + super().__str__()

#     def __init__(
#         self, n_in, n_out, activation_fn=None, 
#         W_init=None, b_init=None, use_bias=True, regularization=True
#     ):
#         super().__init__(n_in, n_out, activation_fn)
#         if W_init is None:
#             W_init = semi_orthogonal_glorot_weights(n_in, n_out) 
#             self._W = nn.Parameter(torch.tensor(W_init))
#         else:
#             self._W = nn.Parameter(W_init)

#         #define l2 loss term for regularization
#         if regularization:
#             self._l2loss = F_.mse_loss(self.W, torch.zeros_like(self.W), reduction="sum") / 2
#         else:
#             self._l2loss = 0.0

#         #define bias
#         self._use_bias = use_bias
#         if self.use_bias:
#             if b_init is None:
#                 b_init = nn.Parameter(torch.zeros([self.n_out]))
#             self._b = nn.Parameter(b_init)

#     @property
#     def W(self):
#         return self._W

#     @property
#     def b(self):
#         return self._b

#     @property
#     def l2loss(self):
#         return self._l2loss
    
#     @property
#     def use_bias(self):
#         return self._use_bias

#     def forward(self, x):
#         y = torch.matmul(x, self.W)
#         if self.use_bias:
#             y += self.b
#         if self.activation_fn is not None: 
#             y = self.activation_fn(y)
#         return y