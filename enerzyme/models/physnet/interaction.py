from typing import Optional, Union
import numpy as np
import torch
from torch import Tensor
from torch.nn import Dropout, Parameter
from ..functional import segment_sum_coo
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from ..layers.mlp import DenseLayer as _DenseLayer
from ..layers.mlp import ResidualStack as _ResidualStack
from ..layers.mlp import INITIAL_WEIGHT_TYPE, INITIAL_BIAS_TYPE, NeuronLayer, ResidualMLP


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
        self.dropout = Dropout(dropout_rate)
        #transforms radial basis functions to feature space
        self.k2f = DenseLayer(num_rbf, dim_embedding, initial_weight="zero", use_bias=False)
        #rearrange feature vectors for computing the "message"
        self.dense_i = DenseLayer(dim_embedding, dim_embedding, activation_fn, activation_params) # central atoms
        self.dense_j = DenseLayer(dim_embedding, dim_embedding, activation_fn, activation_params) # neighbouring atoms
        #for performing residual transformation on the "message"
        self.residual_stack = ResidualStack(dim_embedding, num_residual, activation_fn, activation_params, dropout_rate=dropout_rate)
        #for performing the final update to the feature vectors
        self.dense = DenseLayer(dim_embedding, dim_embedding)
        self.u = Parameter(torch.ones([dim_embedding]))

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
        xj = segment_sum_coo(g * self.dense_j(xa)[idx_j], idx_i, dim_size=xi.shape[0])
        #add contributions to get the "message" 
        m = xi + xj
        m = self.residual_stack(m)
        if self.activation_fn is not None: 
            m = self.activation_fn(m)
        x = self.u * x + self.dense(m)
        return x


class InteractionBlock(NeuronLayer):
    def __str__(self) -> str:
        return "Interaction Block: " + super().__str__()

    def __init__(
        self, num_rbf: int, dim_embedding: int, num_residual_atomic: int, num_residual_interaction: int, 
        activation_fn: ACTIVATION_KEY_TYPE=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), dropout_rate: float=0.0
    ) -> None:
        super().__init__(num_rbf, dim_embedding)
        #interaction layer
        self.interaction = InteractionLayer(num_rbf, dim_embedding, num_residual_interaction, activation_fn=activation_fn, activation_params=activation_params, dropout_rate=dropout_rate)

        #residual layers
        self.residual_stack = ResidualStack(dim_embedding, num_residual_atomic, activation_fn, activation_params, dropout_rate=dropout_rate)

    def forward(self, x: Tensor, rbf: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        return self.residual_stack(self.interaction(x, rbf, idx_i, idx_j))
    

def OutputBlock(
    dim_embedding: int, num_residual: int, 
    activation_fn: ACTIVATION_KEY_TYPE=None, activation_params: ACTIVATION_PARAM_TYPE=dict(),
    dropout_rate: float=0.0, shallow_ensemble_size: int=1
) -> ResidualMLP:
    default_initial_weight = weight_default()
    return ResidualMLP(
        dim_feature_in=dim_embedding, dim_feature_out=2, num_residual=num_residual,
        activation_fn=activation_fn, activation_params=activation_params,
        initial_weight1=default_initial_weight, initial_weight2=default_initial_weight, initial_weight_out="zero",
        initial_bias_residual=bias_default(), use_bias_out=False, dropout_rate=dropout_rate, shallow_ensemble_size=shallow_ensemble_size
    )
