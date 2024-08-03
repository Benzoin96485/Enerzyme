import torch
from torch import nn, Tensor
from .layer import NeuronLayer, InteractionLayer, ResidualStack, DenseLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from enerzyme.models import activation


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
    

class OutputBlock(NeuronLayer):
    def __str__(self):
        return "Output Block: "+ super().__str__()

    def __init__(self, dim_embedding, num_residual, 
        activation_fn: ACTIVATION_KEY_TYPE=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), dropout_rate: float=0.0
    ):
        super().__init__(dim_embedding, 2, activation_fn, activation_params)
        self.residual_stack = ResidualStack(dim_embedding, num_residual, activation_fn, activation_params, dropout_rate=dropout_rate)
        self.dense = DenseLayer(dim_embedding, 2, initial_weight="zero", use_bias=False)

    def forward(self, x):
        x = self.residual_stack(x)
        if self.activation_fn is not None: 
            x = self.activation_fn(x)
        return self.dense(x)