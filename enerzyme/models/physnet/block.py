import torch
from torch import nn
from .layer import NeuronLayer, InteractionLayer, ResidualLayer, DenseLayer


class InteractionBlock(NeuronLayer):
    def __str__(self):
        return "Interaction block: " + super().__str__()

    def __init__(self, K, F, num_residual_atomic, num_residual_interaction, activation_fn=None, drop_out=0.0):
        super().__init__(K, F, activation_fn)
        #interaction layer
        self._interaction = InteractionLayer(K, F, num_residual_interaction, activation_fn=activation_fn, drop_out=drop_out)

        #residual layers
        self._residual_layer = nn.Sequential(*[
            ResidualLayer(F, F, activation_fn, drop_out=drop_out) for i in range(num_residual_atomic)
        ])

    @property
    def interaction(self):
        return self._interaction
    
    @property
    def residual_layer(self):
        return self._residual_layer

    def forward(self, x, rbf, idx_i, idx_j):
        return self.residual_layer(self.interaction(x, rbf, idx_i, idx_j))
    

class OutputBlock(NeuronLayer):
    def __str__(self):
        return "output"+super().__str__()

    def __init__(self, F, num_residual, activation_fn=None, drop_out=0.0):
        super().__init__(F, 2, activation_fn)
        self._residual_layer = nn.Sequential(*[
            ResidualLayer(F, F, activation_fn, drop_out=drop_out) for i in range(num_residual)
        ])
        self._dense = DenseLayer(F, 2, W_init=torch.zeros([F, 2]), use_bias=False)
    
    @property
    def residual_layer(self):
        return self._residual_layer

    @property
    def dense(self):
        return self._dense

    def forward(self, x):
        x = self.residual_layer(x)
        if self.activation_fn is not None: 
            x = self.activation_fn(x)
        return self.dense(x)