import numpy as np
import torch
from torch import nn
import torch.nn.functional as F_
from .func import softplus_inverse, segment_sum
from .init import semi_orthogonal_glorot_weights


class NeuronLayer(nn.Module):
    def __str__(self):
        return "[ " + str(self.n_in) + " -> " + str(self.n_out) + " ]"

    def __init__(self, n_in, n_out, activation_fn=None):
        super().__init__()
        self._n_in  = n_in  # number of inputs
        self._n_out = n_out # number of outpus
        self._activation_fn = activation_fn # activation function
            
    @property
    def n_in(self):
        return self._n_in
    
    @property
    def n_out(self):
        return self._n_out
    
    @property
    def activation_fn(self):
        return self._activation_fn
    

class RBFLayer(NeuronLayer):
    def __str__(self):
        return "Radial basis function layer: " + super().__str__()

    def __init__(self, K, cutoff, dtype=torch.double):
        super().__init__(1, K, None)
        self._K = K
        self._cutoff = cutoff
        self._dtype = dtype
        centers = softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K))
        self._centers = nn.Parameter(F_.softplus(torch.tensor(np.asarray(centers), dtype=dtype)))
        widths = [softplus_inverse((0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2)] * K
        self._widths = nn.Parameter(F_.softplus(torch.tensor(np.asarray(widths), dtype=dtype, requires_grad=True)))

    @property
    def K(self):
        return self._K

    @property
    def cutoff(self):
        return self._cutoff
    
    @property
    def centers(self):
        return self._centers   

    @property
    def widths(self):
        return self._widths  

    # cutoff function that ensures a smooth cutoff
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3 = x ** 3
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(x, dtype=self._dtype))
    
    def forward(self, D):
        D = torch.unsqueeze(D, -1) # necessary for proper broadcasting behaviour
        rbf = self.cutoff_fn(D) * torch.exp(-self.widths * (torch.exp(-D) - self.centers) ** 2)
        return rbf


class DenseLayer(NeuronLayer):
    def __str__(self):
        return "Dense layer: " + super().__str__()

    def __init__(
        self, n_in, n_out, activation_fn=None, 
        W_init=None, b_init=None, use_bias=True, regularization=True, dtype=torch.double
    ):
        super().__init__(n_in, n_out, activation_fn)
        if W_init is None:
            W_init = semi_orthogonal_glorot_weights(n_in, n_out) 
            self._W = nn.Parameter(torch.tensor(W_init, dtype=dtype))
        else:
            self._W = nn.Parameter(W_init.type(dtype))

        #define l2 loss term for regularization
        if regularization:
            self._l2loss = F_.mse_loss(self.W, torch.zeros_like(self.W, dtype=dtype), reduction="sum") / 2
        else:
            self._l2loss = 0.0

        #define bias
        self._use_bias = use_bias
        if self.use_bias:
            if b_init is None:
                b_init = nn.Parameter(torch.zeros([self.n_out], dtype=dtype))
            self._b = nn.Parameter(b_init.type(dtype))

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def l2loss(self):
        return self._l2loss
    
    @property
    def use_bias(self):
        return self._use_bias

    def forward(self, x):
        y = torch.matmul(x, self.W)
        if self.use_bias:
            y += self.b
        if self.activation_fn is not None: 
            y = self.activation_fn(y)
        return y


class ResidualLayer(NeuronLayer):
    def __str__(self):
        return "Residual layer: " + super().__str__()

    def __init__(
        self, n_in, n_out, activation_fn=None, 
        W_init=None, b_init=None, use_bias=True, drop_out=0.0, dtype=torch.double
    ):
        super().__init__(n_in, n_out, activation_fn)
        self._drop_out = nn.Dropout(drop_out)
        self._dense = DenseLayer(n_in, n_out, activation_fn=activation_fn, 
            W_init=W_init, b_init=b_init, use_bias=use_bias, dtype=dtype)
        self._residual = DenseLayer(n_out, n_out, activation_fn=None, 
            W_init=W_init, b_init=b_init, use_bias=use_bias, dtype=dtype)
      
    @property
    def drop_out(self):
        return self._drop_out
    
    @property
    def dense(self):
        return self._dense

    @property
    def residual(self):
        return self._residual

    def forward(self, x):
        #pre-activation
        if self.activation_fn is not None: 
            y = self.drop_out(self.activation_fn(x))
        else:
            y = self.drop_out(x)
        z = self.residual(self.dense(y)) + x
        return z


class InteractionLayer(NeuronLayer):
    def __str__(self):
        return "Interaction layer: " + super().__str__()

    def __init__(self, K, F, num_residual, activation_fn=None, drop_out=0.0, dtype=torch.double):
        super().__init__(K, F, activation_fn)
        self._drop_out = nn.Dropout(drop_out)
        #transforms radial basis functions to feature space
        self._k2f = DenseLayer(K, F, W_init=torch.zeros([K, F], requires_grad=True, dtype=dtype), use_bias=False, dtype=dtype)
        #rearrange feature vectors for computing the "message"
        self._dense_i = DenseLayer(F, F, activation_fn, dtype=dtype) # central atoms
        self._dense_j = DenseLayer(F, F, activation_fn, dtype=dtype) # neighbouring atoms
        #for performing residual transformation on the "message"
        self._residual_layer = nn.Sequential(*[
            ResidualLayer(F, F, activation_fn, drop_out=drop_out, dtype=dtype) for i in range(num_residual)
        ])
        #for performing the final update to the feature vectors
        self._dense = DenseLayer(F, F, dtype=dtype)
        self._u = nn.Parameter(torch.ones([F], dtype=dtype))

    @property
    def drop_out(self):
        return self._drop_out

    @property
    def k2f(self):
        return self._k2f

    @property
    def dense_i(self):
        return self._dense_i

    @property
    def dense_j(self):
        return self._dense_j

    @property
    def residual_layer(self):
        return self._residual_layer

    @property
    def dense(self):
        return self._dense

    @property
    def u(self):
        return self._u
    
    def forward(self, x, rbf, idx_i, idx_j):
        #pre-activation
        if self.activation_fn is not None: 
            xa = self.drop_out(self.activation_fn(x))
        else:
            xa = self.drop_out(x)
        #calculate feature mask from radial basis functions
        g = self.k2f(rbf)
        #calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        xj = segment_sum(g * self.dense_j(xa)[idx_j], idx_i)
        #add contributions to get the "message" 
        m = xi + xj 
        m = self.residual_layer(m)
        if self.activation_fn is not None: 
            m = self.activation_fn(m)
        x = self.u * x + self.dense(m)
        return x