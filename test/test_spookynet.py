import logging

from enerzyme.models.layers import atom_embedding
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.extend(["..", "."])
import numpy as np
from numpy.testing import assert_allclose
import torch

dim_feature = 64
initial_alpha = np.random.randn()
initial_beta = np.random.randn()
x = torch.randn(dim_feature)
max_Za = 87
N = 100
idx_i = torch.empty((N, N-1), dtype=int)
idx_j = torch.empty((N, N-1), dtype=int)
for i in range(N):
    for j in range(N - 1):
        idx_i[i, j] = i
for i in range(N):
    c = 0
    for j in range(N):
        if j != i:
            idx_j[i,c] = j
            c += 1
idx_i = idx_i.reshape(-1)
idx_j = idx_j.reshape(-1)
D = torch.rand(*idx_i.shape) * 30 + 3
Za = torch.randint(1, max_Za, (N,))
cutoff_values = torch.rand(*idx_i.shape)
atom_embedding = torch.randn((N, dim_feature))
dtype = "float64"
if dtype == "float64":
    dtype = torch.float64
elif dtype == "float32":
    dtype = torch.float32
x = x.type(dtype)
atom_embedding = atom_embedding.type(dtype)
D = D.type(dtype)
cutoff_values = cutoff_values.type(dtype)


def test_shifted_softplus():
    from enerzyme.models.activation import ShiftedSoftplus as F1
    from spookynet.modules.shifted_softplus import ShiftedSoftplus as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta).type(dtype)
    f2 = F2(dim_feature, initial_alpha, initial_beta).type(dtype)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_swish():
    from enerzyme.models.activation import Swish as F1
    from spookynet.modules.swish import Swish as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta).type(dtype)
    f2 = F2(dim_feature, initial_alpha, initial_beta).type(dtype)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_residual_layer():
    from enerzyme.models.layers.mlp import ResidualLayer as F1
    from spookynet.modules.residual import Residual as F2
    f1 = F1(
        dim_feature_in=dim_feature, dim_feature_out=dim_feature, 
        activation_fn="swish", activation_params={
            "dim_feature": dim_feature,
            "learnable": True
        },
        initial_bias="zero", initial_weight1="orthogonal", initial_weight2="zero"
    ).type(dtype)
    f2 = F2(dim_feature).type(dtype)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_residual_stack():
    from enerzyme.models.layers.mlp import ResidualStack as F1
    from spookynet.modules.residual_stack import ResidualStack as F2
    f1 = F1(
        dim_feature=dim_feature, num_residual=3, 
        activation_fn="swish", activation_params={
            "dim_feature": dim_feature,
            "learnable": True
        },
        initial_bias="zero", initial_weight1="orthogonal", initial_weight2="zero"
    ).type(dtype)
    f2 = F2(dim_feature, 3).type(dtype)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_residual_mlp():
    from enerzyme.models.layers.mlp import ResidualMLP as F1
    from spookynet.modules.residual_mlp import ResidualMLP as F2
    f2 = F2(dim_feature, 3).type(dtype)
    f1 = F1(
        dim_feature_in=dim_feature, dim_feature_out=dim_feature, num_residual=3, 
        activation_fn="swish", activation_params={
            "dim_feature": dim_feature,
            "learnable": True
        },
        initial_weight1="orthogonal", initial_weight2="zero", initial_weight_out=f2.linear.weight.data,
        initial_bias_residual="zero", initial_bias_out=f2.linear.bias.data
    ).type(dtype)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_nuclear_embedding():
    from enerzyme.models.layers.atom_embedding import NuclearEmbedding as F1
    from spookynet.modules.nuclear_embedding import NuclearEmbedding as F2
    f1 = F1(86, dim_feature).type(dtype)
    f2 = F2(dim_feature).type(dtype)
    assert_allclose(f1.get_embedding(Za).detach().numpy(), f2(Za).detach().numpy())


def test_electronic_embedding():
    from enerzyme.models.layers.electron_embedding import ElectronicEmbedding as F1
    from spookynet.modules.electronic_embedding import ElectronicEmbedding as F2
    f1 = F1(dim_feature, 3, attribute="spin").type(dtype)
    f2 = F2(dim_feature, 3).type(dtype)
    S = torch.tensor([2], dtype=dtype)
    assert_allclose(
        f1.get_embedding(atom_embedding=atom_embedding, S=S).detach().numpy(), 
        f2(atom_embedding, S, 1, None).detach().numpy()
    )
    pass


def test_nonlinear_electronic_embedding():
    from enerzyme.models.layers.electron_embedding import NonlinearElectronicEmbedding as F1
    from spookynet.modules.nonlinear_electronic_embedding import NonlinearElectronicEmbedding as F2
    f1 = F1(dim_feature, 3, attribute="spin").type(dtype)
    f2 = F2(dim_feature, 3).type(dtype)
    S = torch.tensor([2], dtype=dtype)
    assert_allclose(
        f1.get_embedding(atom_embedding=atom_embedding, S=S).detach().numpy(), 
        f2(atom_embedding, S, 1, torch.zeros(N, dtype=torch.long)).detach().numpy()
    )
    pass


def test_exponential_gaussian_functions():
    from enerzyme.models.layers.rbf import ExponentialGaussianRBFLayer as F1
    from spookynet.modules.exponential_gaussian_functions import ExponentialGaussianFunctions as F2
    f1 = F1(dim_feature, init_width_flavor="SpookyNet")
    f2 = F2(dim_feature)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy(), rtol=1e-6, atol=1e-6
    )


def test_exponential_bernstein_polynomials():
    pass


def test_gaussian_functions():
    pass


def test_bernstein_polynomials():
    pass


def test_sinc_functions():
    pass


def test_local_interaction():
    pass


def test_nonlocal_interaction():
    pass


def test_interaction_module():
    pass


def test_zbl_repulsion_energy():
    pass


def test_d4_dispersion_energy():
    pass


def test_calculate_distances():
    pass


def test_atomic_properties_static():
    pass


def test_atomic_properties_dynamic():
    pass


def test_atomic_properties():
    pass


def test_energy():
    pass


def test_energy_and_forces():
    pass