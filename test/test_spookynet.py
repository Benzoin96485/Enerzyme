import logging
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


def test_shifted_softplus():
    from enerzyme.models.activation import ShiftedSoftplus as F1
    from spookynet.modules.shifted_softplus import ShiftedSoftplus as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta)
    f2 = F2(dim_feature, initial_alpha, initial_beta)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_swish():
    from enerzyme.models.activation import Swish as F1
    from spookynet.modules.swish import Swish as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta)
    f2 = F2(dim_feature, initial_alpha, initial_beta)
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
    )
    f2 = F2(dim_feature)
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
    )
    f2 = F2(dim_feature, 3)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_residual_mlp():
    from enerzyme.models.layers.mlp import ResidualMLP as F1
    from spookynet.modules.residual_mlp import ResidualMLP as F2
    f2 = F2(dim_feature, 3)
    f1 = F1(
        dim_feature_in=dim_feature, dim_feature_out=dim_feature, num_residual=3, 
        activation_fn="swish", activation_params={
            "dim_feature": dim_feature,
            "learnable": True
        },
        initial_weight1="orthogonal", initial_weight2="zero", initial_weight_out=f2.linear.weight.data,
        initial_bias_residual="zero", initial_bias_out=f2.linear.bias.data
    )
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())


def test_nuclear_embedding():
    pass


def test_electronic_embedding():
    pass


def test_nonlinear_electronic_embedding():
    pass


def test_exponential_gaussian_functions():
    pass


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