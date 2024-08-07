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
num_residual_local_x = 1
num_residual_local_s = 1
num_residual_local_p = 1
num_residual_local_d = 1
num_residual_local = 1
num_residual_nonlocal_q = 1
num_residual_nonlocal_k = 1
num_residual_nonlocal_v = 1
num_residual_pre = 1
num_residual_post = 1
num_residual_output = 1

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
vij = torch.rand(*idx_i.shape, 3) * 30
R = np.random.rand(N, 3) * 20
D = torch.norm(vij, dim=-1, keepdim=True)
pij = vij / D
D = D.squeeze(-1)
dij = torch.rand(*idx_i.shape, 5)
Za = torch.randint(1, max_Za, (N,))
rbf = torch.rand(*idx_i.shape, dim_feature)
cutoff_values = torch.rand(*idx_i.shape)
atom_embedding = torch.randn((N, dim_feature))
Q = torch.randn((N, dim_feature))
Qa = torch.randn((N,))
K = torch.randn((N, dim_feature))
dtype = "float64"
if dtype == "float64":
    dtype = torch.float64
elif dtype == "float32":
    dtype = torch.float32
x = x.type(dtype)
Q = Q.type(dtype)
Qa = Qa.type(dtype)
K = K.type(dtype)
atom_embedding = atom_embedding.type(dtype)
D = D.type(dtype)
R = R.type(dtype)
pij = pij.type(dtype)
dij = dij.type(dtype)
rbf = rbf.type(dtype)
cutoff_values = cutoff_values.type(dtype)


def test_cutoff_function():
    from enerzyme.models.cutoff import bump_transition
    from spookynet.functional import cutoff_function
    assert_allclose(bump_transition(D, 5).detach().numpy(), cutoff_function(D, 5).detach().numpy())


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
    f1 = F1(dim_feature, init_width_flavor="SpookyNet").type(dtype)
    f2 = F2(dim_feature).type(dtype)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy(), rtol=1e-6, atol=1e-6
    )


def test_exponential_bernstein_polynomials():
    from enerzyme.models.layers.rbf import ExponentialBernsteinRBFLayer as F1
    from spookynet.modules.exponential_bernstein_polynomials import ExponentialBernsteinPolynomials as F2
    f1 = F1(dim_feature).type(dtype).type(dtype)
    f2 = F2(dim_feature).type(dtype).type(dtype)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy(), rtol=1e-6, atol=1e-6
    )


def test_gaussian_functions():
    from enerzyme.models.layers.rbf import GaussianRBFLayer as F1
    from spookynet.modules.gaussian_functions import GaussianFunctions as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy()
    )


def test_bernstein_polynomials():
    from enerzyme.models.layers.rbf import BernsteinRBFLayer as F1
    from spookynet.modules.bernstein_polynomials import BernsteinPolynomials as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy()
    )


def test_sinc_functions():
    from enerzyme.models.layers.rbf import SincRBFLayer as F1
    from spookynet.modules.sinc_functions import SincFunctions as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_allclose(
        f1.get_rbf(D, cutoff_values).detach().numpy(), 
        f2(D, cutoff_values).detach().numpy()
    )


def test_local_interaction():
    from enerzyme.models.spookynet.interaction import LocalInteraction as F1
    from spookynet.modules.local_interaction import LocalInteraction as F2
    f1 = F1(
        dim_feature, dim_feature, 
        num_residual_local_x, num_residual_local_s, num_residual_local_p, num_residual_local_d, num_residual_local
    ).type(dtype)
    f2 = F2(dim_feature, dim_feature, 
        num_residual_local_x, num_residual_local_s, num_residual_local_p, num_residual_local_d, num_residual_local
    ).type(dtype)
    assert_allclose(
        f1(atom_embedding, rbf, pij, dij, idx_i, idx_j).detach().numpy(),
        f2(atom_embedding, rbf, pij, dij, idx_i, idx_j).detach().numpy()
    )


def test_attention():
    from enerzyme.models.layers.attention import Attention as F1
    from spookynet.modules.attention import Attention as F2
    f2 = F2(dim_feature, dim_feature, dim_feature).type(dtype)
    f1 = F1(dim_feature, dim_feature).type(dtype)
    f1.omega.copy_(f2.omega)
    assert_allclose(
        f1(Q, K, atom_embedding, 1, None).detach().numpy(),
        f2(Q, K, atom_embedding, 1, None).detach().numpy()
    )


def test_nonlocal_interaction():
    from enerzyme.models.spookynet.interaction import NonlocalInteraction as F1
    from spookynet.modules.nonlocal_interaction import NonlocalInteraction as F2
    f2 = F2(dim_feature, num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v).type(dtype)
    f1 = F1(dim_feature, num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v).type(dtype)
    f1.attention.omega.copy_(f2.attention.omega)
    assert_allclose(
        f1(atom_embedding, 1, None).detach().numpy(),
        f2(atom_embedding, 1, None).detach().numpy()
    )


def test_interaction_module():
    from enerzyme.models.spookynet.interaction import InteractionModule as F1
    from spookynet.modules.interaction_module import InteractionModule as F2
    f2 = F2(
        dim_feature, dim_feature, 
        num_residual_pre, num_residual_local_x, num_residual_local_s, num_residual_local_p, num_residual_local_d, num_residual_local,
        num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v, num_residual_post, num_residual_output
    ).type(dtype)
    f1 = F1(
        dim_feature, dim_feature, 
        num_residual_pre, num_residual_local_x, num_residual_local_s, num_residual_local_p, num_residual_local_d, num_residual_local,
        num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v, num_residual_post, num_residual_output
    ).type(dtype)
    f1.nonlocal_interaction.attention.omega.copy_(f2.nonlocal_interaction.attention.omega)
    with torch.no_grad():
        f1.resblock.stack[-1].weight.copy_(f2.resblock.linear.weight)
    x1, y1 = f1(atom_embedding, rbf, pij, dij, idx_i, idx_j, 1, None)
    x2, y2 = f2(atom_embedding, rbf, pij, dij, idx_i, idx_j, 1, None)
    assert_allclose(x1.detach().numpy(), x2.detach().numpy())
    assert_allclose(y1.detach().numpy(), y2.detach().numpy())


def test_zbl_repulsion_energy():
    from enerzyme.models.layers.zbl import ZBLRepulsionEnergyLayer as F1
    from spookynet.modules.zbl_repulsion_energy import ZBLRepulsionEnergy as F2
    f1 = F1().type(dtype)
    f2 = F2().type(dtype)
    f1.kehalf = f2.kehalf
    f1.a0 = f2.a0
    assert_allclose(
        f1.get_zbf_energy(Za, D, idx_i, idx_j, cutoff_values).detach().numpy(),
        f2(len(Za), Za.type(dtype), D, cutoff_values, idx_i, idx_j).detach().numpy()
    )


def test_switch_function():
    from enerzyme.models.cutoff import smooth_transition
    from spookynet.functional import switch_function
    assert_allclose(
        smooth_transition(D, 10, 1).detach().numpy(),
        switch_function(D, 1, 10).detach().numpy()
    )


def test_electrostatic_energy():
    from enerzyme.models.layers.electrostatics import ElectrostaticEnergyLayer as F1
    from spookynet.modules.electrostatic_energy import ElectrostaticEnergy as F2
    f2 = F2(cutoff=5.0)
    f1 = F1(cutoff_sr=5.0, lr_flavor="smooth")
    f1.kehalf = f2.kehalf
    assert_allclose(
        f1.get_E_ele_a(D, Qa, idx_i, idx_j).detach().numpy(),
        f2(len(Qa), Qa, D, idx_i, idx_j).detach().numpy()
    )


def test_d4_dispersion_energy():
    from enerzyme.models.layers.dispersion.grimme_d4 import GrimmeD4EnergyLayer as F1
    from spookynet.modules.d4_dispersion_energy import D4DispersionEnergy as F2
    f2 = F2()
    f1 = F1()
    f1.Hartree_in_E = f2.convert2eV * 2
    f1.Bohr_in_R = 1 / f2.convert2Bohr
    assert_allclose(
        f1.get_e_disp(Za, Qa, D, idx_i, idx_j).detach().numpy(),
        f2(N, Za, Qa, D, idx_i, idx_j)[0].detach().numpy()
    )

def test_calculate_distances():
    from enerzyme.models.layers.geometry import DistanceLayer as F1
    from spookynet.spookynet import SpookyNet as F2
    f1 = F1()
    f2 = F2()
    assert_allclose(
        f1.get_distance(R, idx_i, idx_j).detach().numpy(),
        f2.calculate_distances(R, idx_i, idx_j).detach().numpy()
    )


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