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
from torch.nn import Sequential
from enerzyme.models.spookynet import SpookyNetCore
from enerzyme.models.spookynet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS


dim_feature = DEFAULT_BUILD_PARAMS["dim_embedding"]
initial_alpha = np.random.randn()
initial_beta = np.random.randn()
x = torch.randn(dim_feature)
max_Za = DEFAULT_BUILD_PARAMS["max_Za"]
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
R = torch.rand(N, 3) * 20
D = torch.norm(vij, dim=-1, keepdim=True)
pij = vij / D
D = D.squeeze(-1)
dij = torch.rand(*idx_i.shape, 5)
Za = torch.randint(1, max_Za, (N,))
rbf = torch.rand(*idx_i.shape, dim_feature)
cutoff_values = torch.rand(*idx_i.shape)
atom_embedding = torch.randn((N, dim_feature))
q = torch.randn((N, dim_feature))
Qa = torch.randn((N,))
K = torch.randn((N, dim_feature))
dtype = "float64"
if dtype == "float64":
    dtype = torch.float64
elif dtype == "float32":
    dtype = torch.float32
x = x.type(dtype)
q = q.type(dtype)
Qa = Qa.type(dtype)
K = K.type(dtype)
atom_embedding = atom_embedding.type(dtype)
D = D.type(dtype)
R = R.type(dtype)
pij = pij.type(dtype)
dij = dij.type(dtype)
rbf = rbf.type(dtype)
cutoff_values = cutoff_values.type(dtype)
Q = torch.tensor([-1], dtype=dtype)
S = torch.tensor([1], dtype=dtype)

def assert_tensor_allclose(t1, t2, **kwargs):
    assert_allclose(t1.detach().numpy(), t2.detach().numpy(), **kwargs)


def initialize():
    global DEFAULT_BUILD_PARAMS
    from spookynet.modules.d4_dispersion_energy import D4DispersionEnergy
    d4 = D4DispersionEnergy()
    DEFAULT_BUILD_PARAMS.update({
        "Bohr_in_R": 1 / d4.convert2Bohr,
        "Hartree_in_E": d4.convert2eV * 2,
    })
    from enerzyme.models.ff import build_model
    from spookynet.spookynet import SpookyNet as F2
    f1: SpookyNetCore = build_model(
        "SpookyNet", 
        DEFAULT_LAYER_PARAMS, 
        DEFAULT_BUILD_PARAMS
    ).type(dtype)
    f2 = F2(use_zbl_repulsion=True, use_electrostatics=True, use_d4_dispersion=True).type(dtype)
    return f1, f2


def test_initialize():
    initialize()


def test_cutoff_function():
    from enerzyme.models.cutoff import bump_transition
    from spookynet.functional import cutoff_function
    assert_tensor_allclose(bump_transition(D, 5), cutoff_function(D, 5))


def test_shifted_softplus():
    from enerzyme.models.activation import ShiftedSoftplus as F1
    from spookynet.modules.shifted_softplus import ShiftedSoftplus as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta).type(dtype)
    f2 = F2(dim_feature, initial_alpha, initial_beta).type(dtype)
    assert_tensor_allclose(f1(x), f2(x))


def test_swish():
    from enerzyme.models.activation import Swish as F1
    from spookynet.modules.swish import Swish as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta).type(dtype)
    f2 = F2(dim_feature, initial_alpha, initial_beta).type(dtype)
    assert_tensor_allclose(f1(x), f2(x))


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
    assert_tensor_allclose(f1(x), f2(x))


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
    assert_tensor_allclose(f1(x), f2(x))


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
    assert_tensor_allclose(f1(x), f2(x))


def test_nuclear_embedding():
    from enerzyme.models.layers.atom_embedding import NuclearEmbedding as F1
    from spookynet.modules.nuclear_embedding import NuclearEmbedding as F2
    f1 = F1(86, dim_feature).type(dtype)
    f2 = F2(dim_feature).type(dtype)
    assert_tensor_allclose(f1.get_embedding(Za), f2(Za))


def test_electronic_embedding():
    from enerzyme.models.layers.electron_embedding import ElectronicEmbedding as F1
    from spookynet.modules.electronic_embedding import ElectronicEmbedding as F2
    f1 = F1(dim_feature, 3, attribute="spin").type(dtype)
    f2 = F2(dim_feature, 3).type(dtype)
    S = torch.tensor([2], dtype=dtype)
    assert_tensor_allclose(
        f1.get_electron_embedding(atom_embedding=atom_embedding, S=S), 
        f2(atom_embedding, S, 1, None)
    )
    pass


def test_nonlinear_electronic_embedding():
    from enerzyme.models.layers.electron_embedding import NonlinearElectronicEmbedding as F1
    from spookynet.modules.nonlinear_electronic_embedding import NonlinearElectronicEmbedding as F2
    f1 = F1(dim_feature, 3, attribute="spin").type(dtype)
    f2 = F2(dim_feature, 3).type(dtype)
    S = torch.tensor([2], dtype=dtype)
    assert_tensor_allclose(
        f1.get_electron_embedding(atom_embedding=atom_embedding, S=S), 
        f2(atom_embedding, S, 1, torch.zeros(N, dtype=torch.long))
    )
    pass


def test_exponential_gaussian_functions():
    from enerzyme.models.layers.rbf import ExponentialGaussianRBFLayer as F1
    from spookynet.modules.exponential_gaussian_functions import ExponentialGaussianFunctions as F2
    f1 = F1(dim_feature, init_width_flavor="SpookyNet").type(dtype)
    f2 = F2(dim_feature).type(dtype)
    assert_tensor_allclose(
        f1.get_rbf(D, cutoff_values), 
        f2(D, cutoff_values), rtol=1e-6, atol=1e-6
    )


def test_exponential_bernstein_polynomials():
    from enerzyme.models.layers.rbf import ExponentialBernsteinRBFLayer as F1
    from spookynet.modules.exponential_bernstein_polynomials import ExponentialBernsteinPolynomials as F2
    f1 = F1(dim_feature).type(dtype).type(dtype)
    f2 = F2(dim_feature).type(dtype).type(dtype)
    assert_tensor_allclose(
        f1.get_rbf(D, cutoff_values), 
        f2(D, cutoff_values), rtol=1e-6, atol=1e-6
    )


def test_gaussian_functions():
    from enerzyme.models.layers.rbf import GaussianRBFLayer as F1
    from spookynet.modules.gaussian_functions import GaussianFunctions as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_tensor_allclose(
        f1.get_rbf(D, cutoff_values), 
        f2(D, cutoff_values)
    )


def test_bernstein_polynomials():
    from enerzyme.models.layers.rbf import BernsteinRBFLayer as F1
    from spookynet.modules.bernstein_polynomials import BernsteinPolynomials as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_tensor_allclose(
        f1.get_rbf(D, cutoff_values), 
        f2(D, cutoff_values)
    )


def test_sinc_functions():
    from enerzyme.models.layers.rbf import SincRBFLayer as F1
    from spookynet.modules.sinc_functions import SincFunctions as F2
    f1 = F1(dim_feature, 5.0).type(dtype)
    f2 = F2(dim_feature, 5.0).type(dtype)
    assert_tensor_allclose(
        f1.get_rbf(D, cutoff_values), 
        f2(D, cutoff_values)
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
    f2.radial_s.weight.data.copy_(f1.radial_s.weight.data)
    f2.radial_p.weight.data.copy_(f1.radial_p.weight.data)
    f2.radial_d.weight.data.copy_(f1.radial_d.weight.data)
    f2.resblock.linear.weight.data.copy_(f1.resblock.output.weight.data)
    f2.resblock_x.linear.weight.data.copy_(f1.resblock_x.output.weight.data)
    f2.resblock_p.linear.weight.data.copy_(f1.resblock_p.output.weight.data)
    f2.resblock_s.linear.weight.data.copy_(f1.resblock_s.output.weight.data)
    f2.resblock_d.linear.weight.data.copy_(f1.resblock_d.output.weight.data)
    f2.projection_d.weight.data.copy_(f1.projection_d.weight.data)
    f2.projection_p.weight.data.copy_(f1.projection_p.weight.data)
    assert_tensor_allclose(
        f1(atom_embedding, rbf, pij, dij, idx_i, idx_j),
        f2(atom_embedding, rbf, pij, dij, idx_i, idx_j)
    )


def test_attention():
    from enerzyme.models.layers.attention import Attention as F1
    from spookynet.modules.attention import Attention as F2
    f2 = F2(dim_feature, dim_feature, dim_feature).type(dtype)
    f1 = F1(dim_feature, dim_feature).type(dtype)
    f1.omega.copy_(f2.omega)
    assert_tensor_allclose(
        f1(q, K, atom_embedding, 1, None),
        f2(q, K, atom_embedding, 1, None)
    )


def test_nonlocal_interaction():
    from enerzyme.models.spookynet.interaction import NonlocalInteraction as F1
    from spookynet.modules.nonlocal_interaction import NonlocalInteraction as F2
    f2 = F2(dim_feature, num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v).type(dtype)
    f1 = F1(dim_feature, num_residual_nonlocal_q, num_residual_nonlocal_k, num_residual_nonlocal_v).type(dtype)
    f1.attention.omega.copy_(f2.attention.omega)
    assert_tensor_allclose(
        f1(atom_embedding, 1, None),
        f2(atom_embedding, 1, None)
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
    g1 = f1.local_interaction
    g2 = f2.local_interaction
    g2.radial_s.weight.data.copy_(g1.radial_s.weight.data)
    g2.radial_p.weight.data.copy_(g1.radial_p.weight.data)
    g2.radial_d.weight.data.copy_(g1.radial_d.weight.data)
    g2.resblock.linear.weight.data.copy_(g1.resblock.output.weight.data)
    g2.resblock_x.linear.weight.data.copy_(g1.resblock_x.output.weight.data)
    g2.resblock_p.linear.weight.data.copy_(g1.resblock_p.output.weight.data)
    g2.resblock_s.linear.weight.data.copy_(g1.resblock_s.output.weight.data)
    g2.resblock_d.linear.weight.data.copy_(g1.resblock_d.output.weight.data)
    g2.projection_d.weight.data.copy_(g1.projection_d.weight.data)
    g2.projection_p.weight.data.copy_(g1.projection_p.weight.data)
    f1.nonlocal_interaction.attention.omega.copy_(f2.nonlocal_interaction.attention.omega)
    with torch.no_grad():
        f1.resblock.output.weight.copy_(f2.resblock.linear.weight)
        f1.resblock.output.bias.copy_(f2.resblock.linear.bias)
    x1, y1 = f1(atom_embedding, rbf, pij, dij, idx_i, idx_j, 1, None)
    x2, y2 = f2(atom_embedding, rbf, pij, dij, idx_i, idx_j, 1, None)
    assert_tensor_allclose(x1, x2)
    assert_tensor_allclose(y1, y2)


def test_zbl_repulsion_energy():
    from enerzyme.models.layers.zbl import ZBLRepulsionEnergyLayer as F1
    from spookynet.modules.zbl_repulsion_energy import ZBLRepulsionEnergy as F2
    f1 = F1().type(dtype)
    f2 = F2().type(dtype)
    f1.kehalf = f2.kehalf
    f1.a0 = f2.a0
    assert_tensor_allclose(
        f1.get_E_zbl_a(Za, D, idx_i, idx_j, cutoff_values),
        f2(len(Za), Za.type(dtype), D, cutoff_values, idx_i, idx_j)
    )


def test_switch_function():
    from enerzyme.models.cutoff import smooth_transition
    from spookynet.functional import switch_function
    assert_tensor_allclose(
        smooth_transition(D, 10, 1),
        switch_function(D, 1, 10)
    )


def test_electrostatic_energy():
    from enerzyme.models.layers.electrostatics import ElectrostaticEnergyLayer as F1
    from spookynet.modules.electrostatic_energy import ElectrostaticEnergy as F2
    f2 = F2(cutoff=5.0 * 0.75, cuton=5.0 * 0.25)
    f1 = F1(cutoff_sr=5.0)
    f1.kehalf = f2.kehalf
    assert_tensor_allclose(
        f1.get_E_ele_a(D, Qa, idx_i, idx_j),
        f2(len(Qa), Qa, D, idx_i, idx_j)
    )


def test_d4_dispersion_energy():
    from enerzyme.models.layers.dispersion.grimme_d4 import GrimmeD4EnergyLayer as F1
    from spookynet.modules.d4_dispersion_energy import D4DispersionEnergy as F2
    f2 = F2()
    f1 = F1()
    f1.Hartree_in_E = f2.convert2eV * 2
    f1.Bohr_in_R = 1 / f2.convert2Bohr
    assert_tensor_allclose(
        f1.get_E_disp_a(Za, Qa, D, idx_i, idx_j),
        f2(N, Za, Qa, D, idx_i, idx_j)[0]
    )


def test_calculate_distances():
    from enerzyme.models.layers.geometry import DistanceLayer as F1
    from spookynet.spookynet import SpookyNet as F2
    f1 = F1()
    f1.with_vector_on()
    f2 = F2()
    output1 = f1.get_output(R, idx_i, idx_j)
    Dij1 = output1["Dij"]
    vij1 = output1["vij"]
    Dij2, vij2 = f2.calculate_distances(R, idx_i, idx_j)
    assert_tensor_allclose(Dij1, Dij2)
    assert_tensor_allclose(vij1, vij2)


def test_atomic_properties_static():
    f1, f2 = initialize()
    net_input = {"Ra": R, "idx_i": idx_i, "idx_j": idx_j}
    output = f1.range_separation(f1.calculate_distance(net_input))
    pij1, dij1, _, _ = f1._atomic_properties_static(output["Dij_sr"], output["vij_sr"])
    _, _, _, _, pij2, dij2, _, _, _ = f2._atomic_properties_static(Za, R, idx_i, idx_j)
    assert_tensor_allclose(pij1, pij2)
    assert_tensor_allclose(dij1, dij2)
    

def test_atomic_properties_dynamic():
    f1, f2 = initialize()
    N_, cutoff_values_, Dij_lr, Dij_sr, pij, dij, idx_i_sr, idx_j_sr, _ = f2._atomic_properties_static(Za, R, idx_i, idx_j)
    _, ea2, qa2, _, _, _, _, _ = f2._atomic_properties_dynamic(
        N_, Q, S, Za, R, cutoff_values_, Dij_lr, idx_i, idx_j, Dij_sr, pij, dij, idx_i_sr, idx_j_sr
    )
    net_input = {"Ra": R, "idx_i": idx_i, "idx_j": idx_j, "Za": Za, "Q": Q, "S": S}
    pre_layers = Sequential(
        f1.calculate_distance, f1.range_separation, f1.atom_embedding, f1.charge_embedding, f1.spin_embedding, f1.radial_basis_function
    )
    output = pre_layers(net_input)
    ea1, qa1 = f1._atomic_properties_dynamic(
        output["atom_embedding"],
        output["charge_embedding"],
        output["spin_embedding"],
        1,
        output["rbf"],
        pij, dij, output["idx_i_sr"], output["idx_j_sr"], None
    )
    qa1 = f1.charge_conservation.get_output(Za, qa1, Q)["Qa"]
    assert_tensor_allclose(ea1, ea2)
    assert_tensor_allclose(qa1, qa2)
    pass


def test_atomic_properties():
    f1, f2 = initialize()
    _, ea2, qa2, _, _, _, _, _ = f2.atomic_properties(Za, Q, S, R, idx_i, idx_j)
    net_input = {"Ra": R, "idx_i": idx_i, "idx_j": idx_j, "Za": Za, "Q": Q, "S": S}
    pre_layers = Sequential(
        f1.calculate_distance, f1.range_separation, f1.atom_embedding, f1.charge_embedding, f1.spin_embedding, f1.radial_basis_function
    )
    output = pre_layers(net_input)
    pij1, dij1, _, _ = f1._atomic_properties_static(
        output["Dij_sr"],
        output["vij_sr"],
    )
    ea1, qa1 = f1._atomic_properties_dynamic(
        output["atom_embedding"],
        output["charge_embedding"],
        output["spin_embedding"],
        1,
        output["rbf"],
        pij1, dij1, output["idx_i_sr"], output["idx_j_sr"], None
    )
    qa1 = f1.charge_conservation.get_output(Za, qa1, Q)["Qa"]
    assert_tensor_allclose(ea1, ea2)
    assert_tensor_allclose(qa1, qa2)


def test_forward():
    torch.autograd.set_detect_anomaly(True)
    f1, f2 = initialize()
    E2, Fa2, M22, _, ea, qa, ea_rep, ea_ele, ea_vdw, *_ = f2.forward(Za, Q, S, R.requires_grad_(), idx_i, idx_j)
    net_input = {"Ra": R.requires_grad_(), "idx_i": idx_i, "idx_j": idx_j, "Za": Za, "Q": Q, "S": S}
    output = f1(net_input)
    E1 = output["E"]
    Fa1 = output["Fa"]
    M21 = output["M2"]
    assert_tensor_allclose(E1, E2)
    assert_tensor_allclose(Fa1, Fa2)
    assert_tensor_allclose(M21, M22)

if __name__ == "__main__":
    test_forward()
