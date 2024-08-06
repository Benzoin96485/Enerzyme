import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
from numpy.testing import assert_allclose
from numpy.random import get_state, set_state
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.extend(["..", "."])
from enerzyme.models.ff import build_model
# from enerzyme.models.physnet import PhysNetCore
from physnet.NeuralNetwork import NeuralNetwork
from physnet.grimme_d3.grimme_d3 import d3_autoang, d3_autoev
from enerzyme.models.init import semi_orthogonal_glorot_weights

F = 128
K = 5
N = 50
M = 20
cutoff=20
R = np.random.rand(N, 3) * 20
idx_i = np.empty((N, N-1), dtype=int)
idx_j = np.empty((N, N-1), dtype=int)
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
offsets = np.random.random((*idx_i.shape, 3))
x = np.random.randn(N, F)
W_init = semi_orthogonal_glorot_weights(F, F) * 10000
b_init = np.random.randn(F)
D = np.random.random(*idx_i.shape) * 30 + 3
rbf = np.random.randn(*idx_i.shape, K)
Z = np.random.randint(0, 94, N)
Qa = np.random.randn(N)
Ea = np.random.randn(N)
Q_tot = np.random.randn()
nci = np.random.rand(*idx_i.shape) * 5
ncj = np.random.rand(*idx_j.shape) * 5
# def set_dtype(dtype):
dtype = "float64"
if dtype == "float64":
    dtype_torch = torch.float64
    dtype_tf = tf.float64
    R = R.astype(np.float64)
    offsets = offsets.astype(np.float64)
    W_init = W_init.type(dtype_torch)
    b_init = b_init.astype(np.float64)
elif dtype == "float32":
    dtype_torch = torch.float32
    dtype_tf = tf.float32
    R = R.astype(np.float32)
    offsets = offsets.astype(np.float32)
    W_init = W_init.type(dtype_torch)
    b_init = b_init.astype(np.float32)

# set_dtype("float64")

default_layer_params = [
    {"name": "Distance"},
    {
        "name": "ExponentialGaussianRBF", 
        "params": {
            "no_basis_at_infinity": False,
            "init_alpha": 1,
            "exp_weighting": False,
            "learnable_shape": True,
            "cutoff_fn": "polynomial",
            "init_width_flavor": "PhysNet"
        }
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core"}
]

def initialize(layer_params=default_layer_params):
    state = get_state()
    physnet_torch = build_model(
        architecture="PhysNet",
        build_params={
            "dim_embedding": F,
            "num_rbf": K,
            "max_Za": 94,
            "cutoff_sr": cutoff,
            "drop_out": 0.0,
            "Hartree_in_E": d3_autoev,
            "Bohr_in_R": d3_autoang
        },
        layer_params=layer_params
    ).type(dtype_torch)
    # physnet_torch = None
    # physnet_torch = PhysNetCore(F, K, 5.0, dtype=dtype, use_dispersion=use_dispersion)
    set_state(state)
    physnet_tf = NeuralNetwork(F, K, cutoff, scope="test", dtype=dtype_tf)
    physnet_tf._embeddings = tf.Variable(physnet_torch.core.embeddings.weight.detach().numpy(), name="embeddings", dtype=dtype_tf)
    return physnet_torch, physnet_tf
    

def test_initialize():
    initialize()


def test_calculate_interatomic_distances():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.geometry import DistanceLayer
    D1 = DistanceLayer().get_distance(
        Ra=torch.from_numpy(R), 
        idx_i=torch.from_numpy(idx_i), 
        idx_j=torch.from_numpy(idx_j), 
        offsets=torch.from_numpy(offsets)
    ).numpy()
    with tf.Session() as sess:
        D2 = physnet_tf.calculate_interatomic_distances(
            R,
            idx_i,
            idx_j,
            offsets
        ).eval()    
    assert_allclose(D1, D2)


def test_RBFLayer():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.rbf import ExponentialGaussianRBFLayer
    from enerzyme.models.layers.geometry import DistanceLayer
    D1 = DistanceLayer().get_distance(
        Ra=torch.from_numpy(R), 
        idx_i=torch.from_numpy(idx_i), 
        idx_j=torch.from_numpy(idx_j), 
        offsets=torch.from_numpy(offsets)
    ).numpy()
    rbf_layer = ExponentialGaussianRBFLayer(
        num_rbf=K,
        no_basis_at_infinity=False,
        init_alpha=1,
        exp_weighting=False,
        learnable_shape=True,
        cutoff_sr=cutoff,
        cutoff_fn="polynomial",
        init_width_flavor="PhysNet"
    ).type(dtype_torch)
    rbf1 = rbf_layer.get_rbf(torch.from_numpy(D1)).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rbf2 = physnet_tf.rbf_layer(D1).eval()
    assert_allclose(rbf1, rbf2, atol=1e-7, rtol=1e-7)


def test_embedding():
    from enerzyme.models.layers.atom_embedding import RandomAtomEmbedding
    embedding = RandomAtomEmbedding(95, F).type(dtype_torch)
    embedding.get_embedding(torch.from_numpy(Z))


def test_DenseLayer():
    from enerzyme.models.physnet.interaction import DenseLayer as DenseLayer_torch
    from physnet.layers.DenseLayer import DenseLayer as Denselayer_tf
    
    dense_layer_torch = DenseLayer_torch(F, F, initial_weight=W_init, initial_bias=torch.from_numpy(b_init)).type(dtype_torch)
    dense_layer_tf = Denselayer_tf(F, F, W_init=W_init.T.detach().numpy(), b_init=b_init, scope="test", dtype=dtype_tf)
    y_dense_torch = dense_layer_torch(torch.from_numpy(x)).detach().numpy()
    l2loss_torch = dense_layer_torch.l2loss().detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_dense_tf = dense_layer_tf(x).eval()
        l2loss_tf = dense_layer_tf.l2loss.eval()
    assert_allclose(y_dense_torch, y_dense_tf)
    assert_allclose(l2loss_torch, l2loss_tf)


def test_ResidualLayer():
    from enerzyme.models.layers.mlp import ResidualLayer as ResidualLayer_torch
    from physnet.layers.ResidualLayer import ResidualLayer as ResidualLayer_tf
    residual_layer_torch = ResidualLayer_torch(F, F, initial_weight1=W_init, initial_weight2=W_init, initial_bias=torch.from_numpy(b_init)).type(dtype_torch)
    residual_layer_tf = ResidualLayer_tf(F, F, W_init=W_init.T.detach().numpy(), b_init=b_init, scope="test", dtype=dtype_tf)
    y_residual_torch = residual_layer_torch(torch.from_numpy(x.copy())).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_residual_tf = residual_layer_tf(x.copy()).eval()
    assert_allclose(y_residual_torch, y_residual_tf)


def test_InteractionLayer():
    from enerzyme.models.physnet.interaction import InteractionLayer as InteractionLayer_torch
    from physnet.layers.InteractionLayer import InteractionLayer as InteractionLayer_tf
    state = get_state()
    interaction_layer_torch = InteractionLayer_torch(K, F, 3).type(dtype_torch)
    set_state(state)
    interaction_layer_tf = InteractionLayer_tf(K, F, 3, scope="test", dtype=dtype_tf)
    y_interaction_torch = interaction_layer_torch(
        torch.from_numpy(x.copy()), 
        torch.from_numpy(rbf.copy()), 
        torch.from_numpy(idx_i), 
        torch.from_numpy(idx_j)
    ).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_interaction_tf = interaction_layer_tf(x.copy(), rbf.copy(), idx_i, idx_j).eval()
    assert_allclose(y_interaction_torch, y_interaction_tf)


def test_InteractionBlock():
    from enerzyme.models.physnet.interaction import InteractionBlock as InteractionBlock_torch
    from physnet.layers.InteractionBlock import InteractionBlock as InteractionBlock_tf
    state = get_state()
    interaction_block_torch = InteractionBlock_torch(K, F, 3, 3).type(dtype_torch)
    set_state(state)
    interaction_block_tf = InteractionBlock_tf(K, F, 3, 3, scope="test", dtype=dtype_tf)
    y_interaction_torch = interaction_block_torch(
        torch.from_numpy(x.copy()), 
        torch.from_numpy(rbf.copy()), 
        torch.from_numpy(idx_i), 
        torch.from_numpy(idx_j)
    ).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_interaction_tf = interaction_block_tf(x.copy(), rbf.copy(), idx_i, idx_j).eval()
    assert_allclose(y_interaction_torch, y_interaction_tf)


def test_OutputBlock():
    from enerzyme.models.physnet.interaction import OutputBlock as OutputBlock_torch
    from physnet.layers.OutputBlock import OutputBlock as OutputBlock_tf
    state = get_state()
    output_block_torch = OutputBlock_torch(F, 3).type(dtype_torch)
    set_state(state)
    output_block_tf = OutputBlock_tf(F, 3, scope="test", dtype=dtype_tf)
    y_output_torch = output_block_torch(
        torch.from_numpy(x.copy())
    ).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_output_tf = output_block_tf(x.copy()).eval()
    assert_allclose(y_output_torch, y_output_tf)


def test_atomic_properties():
    physnet_torch, physnet_tf = initialize()
    output = physnet_torch({
        "Za": torch.from_numpy(Z.copy()), 
        "Ra": torch.from_numpy(R), 
        "idx_i": torch.from_numpy(idx_i.copy()), 
        "idx_j": torch.from_numpy(idx_j.copy()), 
        "offsets": torch.from_numpy(offsets)
    })
    Ea_torch = output["Ea"].detach().numpy()
    Qa_torch = output["Qa"].detach().numpy()
    Dij_lr_torch = output["Dij"].detach().numpy()
    nhloss_torch = output["nh_loss"].detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Ea_tf, Qa_tf, Dij_lr_tf, nhloss_tf = physnet_tf.atomic_properties(
            Z, R, idx_i, idx_j, offsets
        )
        Ea_tf = Ea_tf.eval()
        Qa_tf = Qa_tf.eval()
        Dij_lr_tf = Dij_lr_tf.eval()
        nhloss_tf = nhloss_tf.eval()
    assert_allclose(Ea_torch, Ea_tf)
    assert_allclose(Qa_torch, Qa_tf)
    assert_allclose(Dij_lr_torch, Dij_lr_tf)
    assert_allclose(nhloss_torch, nhloss_tf)


def test_ncoord():
    from enerzyme.models.layers.dispersion.grimme_d3 import _ncoord as f1
    from physnet.grimme_d3.grimme_d3 import _ncoord as f2
    cn_torch = f1(
        Zi=torch.from_numpy(Z)[torch.from_numpy(idx_i)],
        Zj=torch.from_numpy(Z)[torch.from_numpy(idx_j)], 
        Dij=torch.from_numpy(D),
        idx_i=torch.from_numpy(idx_i),
        cutoff=2
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cn_tf = f2(Z[idx_i], Z[idx_j], D, idx_i, 2).eval()
    assert_allclose(cn_torch, cn_tf)


def test_getc6():
    from enerzyme.models.layers.dispersion.grimme_d3 import _getc6 as f1
    from physnet.grimme_d3.grimme_d3 import _getc6 as f2
    c6_torch = f1(
        Zi=torch.from_numpy(Z)[torch.from_numpy(idx_i)],
        Zj=torch.from_numpy(Z)[torch.from_numpy(idx_j)],
        nci=torch.from_numpy(nci),
        ncj=torch.from_numpy(ncj)
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c6_tf = f2(
            np.stack([Z[idx_i], Z[idx_j]], axis=1),
            nci, ncj
        ).eval()
    assert_allclose(c6_torch, c6_tf)


def test_edisp():
    from enerzyme.models.layers.dispersion.grimme_d3 import GrimmeD3EnergyLayer
    from physnet.grimme_d3.grimme_d3 import edisp as edisp_tf
    disp_layer = GrimmeD3EnergyLayer(Hartree_in_E=1, Bohr_in_R=1)
    e_torch = disp_layer.get_e_disp(torch.from_numpy(Z.copy()), torch.from_numpy(D.copy()), torch.from_numpy(idx_i), torch.from_numpy(idx_j)).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        e_tf = edisp_tf(Z, D, idx_i, idx_j).eval()
    assert_allclose(e_torch, e_tf, rtol=1e-7, atol=1e-7)


def test_electrostatic_energy_per_atom():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.electrostatics import ElectrostaticEnergyLayer
    ele_layer = ElectrostaticEnergyLayer(
        cutoff_sr=cutoff,
        cutoff_lr=None,
        cutoff_fn="polynomial"
    )
    ele_layer.kehalf = physnet_tf.kehalf
    e_torch = ele_layer.get_E_ele_a(
        Dij=torch.from_numpy(D.copy()), 
        Qa=torch.from_numpy(Qa.copy()), 
        idx_i=torch.from_numpy(idx_i), 
        idx_j=torch.from_numpy(idx_j)
    ).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        e_tf = physnet_tf.electrostatic_energy_per_atom(D, Qa, idx_i, idx_j).eval()
    assert_allclose(e_torch, e_tf)


def test_energy_from_scaled_atomic_properties():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.dispersion.grimme_d3 import GrimmeD3EnergyLayer
    from enerzyme.models.layers.electrostatics import ElectrostaticEnergyLayer
    from enerzyme.models.layers.reduce import EnergyReduceLayer
    ele_layer = ElectrostaticEnergyLayer(
        cutoff_sr=cutoff,
        cutoff_lr=None,
        cutoff_fn="polynomial"
    )
    ele_layer.kehalf = physnet_tf.kehalf
    disp_layer = GrimmeD3EnergyLayer(Bohr_in_R=d3_autoang, Hartree_in_E=d3_autoev)
    reduce_layer = EnergyReduceLayer()
    e_torch = reduce_layer(disp_layer(ele_layer({
        "Ea": torch.from_numpy(Ea.copy()), 
        "Qa": torch.from_numpy(Qa.copy()), 
        "Dij": torch.from_numpy(D.copy()),
        "Za": torch.from_numpy(Z.copy()),
        "idx_i": torch.from_numpy(idx_i), 
        "idx_j": torch.from_numpy(idx_j)
    })))["E"]
    e_torch = e_torch.detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        e_tf = physnet_tf.energy_from_scaled_atomic_properties(Ea, Qa, D, Z, idx_i, idx_j).eval()
    assert_allclose(e_torch, e_tf)


def test_scaled_charges():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.electrostatics import ChargeConservationLayer
    Q_layer = ChargeConservationLayer()
    q_torch = Q_layer.get_corrected_Qa(
        torch.from_numpy(Z.copy()),
        torch.from_numpy(Qa.copy()), 
        torch.tensor(Q_tot)
    )["Qa"].detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        q_tf = physnet_tf.scaled_charges(Z, Qa, Q_tot).eval()
    assert_allclose(q_torch, q_tf, rtol=1e-7, atol=1e-7)


def test_energy_from_atomic_properties():
    _, physnet_tf = initialize()
    from enerzyme.models.layers.dispersion.grimme_d3 import GrimmeD3EnergyLayer
    from enerzyme.models.layers.electrostatics import ElectrostaticEnergyLayer, ChargeConservationLayer
    from enerzyme.models.layers.reduce import EnergyReduceLayer
    Q_layer = ChargeConservationLayer()
    ele_layer = ElectrostaticEnergyLayer(
        cutoff_sr=cutoff,
        cutoff_lr=None,
        cutoff_fn="polynomial"
    )
    ele_layer.kehalf = physnet_tf.kehalf
    disp_layer = GrimmeD3EnergyLayer(Bohr_in_R=d3_autoang, Hartree_in_E=d3_autoev)
    reduce_layer = EnergyReduceLayer()
    e_torch = reduce_layer(disp_layer(ele_layer(Q_layer({
        "Ea": torch.from_numpy(Ea.copy()), 
        "Qa": torch.from_numpy(Qa.copy()), 
        "Dij": torch.from_numpy(D.copy()),
        "Za": torch.from_numpy(Z.copy()),
        "idx_i": torch.from_numpy(idx_i), 
        "idx_j": torch.from_numpy(idx_j),
        "Q": torch.tensor(Q_tot)
    }))))["E"]
    e_torch = e_torch.detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        e_tf = physnet_tf.energy_from_atomic_properties(Ea, Qa, D, Z, idx_i, idx_j, Q_tot).eval()
    assert_allclose(e_torch, e_tf)
    pass


def test_energy_and_forces():
    torch.autograd.set_detect_anomaly(True)
    physnet_torch, physnet_tf = initialize(default_layer_params + [
        {"name": "AtomicAffine", "params": {
            "shifts": {
                "Ea": {"values": 0, "learnable": True},
                "Qa": {"values": 0, "learnable": True}
            },
            "scales": {
                "Ea": {"values": 1, "learnable": True},
                "Qa": {"values": 1, "learnable": True}
            }
        }},
        {"name": "ChargeConservation"},
        {"name": "AtomicCharge2Dipole"},
        {"name": "ElectrostaticEnergy", "params": {"cutoff_fn": "polynomial"}},
        {"name": "GrimmeD3Energy", "params": {"learnable": True}},
        {"name": "EnergyReduce"},
        {"name": "Force"}
    ])
    output = physnet_torch({
        "Za": torch.from_numpy(Z.copy()),
        "Ra": torch.tensor(R, requires_grad=True),
        "idx_i": torch.from_numpy(idx_i),
        "idx_j": torch.from_numpy(idx_j),
        "Q": torch.tensor(Q_tot),
        "offsets": torch.from_numpy(offsets)
    })
    e_torch = output["E"].detach().numpy()
    f_torch = output["Fa"].detach().numpy()
    with tf.Session() as sess:
        R_tf = tf.Variable(R)
        sess.run(tf.global_variables_initializer())
        e_tf, f_tf = physnet_tf.energy_and_forces(Z, R_tf, idx_i, idx_j, Q_tot, offsets=offsets)
        e_tf = e_tf.eval()
        f_tf = f_tf.eval()
    assert_allclose(e_torch, e_tf, rtol=1e-7, atol=1e-7)
    assert_allclose(f_torch, f_tf, rtol=1e-7, atol=1e-7)
