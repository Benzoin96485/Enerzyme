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
from neural_network.NeuralNetwork import NeuralNetwork
# from enerzyme.models.physnet.init import semi_orthogonal_glorot_weights

F = 128
K = 5
N = 50
M = 20
cutoff=5.0
R = np.random.rand(N, 3) * 10
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
# W_init = semi_orthogonal_glorot_weights(F, F)
b_init = np.random.randn(F)
D = np.random.random(*idx_i.shape) * 30 + 3
rbf = np.random.randn(*idx_i.shape, K)
Z = np.random.randint(0, 94, N)
Qa = np.random.randn(N)
Ea = np.random.randn(N)
Q_tot = np.random.randn()


def initialize(dtype="float64", use_dispersion=True):
    global R, offsets
    if dtype == "float64":
        dtype_torch = torch.float64
        dtype_tf = tf.float64
    elif dtype == "float32":
        dtype_torch = torch.float32
        dtype_tf = tf.float32
        R = R.astype(np.float32)
        offsets = offsets.astype(np.float32)
    state = get_state()
    physnet_torch = build_model(
        architecture="PhysNet",
        build_params={
            "dim_embedding": F,
            "num_rbf": K,
            "max_Za": 95,
            "cutoff_sr": cutoff
        },
        layer_params=[
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
    )
    # physnet_torch = None
    # physnet_torch = PhysNetCore(F, K, 5.0, dtype=dtype, use_dispersion=use_dispersion)
    set_state(state)
    physnet_tf = NeuralNetwork(F, K, cutoff, scope="test", dtype=dtype_tf, use_dispersion=use_dispersion)
    # physnet_tf._embeddings = tf.Variable(physnet_torch.embeddings.weight.detach().numpy(), name="embeddings", dtype=dtype_tf)
    return physnet_torch, physnet_tf
    


def test_initialize():
    initialize()


# def test_grimme_d3_coefficient():
#     physnet_torch, physnet_tf = initialize()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         assert_allclose(physnet_torch.a1.detach().numpy(), physnet_tf.a1.eval())
#         assert_allclose(physnet_torch.a2.detach().numpy(), physnet_tf.a2.eval())
#         assert_allclose(physnet_torch.s6.detach().numpy(), physnet_tf.s6.eval())
#         assert_allclose(physnet_torch.s8.detach().numpy(), physnet_tf.s8.eval())


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
    )
    rbf1 = rbf_layer.get_rbf(torch.from_numpy(D1)).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rbf2 = physnet_tf.rbf_layer(D1).eval()
    assert_allclose(rbf1, rbf2, atol=1e-7, rtol=1e-7)


def test_embedding():
    from enerzyme.models.layers.embedding import RandomAtomEmbedding
    embedding = RandomAtomEmbedding(95, F)
    embedding.get_embedding(torch.from_numpy(Z))


# def test_DenseLayer():
#     from enerzyme.models.physnet.layer import DenseLayer as DenseLayer_torch
#     from neural_network.layers.DenseLayer import DenseLayer as Denselayer_tf
    
#     dense_layer_torch = DenseLayer_torch(F, F, W_init=torch.from_numpy(W_init), b_init=torch.from_numpy(b_init))
#     dense_layer_tf = Denselayer_tf(F, F, W_init=W_init, b_init=b_init, scope="test", dtype=tf.float64)
#     y_dense_torch = dense_layer_torch(torch.from_numpy(x)).detach().numpy()
#     l2loss_torch = dense_layer_torch.l2loss.detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y_dense_tf = dense_layer_tf(x).eval()
#         l2loss_tf = dense_layer_tf.l2loss.eval()
#     assert_allclose(y_dense_torch, y_dense_tf)
#     assert_allclose(l2loss_torch, l2loss_tf)


# def test_ResidualLayer():
#     from enerzyme.models.physnet.layer import ResidualLayer as ResidualLayer_torch
#     from enerzyme.models.physnet.init import semi_orthogonal_glorot_weights
#     from neural_network.layers.ResidualLayer import ResidualLayer as ResidualLayer_tf
#     residual_layer_torch = ResidualLayer_torch(F, F, W_init=torch.from_numpy(W_init), b_init=torch.from_numpy(b_init))
#     residual_layer_tf = ResidualLayer_tf(F, F, W_init=W_init, b_init=b_init, scope="test", dtype=tf.float64)
#     y_residual_torch = residual_layer_torch(torch.from_numpy(x.copy())).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y_residual_tf = residual_layer_tf(x.copy()).eval()
#     assert_allclose(y_residual_torch, y_residual_tf)


# def test_InteractionLayer():
#     from enerzyme.models.physnet.layer import InteractionLayer as InteractionLayer_torch
#     from neural_network.layers.InteractionLayer import InteractionLayer as InteractionLayer_tf
#     state = get_state()
#     interaction_layer_torch = InteractionLayer_torch(K, F, 3)
#     set_state(state)
#     interaction_layer_tf = InteractionLayer_tf(K, F, 3, scope="test", dtype=tf.float64)
#     y_interaction_torch = interaction_layer_torch(
#         torch.from_numpy(x.copy()), 
#         torch.from_numpy(rbf.copy()), 
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j)
#     ).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y_interaction_tf = interaction_layer_tf(x.copy(), rbf.copy(), idx_i, idx_j).eval()
#     assert_allclose(y_interaction_torch, y_interaction_tf)


# def test_InteractionBlock():
#     from enerzyme.models.physnet.block import InteractionBlock as InteractionBlock_torch
#     from neural_network.layers.InteractionBlock import InteractionBlock as InteractionBlock_tf
#     state = get_state()
#     interaction_block_torch = InteractionBlock_torch(K, F, 3, 3)
#     set_state(state)
#     interaction_block_tf = InteractionBlock_tf(K, F, 3, 3, scope="test", dtype=tf.float64)
#     y_interaction_torch = interaction_block_torch(
#         torch.from_numpy(x.copy()), 
#         torch.from_numpy(rbf.copy()), 
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j)
#     ).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y_interaction_tf = interaction_block_tf(x.copy(), rbf.copy(), idx_i, idx_j).eval()
#     assert_allclose(y_interaction_torch, y_interaction_tf)


# def test_OutputBlock():
#     from enerzyme.models.physnet.block import OutputBlock as OutputBlock_torch
#     from neural_network.layers.OutputBlock import OutputBlock as OutputBlock_tf
#     state = get_state()
#     output_block_torch = OutputBlock_torch(F, 3)
#     set_state(state)
#     output_block_tf = OutputBlock_tf(F, 3, scope="test", dtype=tf.float64)
#     y_output_torch = output_block_torch(
#         torch.from_numpy(x.copy())
#     ).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y_output_tf = output_block_tf(x.copy()).eval()
#     assert_allclose(y_output_torch, y_output_tf)


# def test_atomic_properties():
#     physnet_torch, physnet_tf = initialize("float32")
#     Ea_torch, Qa_torch, Dij_lr_torch, nhloss_torch = physnet_torch.atomic_properties(
#         torch.from_numpy(Z.copy()), 
#         torch.tensor(R, dtype=torch.float32), 
#         torch.from_numpy(idx_i.copy()), 
#         torch.from_numpy(idx_j.copy()), 
#         torch.tensor(offsets, dtype=torch.float32)
#     )
#     Ea_torch = Ea_torch.detach().numpy()
#     Qa_torch = Qa_torch.detach().numpy()
#     Dij_lr_torch = Dij_lr_torch.detach().numpy()
#     nhloss_torch = nhloss_torch.detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         Ea_tf, Qa_tf, Dij_lr_tf, nhloss_tf = physnet_tf.atomic_properties(
#             Z, R, idx_i, idx_j, offsets
#         )
#         Ea_tf = Ea_tf.eval()
#         Qa_tf = Qa_tf.eval()
#         Dij_lr_tf = Dij_lr_tf.eval()
#         nhloss_tf = nhloss_tf.eval()
#     assert_allclose(Ea_torch, Ea_tf)
#     assert_allclose(Qa_torch, Qa_tf)
#     assert_allclose(Dij_lr_torch, Dij_lr_tf, rtol=1e-6, atol=1e-6)
#     assert_allclose(nhloss_torch, nhloss_tf)


# def test_edisp():
#     from enerzyme.models.physnet.d3 import edisp as edisp_torch
#     from neural_network.grimme_d3.grimme_d3 import edisp as edisp_tf
#     e_torch = edisp_torch(torch.from_numpy(Z.copy()), torch.from_numpy(D.copy()), torch.from_numpy(idx_i), torch.from_numpy(idx_j)).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         e_tf = edisp_tf(Z, D, idx_i, idx_j).eval()
#     assert_allclose(e_torch, e_tf)


# def test_electrostatic_energy_per_atom():
#     physnet_torch, physnet_tf = initialize("float32")
#     e_torch = physnet_torch.electrostatic_energy_per_atom(
#         torch.from_numpy(D.copy()), 
#         torch.from_numpy(Qa.copy()), 
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j)
#     ).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         e_tf = physnet_tf.electrostatic_energy_per_atom(D, Qa, idx_i, idx_j).eval()
#     assert_allclose(e_torch, e_tf)


# def test_energy_from_scaled_atomic_properties():
#     physnet_torch, physnet_tf = initialize()
#     e_torch = physnet_torch.energy_from_scaled_atomic_properties(
#         torch.from_numpy(Ea.copy()), 
#         torch.from_numpy(Qa.copy()), 
#         torch.from_numpy(D.copy()),
#         torch.from_numpy(Z.copy()),
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j)
#     )
#     e_torch = e_torch.detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         e_tf = physnet_tf.energy_from_scaled_atomic_properties(Ea, Qa, D, Z, idx_i, idx_j).eval()
#     assert_allclose(e_torch, e_tf)


# def test_scaled_charges():
#     physnet_torch, physnet_tf = initialize()
#     q_torch = physnet_torch.scaled_charges(
#         torch.from_numpy(Z.copy()),
#         torch.from_numpy(Qa.copy()), 
#         torch.tensor(Q_tot)
#     ).detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         q_tf = physnet_tf.scaled_charges(Z, Qa, Q_tot).eval()
#     assert_allclose(q_torch, q_tf, rtol=1e-7, atol=1e-7)


# def test_energy_from_atomic_properties():
#     physnet_torch, physnet_tf = initialize()
#     e_torch = physnet_torch.energy_from_atomic_properties(
#         torch.from_numpy(Ea.copy()), 
#         torch.from_numpy(Qa.copy()), 
#         torch.from_numpy(D.copy()),
#         torch.from_numpy(Z.copy()),
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j),
#         torch.tensor(Q_tot)
#     )
#     e_torch = e_torch.detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         e_tf = physnet_tf.energy_from_atomic_properties(Ea, Qa, D, Z, idx_i, idx_j, Q_tot).eval()
#     assert_allclose(e_torch, e_tf)
#     pass


# def test_energy_from_atomic_properties():
#     physnet_torch, physnet_tf = initialize()
#     e_torch = physnet_torch.energy_from_atomic_properties(
#         torch.from_numpy(Ea.copy()), 
#         torch.from_numpy(Qa.copy()), 
#         torch.from_numpy(D.copy()),
#         torch.from_numpy(Z.copy()),
#         torch.from_numpy(idx_i), 
#         torch.from_numpy(idx_j),
#         torch.tensor(Q_tot)
#     )
#     e_torch = e_torch.detach().numpy()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         e_tf = physnet_tf.energy_from_atomic_properties(Ea, Qa, D, Z, idx_i, idx_j, Q_tot).eval()
#     assert_allclose(e_torch, e_tf)
#     pass


# def test_energy_and_forces():
#     torch.autograd.set_detect_anomaly(True)
#     physnet_torch, physnet_tf = initialize("float32")
#     e_torch, f_torch = physnet_torch.energy_and_forces(
#         torch.from_numpy(Z.copy()),
#         torch.tensor(R, requires_grad=True),
#         torch.from_numpy(idx_i),
#         torch.from_numpy(idx_j),
#         torch.tensor(Q_tot, dtype=torch.float32),
#         offsets=torch.from_numpy(offsets)
#     )
#     e_torch = e_torch.detach().numpy()
#     f_torch = f_torch.detach().numpy()
#     with tf.Session() as sess:
#         R_tf = tf.Variable(R)
#         sess.run(tf.global_variables_initializer())
#         e_tf, f_tf = physnet_tf.energy_and_forces(Z, R_tf, idx_i, idx_j, Q_tot, offsets=offsets)
#         e_tf = e_tf.eval()
#         f_tf = f_tf.eval()
#     assert_allclose(e_torch, e_tf, rtol=1e-7, atol=1e-7)
#     assert_allclose(f_torch, f_tf, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    test_RBFLayer()
#     import time
#     from enerzyme.models.physnet.d3 import edisp
    
#     N = 300
#     idx_i = np.empty((N, N-1), dtype=int)
#     idx_j = np.empty((N, N-1), dtype=int)
#     for i in range(N):
#         for j in range(N - 1):
#             idx_i[i, j] = i
#     for i in range(N):
#         c = 0
#         for j in range(N):
#             if j != i:
#                 idx_j[i,c] = j
#                 c += 1
#     idx_i = torch.tensor(idx_i.reshape(-1))
#     idx_j = torch.tensor(idx_j.reshape(-1))
#     D = torch.tensor(np.random.random(*idx_i.shape) * 30 + 3)
#     Z = torch.tensor(np.random.randint(0, 94, N)) 
#     start_time = time.time()
#     e = edisp(Z, D, idx_i, idx_j, c6_version=1)
#     end_time = time.time()
#     print(f"v1 c6: {end_time - start_time} s")
#     start_time = time.time()
#     edisp(Z, D, idx_i, idx_j, c6_version=2)
#     end_time = time.time()
#     print(f"v2 c6: {end_time - start_time} s")