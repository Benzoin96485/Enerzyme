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
from mlff.models.physnet import PhysNet
from neural_network.NeuralNetwork import NeuralNetwork
from mlff.models.physnet.init import semi_orthogonal_glorot_weights

F = 128
K = 60
N = 50
M = 20
R = np.random.randn(N, 3) / 10
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
W_init = semi_orthogonal_glorot_weights(F, F)
b_init = np.random.randn(F)

rbf = np.random.randn(*idx_i.shape, K)


def initialize():
    state = get_state()
    physnet_torch = PhysNet(F, K, 5.0)
    set_state(state)
    physnet_tf = NeuralNetwork(F, K, 5.0, scope="test", dtype=tf.float64)
    return physnet_torch, physnet_tf


def test_initialize():
    initialize()


def test_grimme_d3_coefficient():
    physnet_torch, physnet_tf = initialize()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert_allclose(physnet_torch.a1.detach().numpy(), physnet_tf.a1.eval())
        assert_allclose(physnet_torch.a2.detach().numpy(), physnet_tf.a2.eval())
        assert_allclose(physnet_torch.s6.detach().numpy(), physnet_tf.s6.eval())
        assert_allclose(physnet_torch.s8.detach().numpy(), physnet_tf.s8.eval())


def test_calculate_interatomic_distances():
    physnet_torch, physnet_tf = initialize()
    
    D1 = physnet_torch.calculate_interatomic_distances(
        torch.from_numpy(R), 
        torch.from_numpy(idx_i), 
        torch.from_numpy(idx_j), 
        torch.from_numpy(offsets)
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
    physnet_torch, physnet_tf = initialize()
    D1 = physnet_torch.calculate_interatomic_distances(
        torch.from_numpy(R), 
        torch.from_numpy(idx_i), 
        torch.from_numpy(idx_j), 
        torch.from_numpy(offsets)
    ).numpy()
    rbf1 = physnet_torch.rbf_layer(torch.tensor(D1)).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rbf2 = physnet_tf.rbf_layer(D1).eval()
    assert_allclose(rbf1, rbf2, atol=1e-7, rtol=1e-7)


def test_DenseLayer():
    from mlff.models.physnet.layer import DenseLayer as DenseLayer_torch
    from neural_network.layers.DenseLayer import DenseLayer as Denselayer_tf
    
    dense_layer_torch = DenseLayer_torch(F, F, W_init=torch.from_numpy(W_init), b_init=torch.from_numpy(b_init))
    dense_layer_tf = Denselayer_tf(F, F, W_init=W_init, b_init=b_init, scope="test", dtype=tf.float64)
    y_dense_torch = dense_layer_torch(torch.from_numpy(x)).numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_dense_tf = dense_layer_tf(x).eval()
    assert_allclose(y_dense_torch, y_dense_tf)


def test_ResidualLayer():
    from mlff.models.physnet.layer import ResidualLayer as ResidualLayer_torch
    from mlff.models.physnet.init import semi_orthogonal_glorot_weights
    from neural_network.layers.ResidualLayer import ResidualLayer as ResidualLayer_tf
    residual_layer_torch = ResidualLayer_torch(F, F, W_init=torch.from_numpy(W_init), b_init=torch.from_numpy(b_init))
    residual_layer_tf = ResidualLayer_tf(F, F, W_init=W_init, b_init=b_init, scope="test", dtype=tf.float64)
    y_residual_torch = residual_layer_torch(torch.from_numpy(x.copy())).numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_residual_tf = residual_layer_tf(x.copy()).eval()
    assert_allclose(y_residual_torch, y_residual_tf)


def test_InteractionLayer():
    from mlff.models.physnet.layer import InteractionLayer as InteractionLayer_torch
    from neural_network.layers.InteractionLayer import InteractionLayer as InteractionLayer_tf
    state = get_state()
    interaction_layer_torch = InteractionLayer_torch(K, F, 3)
    set_state(state)
    interaction_layer_tf = InteractionLayer_tf(K, F, 3, scope="test", dtype=tf.float64)
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
    from mlff.models.physnet.block import InteractionBlock as InteractionBlock_torch
    from neural_network.layers.InteractionBlock import InteractionBlock as InteractionBlock_tf
    state = get_state()
    interaction_block_torch = InteractionBlock_torch(K, F, 3, 3)
    set_state(state)
    interaction_block_tf = InteractionBlock_tf(K, F, 3, 3, scope="test", dtype=tf.float64)
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
    from mlff.models.physnet.block import OutputBlock as OutputBlock_torch
    from neural_network.layers.OutputBlock import OutputBlock as OutputBlock_tf
    state = get_state()
    output_block_torch = OutputBlock_torch(F, 3)
    set_state(state)
    output_block_tf = OutputBlock_tf(F, 3, scope="test", dtype=tf.float64)
    y_output_torch = output_block_torch(
        torch.from_numpy(x.copy())
    ).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_output_tf = output_block_tf(x.copy()).eval()
    assert_allclose(y_output_torch, y_output_tf)