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
sys.path.extend(["/home/gridsan/wlluo/src/MLFF/Enerzyme/", "/home/gridsan/wlluo/src/MLFF/PhysNet/neural_network"])
from mlff.models.physnet import PhysNet
from neural_network.NeuralNetwork import NeuralNetwork


F = 128
K = 60


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
    R = np.random.randn(K, 3) / 10
    idx_i = np.random.randint(0, K, 20)
    idx_j = np.random.randint(0, K, 20)
    offsets = np.random.random((20, 3))
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


def test_RBFlayer():
    physnet_torch, physnet_tf = initialize()
    R = np.random.randn(K, 3)
    idx_i = np.random.randint(0, K, 20)
    idx_j = np.random.randint(0, K, 20)
    offsets = np.random.random((20, 3))
    D1 = physnet_torch.calculate_interatomic_distances(
        torch.from_numpy(R), 
        torch.from_numpy(idx_i), 
        torch.from_numpy(idx_j), 
        torch.from_numpy(offsets)
    ).numpy()
    D = D1
    rbf1 = physnet_torch.rbf_layer(torch.from_numpy(D)).detach().numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rbf2 = physnet_tf.rbf_layer(D).eval()
    assert_allclose(rbf1, rbf2, atol=1e-7, rtol=1e-7)


def test_DenseLayer():
    from mlff.models.physnet.layer import DenseLayer as DenseLayer_torch
    from mlff.models.physnet.init import semi_orthogonal_glorot_weights
    from neural_network.layers.DenseLayer import DenseLayer as Denselayer_tf
    W_init = semi_orthogonal_glorot_weights(F, F)
    b_init = np.random.randn(F)
    x = np.random.randn(K, F)
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
    W_init = semi_orthogonal_glorot_weights(F, F)
    b_init = np.random.randn(F)
    x = np.random.randn(K, F)
    residual_layer_torch = ResidualLayer_torch(F, F, W_init=torch.from_numpy(W_init), b_init=torch.from_numpy(b_init))
    residual_layer_tf = ResidualLayer_tf(F, F, W_init=W_init, b_init=b_init, scope="test", dtype=tf.float64)
    y_residual_torch = residual_layer_torch(torch.from_numpy(x.copy())).numpy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_residual_tf = residual_layer_tf(x.copy()).eval()
    assert_allclose(y_residual_torch, y_residual_tf)


def test_InteractionLayer():
    from mlff.models.physnet.layer import InteractionLayer as InteractionLayer_torch
    from neural_network.layers.ResidualLayer import InteractionLayer as InteractionLayer_tf
    state = get_state()
    set_state(state)
    pass


def test_InteractionBlock():
    pass