"""Tests for uniform Q/S per-atom init (Q_init_a, S_init_a) transform."""

import os
import tempfile

import h5py
import numpy as np

from enerzyme.data.transform import Transform, UniformSplitQSTransform, wants_uniform_qs_init


def test_wants_uniform_qs_init():
    assert wants_uniform_qs_init(None) is False
    assert wants_uniform_qs_init({}) is False
    assert wants_uniform_qs_init({"uniform_qs_init": False}) is False
    assert wants_uniform_qs_init({"uniform_qs_init": None}) is False
    assert wants_uniform_qs_init({"uniform_qs_init": True}) is True
    assert wants_uniform_qs_init({"uniform_qs_init": {}}) is True
    assert wants_uniform_qs_init({"uniform_qs_init": {"enabled": False}}) is False
    assert wants_uniform_qs_init({"uniform_qs_init": {"q_key": "Q"}}) is True


def test_uniform_split_conserves_totals():
    """Sum of Q_init_a over real atoms equals Q; same for S_init_a and S."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([2, 3], dtype=np.int32))
            g.create_dataset("Za", data=np.array([[6, 6, 0], [7, 7, 7]], dtype=np.int32))
            g.create_dataset("Q", data=np.array([4.0, -3.0], dtype=np.float64))
            g.create_dataset("S", data=np.array([1.0, 2.0], dtype=np.float64))

        with h5py.File(path, "a") as f:
            g = f["data"]
            UniformSplitQSTransform().transform(g)

            qia = np.asarray(g["Q_init_a"][:])
            sia = np.asarray(g["S_init_a"][:])
            for i, n in enumerate([2, 3]):
                assert np.isclose(qia[i, :n].sum(), g["Q"][i])
                assert np.isclose(sia[i, :n].sum(), g["S"][i])
            assert (qia[0, 2:] == 0).all() and (sia[0, 2:] == 0).all()


def test_transform_class_yaml_hook():
    tr = Transform({"uniform_qs_init": True})
    assert len(tr.uniform_qs_inits) == 1
    tr_disabled = Transform({"uniform_qs_init": False})
    assert len(tr_disabled.uniform_qs_inits) == 0
