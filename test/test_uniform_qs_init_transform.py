"""Tests for uniform Q/S per-atom init (Q_init_a, S_init_a) transform."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import h5py
import numpy as np

from enerzyme.data.datahub import DataHub, SingleDataHub, _coerce_dataset_params
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


def test_coerce_dataset_params_maps_transforms_to_preprocessings():
    out = _coerce_dataset_params({"data_path": "x.pkl", "transforms": {"uniform_qs_init": True}})
    assert "transforms" not in out
    assert out["preprocessings"] == {"uniform_qs_init": True}


def test_coerce_dataset_params_preprocessings_wins_on_conflict():
    out = _coerce_dataset_params(
        {
            "transforms": {"uniform_qs_init": True, "negative_gradient": False},
            "preprocessings": {"negative_gradient": True},
        }
    )
    assert out["preprocessings"]["uniform_qs_init"] is True
    assert out["preprocessings"]["negative_gradient"] is True


def _write_tiny_qs_pickle(path: Path) -> None:
    frames = [
        {
            "atom_type": ["C", "O"],
            "coord": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            "total_chrg": 2,
            "total_spin": 0,
        },
        {
            "atom_type": ["N", "H", "H"],
            "coord": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "total_chrg": -3,
            "total_spin": 1,
        },
    ]
    with open(path, "wb") as f:
        pickle.dump(frames, f)


def _assert_init_features(hub: SingleDataHub) -> None:
    assert "Q_init_a" in hub.feature_types
    assert "S_init_a" in hub.feature_types
    feats = hub.features
    assert "Q_init_a" in feats and "S_init_a" in feats
    qia = np.asarray(feats["Q_init_a"])
    sia = np.asarray(feats["S_init_a"])
    n = np.asarray(feats["N"]).ravel()
    q = np.asarray(feats["Q"]).ravel()
    s = np.asarray(feats["S"]).ravel()
    for i, n_atoms in enumerate(n):
        n_atoms = int(n_atoms)
        assert np.isclose(qia[i, :n_atoms].sum(), q[i % len(q)])
        assert np.isclose(sia[i, :n_atoms].sum(), s[i % len(s)])


def test_singledatahub_uniform_qs_via_preprocessings_only(tmp_path: Path):
    """Bugbot: populate hook must honor preprocessings, not only global_transforms."""
    pkl = tmp_path / "qs.pkl"
    _write_tiny_qs_pickle(pkl)
    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=str(pkl),
        data_format="pickle",
        preload=True,
        neighbor_list="",
        features={"Ra": "coord", "Za": "atom_type", "Q": "total_chrg", "S": "total_spin", "N": None},
        targets={},
        preprocessings={"uniform_qs_init": True},
        global_transforms=None,
    )
    assert hub._populate_uniform_qs_init is True
    _assert_init_features(hub)


def test_singledatahub_uniform_qs_via_global_transforms(tmp_path: Path):
    pkl = tmp_path / "qs.pkl"
    _write_tiny_qs_pickle(pkl)
    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=str(pkl),
        data_format="pickle",
        preload=True,
        neighbor_list="",
        features={"Ra": "coord", "Za": "atom_type", "Q": "total_chrg", "S": "total_spin", "N": None},
        targets={},
        preprocessings=None,
        global_transforms={"uniform_qs_init": True},
    )
    assert hub._populate_uniform_qs_init is True
    _assert_init_features(hub)


def test_datahub_multidataset_transforms_remap_enables_uniform_qs(tmp_path: Path):
    """Per-dataset YAML ``transforms:`` must become preprocessings and populate init fields."""
    pkl = tmp_path / "qs.pkl"
    _write_tiny_qs_pickle(pkl)
    hub = DataHub(
        dump_dir=str(tmp_path / "out"),
        datasets={
            "train": {
                "data_path": str(pkl),
                "data_format": "pickle",
                "preload": True,
                "neighbor_list": "",
                "features": {
                    "Ra": "coord",
                    "Za": "atom_type",
                    "Q": "total_chrg",
                    "S": "total_spin",
                    "N": None,
                },
                "targets": {},
                "transforms": {"uniform_qs_init": True},
            }
        },
    )
    train = hub.datahubs["train"]
    assert train.preprocessings == {"uniform_qs_init": True}
    assert train._populate_uniform_qs_init is True
    _assert_init_features(train)
