"""Tests for xtb_population-backed Q/S prior and atomic delta transforms (no tblite required)."""

import os
import tempfile

import h5py
import numpy as np
import pytest

from enerzyme.data.transform import (
    QSDeltaTransform,
    Transform,
    UniformSplitQSTransform,
    XTBQSPriorTransform,
)


def test_qs_delta_zero_sum_with_uniform_prior():
    """After uniform init, transformed Qa / Sa residual labels sum to ~0 per frame."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([2, 3], dtype=np.int32))
            g.create_dataset("Za", data=np.array([[6, 6, 0], [7, 7, 7]], dtype=np.int32))
            g.create_dataset("Ra", data=np.zeros((2, 3, 3), dtype=np.float64))
            g.create_dataset("Q", data=np.array([0.0, -1.0], dtype=np.float64))
            g.create_dataset("S", data=np.array([0.0, 1.0], dtype=np.float64))
            # Per-frame Qa/Sa must match Q/S so delta w.r.t. uniform prior is zero-sum
            g.create_dataset(
                "Qa",
                data=np.array([[0.1, -0.1, 0.0], [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]], dtype=np.float64),
            )
            g.create_dataset(
                "Sa",
                data=np.array([[0.05, -0.05, 0.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
            )

        with h5py.File(path, "a") as f:
            g = f["data"]
            UniformSplitQSTransform().transform(g)
            QSDeltaTransform().transform(g)
            dq = np.asarray(g["Qa"][:])
            ds = np.asarray(g["Sa"][:])
            for i, n in enumerate([2, 3]):
                assert np.isclose(dq[i, :n].sum(), 0.0, atol=1e-9)
                assert np.isclose(ds[i, :n].sum(), 0.0, atol=1e-9)
            assert "Q_delta_a" not in g and "S_delta_a" not in g


def test_xtb_prior_transform_uses_mock_fn():
    """XTBQSPriorTransform with injected callable (no tblite)."""
    calls = []

    def fake_atomic(atoms, max_scf_iter=1, **kw):
        calls.append((len(atoms), int(max_scf_iter)))
        n = len(atoms)
        return np.arange(n, dtype=np.float64) * 0.01, np.zeros(n, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([2], dtype=np.int32))
            g.create_dataset("Za", data=np.array([[1, 1, 0]], dtype=np.int32))
            g.create_dataset("Ra", data=np.zeros((1, 3, 3), dtype=np.float64))
            g.create_dataset("Q", data=np.array([0.0], dtype=np.float64))
            g.create_dataset("S", data=np.array([0.0], dtype=np.float64))

        with h5py.File(path, "a") as f:
            g = f["data"]
            XTBQSPriorTransform(atomic_qs_fn=fake_atomic, max_scf_iter=3).transform(g)
            assert len(calls) == 1 and calls[0] == (2, 3)
            qia = np.asarray(g["Q_init_a"][:])
            assert qia.shape == (1, 3)
            assert np.isclose(qia[0, :2].sum(), 0.0)


def test_transform_yaml_xtb_and_delta_hooks():
    tr = Transform(
        {
            "xtb_qs_prior": {
                "enabled": True,
                "atomic_qs_fn": lambda *a, **k: (np.zeros(1), np.zeros(1)),
            },
            "qs_delta": True,
        }
    )
    assert len(tr.xtb_qs_priors) == 1 and len(tr.qs_deltas) == 1


def test_transform_rejects_uniform_and_xtb_together():
    with pytest.raises(ValueError, match="only one of uniform_qs_init"):
        Transform(
            {
                "uniform_qs_init": True,
                "xtb_qs_prior": {"enabled": True, "atomic_qs_fn": lambda *a, **k: (np.zeros(1), np.zeros(1))},
            }
        )


def test_datahub_rejects_uniform_in_preprocessings_and_xtb_in_global():
    from enerzyme.data.datahub import SingleDataHub

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="do not combine"):
            SingleDataHub(
                dump_dir=tmp,
                data_path=os.path.join(tmp, "nope.aselmdb"),
                features={"Ra": "Ra", "Za": "Za", "N": "N"},
                targets={},
                preprocessings={"uniform_qs_init": True},
                global_transforms={
                    "xtb_qs_prior": {
                        "enabled": True,
                        "atomic_qs_fn": lambda *a, **k: (np.zeros(1), np.zeros(1)),
                    }
                },
                preload=False,
            )


def test_qs_delta_inverse_recovers_qa_sa():
    """inverse_transform adds Q_init_a / S_init_a back and writes Qa / Sa."""
    y = {
        "Qa": [np.array([0.1, -0.1]), np.array([0.0, 0.0, 0.0])],
        "Sa": [np.array([0.05, -0.05]), np.array([0.0, 0.0, 0.0])],
        "Q_init_a": [np.array([0.0, 0.0]), np.array([-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])],
        "S_init_a": [np.array([0.0, 0.0]), np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])],
    }
    tr = Transform({"qs_delta": True})
    tr.inverse_transform(y)
    assert np.allclose(y["Qa"][0], [0.1, -0.1])
    assert np.allclose(y["Sa"][0], [0.05, -0.05])
    assert np.allclose(y["Qa"][1], [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    assert np.allclose(y["Sa"][1], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_transform_forward_order_uniform_before_delta_regardless_of_yaml_key_order():
    """YAML may list qs_delta before uniform_qs_init; forward still applies prior then delta."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([2], dtype=np.int32))
            g.create_dataset("Za", data=np.array([[6, 6, 0]], dtype=np.int32))
            g.create_dataset("Ra", data=np.zeros((1, 3, 3), dtype=np.float64))
            g.create_dataset("Q", data=np.array([0.0], dtype=np.float64))
            g.create_dataset("S", data=np.array([0.0], dtype=np.float64))
            g.create_dataset("Qa", data=np.array([[0.1, -0.1, 0.0]], dtype=np.float64))
            g.create_dataset("Sa", data=np.array([[0.05, -0.05, 0.0]], dtype=np.float64))

        with h5py.File(path, "a") as f:
            g = f["data"]
            Transform({"qs_delta": True, "uniform_qs_init": True}).transform(g)
            assert "Q_init_a" in g and "Qa" in g
            assert "Q_delta_a" not in g and "S_delta_a" not in g
            assert np.isclose(np.asarray(g["Qa"][0, :2]).sum(), 0.0, atol=1e-9)


def test_xtb_prior_uses_ra_frame_count_when_n_compressed():
    """Compressed N (len=1) must still emit one Q_init_a row per Ra frame."""
    calls = []

    def fake_atomic(atoms, max_scf_iter=1, **kw):
        calls.append(len(atoms))
        n = len(atoms)
        return np.full(n, 0.1 / n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([3], dtype=np.int32))
            g.create_dataset("Ra", data=np.zeros((4, 3, 3), dtype=np.float64))
            g.create_dataset("Za", data=np.array([[6, 6, 6]], dtype=np.int32))
            g.create_dataset("Q", data=np.array([0.1], dtype=np.float64))
            g.create_dataset("S", data=np.array([0.0], dtype=np.float64))

        with h5py.File(path, "a") as f:
            g = f["data"]
            XTBQSPriorTransform(atomic_qs_fn=fake_atomic).transform(g)
            qia = np.asarray(g["Q_init_a"][:])
            assert qia.shape == (4, 3)
            assert len(calls) == 4
            assert np.allclose(qia.sum(axis=1), 0.1)


def test_datahub_populate_xtb_from_preprocessings(tmp_path):
    """Populate flags must honor preprocessings, not only global_transforms."""
    import pickle
    from pathlib import Path

    from enerzyme.data.datahub import SingleDataHub
    from enerzyme.data.transform import wants_xtb_qs_prior

    assert wants_xtb_qs_prior({"xtb_qs_prior": {"enabled": True}})

    frames = [
        {
            "atom_type": ["C", "O"],
            "coord": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            "total_chrg": 0,
            "total_spin": 0,
        }
    ]
    pkl = Path(tmp_path) / "qs.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(frames, f)

    def fake_atomic(atoms, max_scf_iter=1, **kw):
        n = len(atoms)
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    hub = SingleDataHub(
        dump_dir=str(Path(tmp_path) / "out"),
        data_path=str(pkl),
        data_format="pickle",
        preload=True,
        neighbor_list="",
        features={"Ra": "coord", "Za": "atom_type", "Q": "total_chrg", "S": "total_spin", "N": None},
        targets={},
        preprocessings={
            "xtb_qs_prior": {
                "enabled": True,
                "atomic_qs_fn": fake_atomic,
            }
        },
        global_transforms=None,
    )
    assert hub._populate_xtb_qs_prior
    assert "Q_init_a" in hub.feature_types and "S_init_a" in hub.feature_types
    assert "Q_init_a" in hub.features


def test_check_xtbml_dependencies_import_error(monkeypatch):
    import builtins
    import enerzyme.qm.xtb_population.deps as deps

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ase" or name.startswith("ase."):
            raise ImportError("no ase")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="requires ASE"):
        deps.check_xtbml_dependencies()
