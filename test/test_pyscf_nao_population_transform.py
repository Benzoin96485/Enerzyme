"""Tests for PySCF NAO Q/S prior and qs_delta (no GPU/PySCF required)."""

import os
import tempfile

import h5py
import numpy as np
import pytest

from enerzyme.data.transform import (
    PySCFNAOQSPriorTransform,
    QSDeltaTransform,
    Transform,
    UniformSplitQSTransform,
)


def test_pyscf_prior_requires_xc_basis_without_mock():
    with pytest.raises(ValueError, match="xc.*basis"):
        PySCFNAOQSPriorTransform()


def test_pyscf_prior_transform_uses_mock_fn():
    calls = []

    def fake_atomic(atoms, max_scf_iter, **_kw):
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
            PySCFNAOQSPriorTransform(
                atomic_qs_fn=fake_atomic,
                xc="b3lyp",
                basis="sto-3g",
                max_scf_iter=2,
            ).transform(g)
            assert len(calls) == 1 and calls[0][0] == 2 and calls[0][1] == 2
            qia = np.asarray(g["Q_init_a"][:])
            assert qia.shape == (1, 3)
            assert np.isclose(qia[0, :2].sum(), 0.0)


def test_transform_yaml_pyscf_and_delta_hooks():
    tr = Transform(
        {
            "pyscf_nao_qs_prior": {
                "enabled": True,
                "xc": "wb97m-v",
                "basis": "def2-svp",
                "atomic_qs_fn": lambda *a, **k: (np.zeros(1), np.zeros(1)),
            },
            "qs_delta": True,
        }
    )
    assert len(tr.pyscf_nao_qs_priors) == 1 and len(tr.qs_deltas) == 1


def test_transform_rejects_uniform_and_pyscf_together():
    with pytest.raises(ValueError, match="only one of uniform_qs_init"):
        Transform(
            {
                "uniform_qs_init": True,
                "pyscf_nao_qs_prior": {
                    "enabled": True,
                    "xc": "b3lyp",
                    "basis": "sto-3g",
                    "atomic_qs_fn": lambda *a, **k: (np.zeros(1), np.zeros(1)),
                },
            }
        )


def test_qs_delta_zero_sum_with_pyscf_prior_mock():
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

        def uniform_prior(atoms, max_scf_iter, **_kw):
            n = len(atoms)
            inv = 1.0 / n
            return np.full(n, 0.0), np.zeros(n)

        with h5py.File(path, "a") as f:
            g = f["data"]
            PySCFNAOQSPriorTransform(
                atomic_qs_fn=uniform_prior,
                xc="b3lyp",
                basis="sto-3g",
            ).transform(g)
            QSDeltaTransform().transform(g)
            dq = np.asarray(g["Qa"][0, :2])
            ds = np.asarray(g["Sa"][0, :2])
            assert np.isclose(dq.sum(), 0.0, atol=1e-9)
            assert np.isclose(ds.sum(), 0.0, atol=1e-9)
            assert "Q_delta_a" not in g


def test_transform_forward_order_pyscf_before_delta():
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
            Transform(
                {
                    "qs_delta": True,
                    "pyscf_nao_qs_prior": {
                        "enabled": True,
                        "xc": "b3lyp",
                        "basis": "sto-3g",
                        "atomic_qs_fn": lambda atoms, max_scf_iter, **kw: (
                            np.zeros(len(atoms)),
                            np.zeros(len(atoms)),
                        ),
                    },
                }
            ).transform(g)
            assert "Q_init_a" in g and np.isclose(np.asarray(g["Qa"][0, :2]).sum(), 0.0, atol=1e-9)


def test_pyscf_prior_uses_ra_frame_count_when_n_compressed():
    calls = []

    def fake_atomic(atoms, max_scf_iter, **_kw):
        calls.append((len(atoms), int(max_scf_iter)))
        n = len(atoms)
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("N", data=np.array([3], dtype=np.int32))
            g.create_dataset("Ra", data=np.zeros((4, 3, 3), dtype=np.float64))
            g.create_dataset("Za", data=np.array([[6, 6, 6]], dtype=np.int32))
            g.create_dataset("Q", data=np.array([0.0], dtype=np.float64))
            g.create_dataset("S", data=np.array([0.0], dtype=np.float64))

        with h5py.File(path, "a") as f:
            g = f["data"]
            PySCFNAOQSPriorTransform(
                atomic_qs_fn=fake_atomic,
                xc="b3lyp",
                basis="sto-3g",
            ).transform(g)
            qia = np.asarray(g["Q_init_a"][:])
            assert qia.shape == (4, 3)
            assert len(calls) == 4


def test_datahub_populate_pyscf_from_preprocessings(tmp_path):
    import pickle
    from pathlib import Path

    from enerzyme.data.datahub import SingleDataHub

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
            "pyscf_nao_qs_prior": {
                "enabled": True,
                "xc": "b3lyp",
                "basis": "sto-3g",
                "atomic_qs_fn": fake_atomic,
            }
        },
        global_transforms=None,
    )
    assert hub._populate_pyscf_nao_qs_prior
    assert "Q_init_a" in hub.feature_types
    assert "Q_init_a" in hub.features


def test_check_pyscf_nao_dependencies_import_error(monkeypatch):
    import builtins
    import enerzyme.qm.pyscf_nao_population.deps as deps

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise ImportError("no pyscf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="requires PySCF"):
        deps.check_pyscf_nao_dependencies(use_gpu=False)
