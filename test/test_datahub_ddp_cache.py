"""DDP cache-miss DataHub: HDF5 is the training source; inverse state on peers."""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from enerzyme.data.datahub import SingleDataHub
from enerzyme.data.transform import Transform
from enerzyme.tasks.distributed import LaunchEnv, run_rank0_exclusive


def _write_co_frames(pkl: Path, energies) -> None:
    frames = [
        {
            "atom_type": ["C", "O"],
            "coord": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            "total_chrg": 0,
            "energy": float(e),
            "grad": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        }
        for e in energies
    ]
    with open(pkl, "wb") as f:
        pickle.dump(frames, f)


def _hub_kwargs(pkl: Path, dump_dir: Path, global_transforms: dict) -> dict:
    return dict(
        dump_dir=str(dump_dir),
        data_path=str(pkl),
        data_format="pickle",
        preload=True,
        neighbor_list="",
        compressed=False,
        features={
            "Ra": "coord",
            "Za": "atom_type",
            "Q": "total_chrg",
            "N": None,
        },
        targets={"E": "energy", "Fa": "grad"},
        preprocessings=None,
        global_transforms=global_transforms,
    )


def _metric_snapshot(hub: SingleDataHub, idx: int = 0) -> dict:
    return {
        "E": [float(np.asarray(hub.data["E"]).ravel()[idx])],
        "Fa": [np.array(hub.data["Fa"][idx], copy=True)],
        "Za": [np.array(hub.data["Za"][idx], copy=True)],
    }


def _build_ddp_hubs(monkeypatch, kwargs: dict):
    """Construct two SingleDataHubs as torchrun ranks 0 and 1 on one dump dir.

    A barrier at ``run_rank0_exclusive`` makes both ranks finish
    ``Transform.__init__`` before rank 0 writes the HDF5 / statistics file.
    """
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]
    rank_by_thread: dict[int, int] = {}
    exclusive_gate = threading.Barrier(2)
    orig_exclusive = run_rank0_exclusive

    def fake_detect(environ=None):
        return envs[rank_by_thread[threading.get_ident()]]

    def gated_exclusive(*args, **kw):
        exclusive_gate.wait(timeout=30)
        return orig_exclusive(*args, **kw)

    monkeypatch.setattr("enerzyme.tasks.distributed.detect_launch_env", fake_detect)
    monkeypatch.setattr("enerzyme.tasks.distributed.run_rank0_exclusive", gated_exclusive)

    hubs: list = [None, None]
    errors: list = [None, None]

    def _build(rank: int) -> None:
        rank_by_thread[threading.get_ident()] = rank
        try:
            hubs[rank] = SingleDataHub(**kwargs)
        except Exception as exc:
            errors[rank] = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_build, [0, 1]))

    for rank, err in enumerate(errors):
        if err is not None:
            raise err
    return hubs[0], hubs[1]


def test_ddp_cache_miss_peer_reads_transformed_hdf5(tmp_path, monkeypatch):
    """Typical transforms: rank 0 writes processed HDF5; peers train on that file.

    ``atomic_energy`` / ``negative_gradient`` have no fitted state. Inverse on
    every DDP rank uses the same CSV / sign-flip from YAML, so val metrics
    all-reduce consistently. The reported 'unfitted scaler corrupts metrics'
    path does not apply here.
    """
    ae_csv = tmp_path / "ae.csv"
    ae_csv.write_text("atom_type,atomic_energy\nH,-0.5\nC,-10.0\nO,-20.0\n")
    pkl = tmp_path / "labeled.pkl"
    _write_co_frames(pkl, [-35.5])
    kwargs = _hub_kwargs(
        pkl,
        tmp_path / "out",
        {"atomic_energy": str(ae_csv), "negative_gradient": True},
    )
    rank0, rank1 = _build_ddp_hubs(monkeypatch, kwargs)

    # HDF5 is already transformed (E shifted, Fa sign-flipped) on both ranks.
    for hub in (rank0, rank1):
        e = float(np.asarray(hub.data["E"]).ravel()[0])
        fa0 = np.asarray(hub.data["Fa"][0][0])
        assert np.isclose(e, -5.5)
        assert np.allclose(fa0, [-0.1, 0.0, 0.0])

    y0 = _metric_snapshot(rank0)
    y1 = _metric_snapshot(rank1)
    rank0.global_transform.inverse_transform(y0)
    rank1.global_transform.inverse_transform(y1)
    assert np.isclose(y0["E"][0], -35.5)
    assert np.isclose(y1["E"][0], -35.5)
    assert np.allclose(y0["Fa"][0][0], [0.1, 0.0, 0.0])
    assert np.allclose(y1["Fa"][0][0], [0.1, 0.0, 0.0])


def test_total_energy_normalization_peer_reloads_statistics(tmp_path, monkeypatch):
    """Fitted mean/std live on disk; cache-miss peers must reload before inverse."""
    pkl = tmp_path / "labeled.pkl"
    _write_co_frames(pkl, [1.0, 3.0])
    kwargs = _hub_kwargs(
        pkl,
        tmp_path / "out",
        {"total_energy_normalization": str(tmp_path)},
    )
    rank0, rank1 = _build_ddp_hubs(monkeypatch, kwargs)

    n0 = rank0.global_transform.normalizations[0]
    n1 = rank1.global_transform.normalizations[0]
    assert n0.loaded and n1.loaded
    assert np.isclose(n0.shift, n1.shift)
    assert np.isclose(n0.scale, n1.scale)

    y0 = _metric_snapshot(rank0)
    y1 = _metric_snapshot(rank1)
    rank0.global_transform.inverse_transform(y0)
    rank1.global_transform.inverse_transform(y1)
    assert np.isclose(y0["E"][0], 1.0)
    assert np.isclose(y1["E"][0], 1.0)


def test_total_energy_normalization_inverse_loads_late_statistics(tmp_path):
    """Object built before statistics.data exists can still inverse after rank 0 writes."""
    stats_dir = tmp_path
    peer = Transform(
        {"total_energy_normalization": str(stats_dir)},
        str(stats_dir),
    )
    assert peer.normalizations[0].loaded is False

    joblib_scale = 2.0
    joblib_shift = 10.0
    import joblib

    joblib.dump(
        {"shift": joblib_shift, "scale": joblib_scale},
        stats_dir / "statistics.data",
    )
    y = {"E": [1.0], "Fa": [np.array([[1.0, 0.0, 0.0]], dtype=np.float64)]}
    peer.inverse_transform(y)
    assert np.isclose(y["E"][0], 1.0 * joblib_scale + joblib_shift)
    assert np.allclose(y["Fa"][0], [[2.0, 0.0, 0.0]])


def test_reload_fitted_state_is_noop_without_normalization():
    tf = Transform({"negative_gradient": True})
    tf.reload_fitted_state()
    assert tf.normalizations == []
