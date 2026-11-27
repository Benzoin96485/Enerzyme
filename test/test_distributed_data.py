"""Cache publication, partition order, and rank-owned outputs during DDP startup."""
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from addict import Dict

from enerzyme.data.datahub import SingleDataHub
from enerzyme.models.launcher import FFDataset
from enerzyme.tasks import distributed
from enerzyme.tasks.splitter import Splitter
from enerzyme.train import validate_parallel_config


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "trajectory.pkl"
    with source.open("wb") as stream:
        pickle.dump({
            "Ra": np.zeros((6, 2, 3)), "Za": np.ones((6, 2), dtype=int),
            "E": [1., 2., 4., 6., 7., 10.], "Fa": np.ones((6, 2, 3)),
        }, stream)
    return dict(dump_dir=str(tmp_path), data_path=str(source), features={"Ra": "Ra", "Za": "Za", "N": "N"},
                targets={"E": "E", "Fa": "Fa"}, global_transforms={"total_energy_normalization": str(tmp_path)})


@pytest.fixture
def ranks(monkeypatch):
    local = threading.local()

    def launch():
        rank = getattr(local, "rank", 0)
        return distributed.LaunchEnv("torchrun", rank, rank, 2, 2)

    monkeypatch.setattr(distributed, "detect_launch_env", launch)
    def run(fn):
        def invoke(rank):
            local.rank = rank
            return fn(rank)

        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(invoke, (0, 1)))

    return run


def test_shared_cold_cache_retry_and_warm_state(dataset, ranks, monkeypatch):
    calls = []
    build = SingleDataHub._init_data
    def count(hub):
        calls.append(distributed.global_rank())
        return build(hub)

    monkeypatch.setattr(SingleDataHub, "_init_data", count)
    hubs = ranks(lambda rank: SingleDataHub(**dataset))
    assert calls == [0]
    h5file = Path(hubs[0].file.filename)
    original = h5file.read_bytes()
    cached = hubs[0].data["E"][:]
    for hub in hubs:
        assert np.array_equal(hub.data["E"][:], cached)
        assert hub.global_transform.normalizations[0].shift == 5
        hub.file.close()

    # A failed refresh must reach both participants and leave the previous cache readable.
    with monkeypatch.context() as patch:
        def fail(hub):
            raise ValueError("interrupted refresh")

        patch.setattr(SingleDataHub, "_init_neighbor_list", fail)
        def refresh(rank):
            with pytest.raises((ValueError, RuntimeError), match="interrupted refresh"):
                SingleDataHub(**dataset, preload=False)

        ranks(refresh)
    assert h5file.read_bytes() == original
    for hub in ranks(lambda rank: SingleDataHub(**dataset, preload=False)):
        hub.file.close()
    assert calls == [0, 0, 0]

    # Warm readers recover the inverse convention belonging to these exact HDF5 values.
    joblib.dump({"shift": 99., "scale": 4.}, h5file.parent.parent / "statistics.data")
    for hub in ranks(lambda rank: SingleDataHub(**dataset)):
        assert hub.global_transform.normalizations[0].shift == 5
        assert np.array_equal(hub.data["E"][:], cached)
        hub.file.close()
    assert calls == [0, 0, 0]


def test_local_lightning_child_loads_published_split_and_cache(dataset, monkeypatch):
    parent = SingleDataHub(**dataset)
    def split(hub, seed):
        splitter = Splitter(parts=["training", "test"], ratios=["50%"], preload=False, seed=seed)
        return splitter.get_split({"default": hub.features}, {"default": hub.preload_path})

    expected = split(parent, 11)
    parent.file.close()
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert distributed.is_lightning_subprocess()
    child = SingleDataHub(**dataset, preload=False)
    try:
        assert split(child, 22) == expected
    finally:
        child.file.close()


@pytest.mark.parametrize("in_memory", [False, True])
def test_worker_reads_original_partition_order(dataset, in_memory):
    hub = SingleDataHub(**dataset)
    features = {"empty": hub.features, "source": hub.features}
    targets = {"empty": hub.targets, "source": hub.targets}
    samples = FFDataset(features, targets, {"empty": [], "source": [4, 1]}, data_in_memory=in_memory)
    worker = pickle.loads(pickle.dumps(samples))
    expected = hub.data["E"][:][[4, 1, 3]]
    hub.file.close()
    worker.expand_with_indices({"source": [3]})
    assert [worker[i][1]["E"] for i in range(len(worker))] == list(expected)
    assert worker._indices_to_key_indices([1, 2])["source"].tolist() == [1, 2]
    assert len(FFDataset(features, targets, {"source": []}, data_in_memory=in_memory)) == 0


@pytest.mark.parametrize("settings", [{"committee_size": 2}, {"active_learning_params": {"active": True}}])
def test_distributed_mode_rejected_before_setup(settings, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = Dict(Trainer={}, Modelhub={"external_FFs": {"model": {"active": True, "Trainer": settings}}})
    with pytest.raises(ValueError, match="single process"):
        validate_parallel_config(config, training=True)


def test_only_rank_zero_materializes_predictions(tmp_path, ranks):
    from enerzyme.predict import FFPredict
    import pandas as pd

    calls = []
    def predict(rank):
        def evaluate():
            calls.append(rank)
            return {"y_pred": {"data_key": ["dataset"], "E": [4.]} if rank == 0 else {},
                    "y_truth": {}, "metric_score": pd.DataFrame({"E_rmse": [0.]}, index=["model"])}

        ff = SimpleNamespace(evaluate=evaluate, trainer=SimpleNamespace(non_target_features=[]))
        runner = FFPredict.__new__(FFPredict)
        runner.output_dir = str(tmp_path)
        runner.modelhub = SimpleNamespace(models={"FF": {"first": ff, "second": ff}})
        runner.datahub = SimpleNamespace(datahubs={"dataset": SimpleNamespace(preload_path=tmp_path)},
                                         targets={})
        runner.new_data = True
        runner.simple_predict = False
        runner.predict()
        runner._simple_predict(non_target_features=[])

    ranks(predict)
    assert sorted(calls) == [0] * 4 + [1] * 4
    assert (tmp_path / "metric.csv").exists()
    for model in ("first", "second"):
        assert joblib.load(tmp_path / f"{model}-prediction.pkl")["E"].tolist() == [4.]


def test_split_legacy_preload_and_partial_refresh(tmp_path):
    features = {key: {"Ra": np.arange(10)} for key in ("a", "b")}
    paths = {key: str(tmp_path / key) for key in features}
    parts = [
        {"name": "training", "sources": [{"dataset": key, "ratio": 0.5} for key in features]},
        {"name": "test", "sources": [{"dataset": key} for key in features]},
    ]
    usable = Splitter(parts=parts)
    for path in paths.values():
        target = Path(path) / ("split_" + usable.splitter.hash)
        target.mkdir(parents=True)
        np.savez(target / "split.npz", training=[7, 9], test=[4])
    legacy = Splitter(parts=parts, save=False).get_split(features, paths)
    assert legacy["training"] == {"a": [7, 9], "b": [7, 9]}
    new = Splitter(parts=parts, preload=False, seed=12).get_split(features, paths)
    # One archive came from an earlier publication.
    b = Path(paths["b"]) / ("split_" + usable.splitter.hash) / "split.npz"
    np.savez(b, training=[7, 9], test=[4])
    recovered = Splitter(parts=parts, seed=12).get_split(features, paths)
    assert recovered == new
    # The archive stays consumable by ordinary readers of the previous format.
    with np.load(b) as archive:
        assert set(archive.files) == {"training", "test"}


def test_single_process_cpu_lightning_committee_on_multigpu_host(monkeypatch):
    import torch

    monkeypatch.setattr("enerzyme.train.detect_launch_env", lambda: distributed.LaunchEnv("single"))
    monkeypatch.setattr("enerzyme.train.is_lightning_subprocess", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    config = Dict(Trainer={"lightning": True, "committee_size": 2, "cuda": False},
                  Modelhub={"internal_FFs": {"model": {"active": True}}})
    validate_parallel_config(config, training=True)
    del config.Trainer["cuda"]
    with pytest.raises(ValueError, match="single process"):
        validate_parallel_config(config, training=True)
