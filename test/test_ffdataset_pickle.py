"""DataLoader forkserver must pickle FFDataset without live h5py handles."""

from __future__ import annotations

import pickle

import h5py
import numpy as np

from enerzyme.data.datahub import FieldDataset
from enerzyme.models.ff import FFDataset


def _hdf5_fields(tmp_path):
    h5path = tmp_path / "tiny.hdf5"
    with h5py.File(h5path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("Ra", data=np.arange(12, dtype=np.float64).reshape(4, 1, 3))
        group.create_dataset("Za", data=np.array([[6]]))
        group.create_dataset("E", data=np.arange(4, dtype=np.float64))
        group.create_dataset("Fa", data=np.ones((4, 1, 3)))
    handle = h5py.File(h5path, "r")
    features = {"ds": FieldDataset({"Ra": handle["data"]["Ra"], "Za": handle["data"]["Za"]})}
    targets = {"ds": FieldDataset({"E": handle["data"]["E"], "Fa": handle["data"]["Fa"]})}
    return handle, features, targets


def test_ffdataset_pickle_in_memory_drops_h5py(tmp_path):
    handle, features, targets = _hdf5_fields(tmp_path)
    dataset = FFDataset(features, targets, {"ds": [0, 1, 2, 3]}, data_in_memory=True)
    restored = pickle.loads(pickle.dumps(dataset))
    feat, tgt, key = restored[0]
    assert key == "ds"
    assert np.asarray(feat["Ra"]).shape[-1] == 3
    assert restored.full_features == {}
    handle.close()


def test_ffdataset_pickle_hdf5_backed_reopens(tmp_path):
    handle, features, targets = _hdf5_fields(tmp_path)
    dataset = FFDataset(features, targets, {"ds": [0, 2]}, data_in_memory=False)
    restored = pickle.loads(pickle.dumps(dataset))
    feat, tgt, key = restored[0]
    orig_feat, orig_tgt, orig_key = dataset[0]
    assert key == orig_key == "ds"
    assert int(np.asarray(tgt["E"]).ravel()[0]) == int(np.asarray(orig_tgt["E"]).ravel()[0])
    handle.close()


def test_collate_and_metrics_are_picklable():
    import torch
    from enerzyme.tasks.batch import CollateBatch
    from enerzyme.tasks.metrics import Metrics, build_single_metric

    collate = CollateBatch(
        pyg=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
        otf_graph=True,
        generator_config=None,
        generator_training=True,
    )
    pickle.dumps(collate)
    pickle.dumps(build_single_metric("rmse"))
    pickle.dumps(Metrics({"E": {"rmse": 1.0}}))
