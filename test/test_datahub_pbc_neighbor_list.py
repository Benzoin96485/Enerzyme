"""Datahub + batch-collation integration tests for the PBC ``cutoff`` neighbor list.

Covers the new ``Datahub.neighbor_list: cutoff`` path end to end: SingleDataHub
caches ``idx_i``/``idx_j``/``offsets``/``N_pair`` into its HDF5 store from raw
``Ra``/``Za``/``N``/``cell``/``pbc`` fields, and ``_decorate_batch_input`` collates
them into a batch-global, periodic-boundary-correct graph (offsets are Cartesian
shift vectors, never index-shifted by the running atom count, unlike idx_i/idx_j).

Also regression-checks that the existing ``neighbor_list: full`` path (used by
every pre-existing model config) is completely unaffected by these additions.
"""
import pickle

import numpy as np
import pytest

from enerzyme.data.datahub import SingleDataHub
from enerzyme.tasks.batch import _decorate_batch_input, _decorate_pyg_batch_input
import torch


def _write_pickle(path, records):
    keys = set()
    for r in records:
        keys.update(r.keys())
    data = {k: [r.get(k) for r in records] for k in keys}
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _simple_cubic_frame(n_atoms, a, seed):
    rng = np.random.default_rng(seed)
    Ra = rng.uniform(0.2, a - 0.2, size=(n_atoms, 3))
    Za = np.array([1] * n_atoms, dtype=int)
    return Ra, Za


def test_cutoff_neighbor_list_datahub_caching(tmp_path):
    a = 4.0
    records = []
    for seed in range(3):
        Ra, Za = _simple_cubic_frame(5, a, seed)
        records.append({
            "Ra": Ra, "Za": Za, "N": len(Za),
            "cell": np.eye(3) * a, "pbc": np.array([1, 1, 1]),
            "Q": 0, "E": 0.0,
        })
    data_path = tmp_path / "toy.pkl"
    _write_pickle(data_path, records)

    hub = SingleDataHub(
        dump_dir=str(tmp_path),
        data_format="pickle",
        data_path=str(data_path),
        features={"Ra": None, "Za": None, "N": None, "cell": None, "pbc": None, "Q": None},
        targets={"E": None},
        neighbor_list="cutoff",
        neighbor_list_cutoff=2.0,
        preload=False,
    )
    assert "idx_i" in hub.data and "idx_j" in hub.data and "offsets" in hub.data and "N_pair" in hub.data
    assert hub.data["offsets"].shape[0] == 3
    assert hub.data["offsets"].shape[-1] == 3

    for i in range(3):
        n_pair = int(hub.data["N_pair"][i])
        idx_i = np.asarray(hub.data["idx_i"][i][:n_pair])
        idx_j = np.asarray(hub.data["idx_j"][i][:n_pair])
        offsets = np.asarray(hub.data["offsets"][i][:n_pair])
        Ra_i = np.asarray(hub.data["Ra"][i][:5])
        d = np.linalg.norm(Ra_i[idx_j] + offsets - Ra_i[idx_i], axis=-1)
        assert np.all(d <= 2.0 + 1e-8)
        assert np.all(d > 1e-8)
        # padding beyond N_pair must be the -1 sentinel, matching the "full" convention
        assert np.all(np.asarray(hub.data["idx_i"][i][n_pair:]) == -1)


def test_cutoff_neighbor_list_batch_collation(tmp_path):
    a = 4.0
    records = []
    for seed in range(2):
        Ra, Za = _simple_cubic_frame(4, a, seed + 10)
        records.append({
            "Ra": Ra, "Za": Za, "N": len(Za),
            "cell": np.eye(3) * a, "pbc": np.array([1, 1, 1]),
            "Q": 0, "E": 0.0,
        })
    data_path = tmp_path / "toy2.pkl"
    _write_pickle(data_path, records)

    hub = SingleDataHub(
        dump_dir=str(tmp_path),
        data_format="pickle",
        data_path=str(data_path),
        features={"Ra": None, "Za": None, "N": None, "cell": None, "pbc": None, "Q": None},
        targets={"E": None},
        neighbor_list="cutoff",
        neighbor_list_cutoff=2.0,
        preload=False,
    )
    features = hub.features
    batch = [(features.loc(0), None, "a"), (features.loc(1), None, "b")]
    batch_features, _ = _decorate_batch_input(batch, dtype=torch.float64, device="cpu", otf_graph=False)

    assert "idx_i" in batch_features and "offsets" in batch_features
    n0 = int(features.loc(0)["N"])
    n1 = int(features.loc(1)["N"])
    idx_i = batch_features["idx_i"].numpy()
    idx_j = batch_features["idx_j"].numpy()
    offsets = batch_features["offsets"].numpy()
    assert idx_i.max() < n0 + n1
    # second structure's pairs must be offset by n0 (atom-count shift), offsets untouched.
    assert offsets.shape == (len(idx_i), 3)
    Ra_all = batch_features["Ra"].detach().numpy()
    d = np.linalg.norm(Ra_all[idx_j] + offsets - Ra_all[idx_i], axis=-1)
    assert np.all(d <= 2.0 + 1e-8)
    assert np.all(d > 1e-8)
    assert batch_features["cell"].shape == (2, 3, 3)
    assert batch_features["pbc"].shape == (2, 3)


def test_cutoff_neighbor_list_pyg_batch_collation(tmp_path):
    """Same as test_cutoff_neighbor_list_batch_collation but for the PyG batch path."""
    a = 4.0
    records = []
    for seed in range(2):
        Ra, Za = _simple_cubic_frame(4, a, seed + 20)
        records.append({
            "Ra": Ra, "Za": Za, "N": len(Za),
            "cell": np.eye(3) * a, "pbc": np.array([1, 1, 1]),
            "Q": 0, "E": 0.0,
        })
    data_path = tmp_path / "toy_pyg.pkl"
    _write_pickle(data_path, records)

    hub = SingleDataHub(
        dump_dir=str(tmp_path),
        data_format="pickle",
        data_path=str(data_path),
        features={"Ra": None, "Za": None, "N": None, "cell": None, "pbc": None, "Q": None},
        targets={"E": None},
        neighbor_list="cutoff",
        neighbor_list_cutoff=2.0,
        preload=False,
    )
    features = hub.features
    batch = [(features.loc(0), {}, "a"), (features.loc(1), {}, "b")]
    batch_features, _ = _decorate_pyg_batch_input(
        batch, dtype=torch.float64, device="cpu", otf_graph=False
    )

    assert batch_features.cell.shape == (2, 3, 3)
    assert batch_features.pbc.shape == (2, 3)
    idx_i, idx_j = batch_features.edge_index
    Ra_all = batch_features.Ra.detach()
    d = torch.norm(Ra_all[idx_j] + batch_features.offsets - Ra_all[idx_i], dim=-1)
    assert torch.all(d <= 2.0 + 1e-6)
    assert torch.all(d > 1e-8)


def test_full_neighbor_list_unaffected(tmp_path):
    """Regression: neighbor_list='full' (every pre-existing config) is untouched."""
    records = []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        Ra = rng.uniform(-1, 1, size=(4, 3))
        Za = np.array([1, 1, 1, 1])
        records.append({"Ra": Ra, "Za": Za, "N": 4, "Q": 0, "E": 0.0})
    data_path = tmp_path / "toy3.pkl"
    _write_pickle(data_path, records)

    hub = SingleDataHub(
        dump_dir=str(tmp_path),
        data_format="pickle",
        data_path=str(data_path),
        features={"Ra": None, "Za": None, "N": None, "Q": None},
        targets={"E": None},
        neighbor_list="full",
        preload=False,
    )
    assert "cell" not in hub.data and "pbc" not in hub.data and "offsets" not in hub.data
    # N is identical (4) across both frames, so the pre-existing "full" path
    # compresses N_pair/idx_i/idx_j into a single shared row (unrelated to this
    # change; asserting it still behaves this way is the regression check).
    n_pair_data = hub.data["N_pair"]
    for i in range(2):
        n_pair = int(n_pair_data[0 if len(n_pair_data) == 1 else i])
        assert n_pair == 4 * 3  # full all-pairs, N=4


def test_otf_fallback_rejects_periodic_sample_without_cached_edges(tmp_path):
    feature = {
        "N": 3, "Za": np.array([1, 1, 1]), "Ra": np.random.default_rng(0).uniform(0, 2, size=(3, 3)),
        "cell": np.eye(3) * 4.0, "pbc": np.array([1, 1, 1]),
    }
    batch = [(feature, None, "a")]
    with pytest.raises(ValueError):
        _decorate_batch_input(batch, dtype=torch.float64, device="cpu", otf_graph=True)
