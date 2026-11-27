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


@pytest.mark.parametrize("pyg", [False, True])
@pytest.mark.parametrize("periodic_first", [False, True])
def test_mixed_open_and_periodic_collation_uses_global_indices(pyg, periodic_first):
    closed = {"N": 2, "Ra": np.array([[0.1, 0., 0.], [1.9, 0., 0.]]), "Za": [1, 1],
              "cell": 2*np.eye(3), "pbc": [1, 0, 0], "idx_i": [0, 1, -1], "idx_j": [1, 0, -1],
              "offsets": [[-2., 0., 0.], [2., 0., 0.], [0., 0., 0.]], "N_pair": 2}
    opened = {"N": 2, "Ra": np.array([[0., 0., 0.], [0.5, 0., 0.]]), "Za": [1, 1]}
    frames = [closed, opened] if periodic_first else [opened, closed]
    collate = _decorate_pyg_batch_input if pyg else _decorate_batch_input
    batch, _ = collate([(f, {}, "sample") for f in frames], torch.float64, "cpu")
    assert batch["cell"].shape == (2, 3, 3) and batch["pbc"].shape == (2, 3)
    assert batch["idx_i"].tolist() == [0, 1, 2, 3]
    assert batch["idx_j"].tolist() == [1, 0, 3, 2]
    if pyg:
        assert torch.equal(batch.edge_index, torch.stack((batch.idx_i, batch.idx_j)))
        for graph in batch.to_data_list():
            assert graph.idx_i.tolist() == [0, 1] and graph.idx_j.tolist() == [1, 0]
    distances = (batch["Ra"][batch["idx_j"]] + batch["offsets"] - batch["Ra"][batch["idx_i"]]).norm(dim=-1)
    values = [0.2, 0.2, 0.5, 0.5] if periodic_first else [0.5, 0.5, 0.2, 0.2]
    torch.testing.assert_close(distances, torch.tensor(values, dtype=torch.float64))


@pytest.mark.parametrize("pyg", [False, True])
def test_open_metadata_can_use_on_the_fly_and_periodic_edges_require_shifts(pyg):
    collate = _decorate_pyg_batch_input if pyg else _decorate_batch_input
    frame = {"N": 2, "Ra": [[0., 0., 0.], [0.5, 0., 0.]], "Za": [1, 1],
             "cell": np.zeros((3,3)), "pbc": [False]*3}
    batch, _ = collate([(frame, {}, "sample")], torch.float64, "cpu")
    assert batch["idx_i"].tolist() == [0, 1]
    periodic = dict(frame, cell=np.eye(3), pbc=[True, False, False], idx_i=[0, 1], idx_j=[1, 0])
    with pytest.raises(ValueError, match="offsets"):
        collate([(periodic, {}, "sample")], torch.float64, "cpu")


def test_cutoff_cache_distinguishes_boundary_conditions_and_geometry_alias(tmp_path):
    path = tmp_path / "geometry.pkl"
    fields = {"N": 2, "Za": [1, 1], "Ra": [[0.1, 0., 0.], [1.9, 0., 0.]],
              "alternative": [[0.1, 0., 0.], [1., 0., 0.]], "cell": 2*np.eye(3), "pbc": [1,0,0]}
    _write_pickle(path, [fields, fields])

    def cache(features):
        h = SingleDataHub(dump_dir=str(tmp_path), data_path=str(path), data_format="pickle",
                          features=features, targets={}, neighbor_list="cutoff",
                          neighbor_list_cutoff=0.5, preload=True)
        n = np.asarray(h.data["N_pair"]).tolist()
        h.file.close()
        return h.preload_path, n

    common = {"N": None, "Ra": None, "Za": None}
    periodic = {**common, "cell": None, "pbc": None}
    a = cache(common)
    b = cache(periodic)
    c = cache({**periodic, "Ra": "alternative"})
    assert [a[1], b[1], c[1]] == [[0, 0], [2, 2], [0, 0]]
    assert len({a[0], b[0], c[0]}) == 3
    assert cache(periodic) == b


@pytest.mark.parametrize("data_format", ["hdf5", "aselmdb"])
def test_periodic_training_geometry_from_database_and_preloaded_cache(tmp_path, data_format):
    cell = np.diag([2., 3., 4.])
    positions = np.array([[0.1, 0., 0.], [1.9, 0., 0.]])
    path = tmp_path / f"frames.{data_format}"
    if data_format == "hdf5":
        import h5py

        with h5py.File(path, "w") as store:
            raw = store.create_group("data")
            raw["Ra"] = [positions, positions]
            raw["N"] = [2, 2]
            raw["Za"] = [[1, 1], [1, 1]]
            raw["cell"] = [cell]  # One shared lattice, two configurations.
            raw["pbc"] = [[True, False, False], [False, False, False]]
    else:
        from ase import Atoms
        from ase.db import connect

        with connect(str(path)) as db:
            for pbc in ([True, False, False], [False, False, False]):
                db.write(Atoms("H2", positions=positions, cell=cell, pbc=pbc))

    options = dict(dump_dir=str(tmp_path), data_path=str(path), data_format=data_format,
                   features=["N", "Za", "Ra", "cell", "pbc"], targets={},
                   neighbor_list="cutoff", neighbor_list_cutoff=0.4, preload=True)
    for _ in range(2):
        hub = SingleDataHub(**options)
        try:
            assert hub.n_datapoint == 2 and hub.max_N == 2
            assert np.asarray(hub.data["N_pair"]).tolist() == [2, 0]
            assert np.asarray(hub.data["idx_i"][1]).tolist() == [-1, -1]
            assert not np.any(hub.data["offsets"][1])
            np.testing.assert_allclose(hub.features.loc(0)["cell"], cell)
        finally:
            hub.file.close()


def test_full_neighbor_list_rejects_a_periodic_dataset(tmp_path):
    path = tmp_path / "periodic.pkl"
    frame = {"N": 2, "Za": [1, 1], "Ra": [[0.1, 0., 0.], [1.9, 0., 0.]],
             "cell": 2*np.eye(3), "pbc": [True, False, False]}
    _write_pickle(path, [frame, frame])
    with pytest.raises(ValueError, match="Periodic.*cutoff"):
        SingleDataHub(dump_dir=str(tmp_path), data_path=str(path), data_format="pickle",
                      features=["N", "Ra", "Za", "pbc", "cell"], targets={}, neighbor_list="full")
