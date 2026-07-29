"""Minimal tests for ASE LMDB dataset helpers."""
from pathlib import Path

import ase.units
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from enerzyme.data.datahub import ASELMDBDataset, SingleDataHub, _get_single_aselmdb_data_path


def _write_toy_aselmdb(db_path, energy_ev=-1.0, forces_ev=None, dipole=None, charge=0, spin=1, index=0):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    if forces_ev is None:
        forces_ev = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    calc_kwargs = {"energy": energy_ev, "forces": forces_ev}
    if dipole is not None:
        calc_kwargs["dipole"] = np.asarray(dipole, dtype=float)
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    with connect(str(db_path)) as db:
        db.write(atoms, data={"charge": charge, "spin": spin, "index": index}, index=index)
    return energy_ev, forces_ev


def test_get_single_aselmdb_data_path_file_and_glob(tmp_path):
    f1 = tmp_path / "a.aselmdb"
    f2 = tmp_path / "b.aselmdb"
    f1.write_text("")
    f2.write_text("")
    (tmp_path / "README.txt").write_text("ignore me")
    (tmp_path / "a.aselmdb-lock").write_text("lock")
    (tmp_path / "notes.json").write_text("{}")
    (tmp_path / "subdir").mkdir()
    assert _get_single_aselmdb_data_path(str(f1)) == [str(f1)]
    found = _get_single_aselmdb_data_path(str(tmp_path))
    assert found == sorted([str(f1), str(f2)])
    # Explicit globs still work; lock files are filtered out.
    globbed = _get_single_aselmdb_data_path(str(tmp_path / "*.aselmdb*"))
    assert set(globbed) == {str(f1), str(f2)}


def test_aselmdb_dataset_directory_skips_non_db_and_fails_loud(tmp_path):
    """Directory loads must ignore junk/locks and raise on a bad DB file."""
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    _write_toy_aselmdb(db_dir / "a.aselmdb", energy_ev=-1.0, index=0)
    _write_toy_aselmdb(db_dir / "b.aselmdb", energy_ev=-2.0, index=1)
    (db_dir / "README.txt").write_text("not a db")
    (db_dir / "a.aselmdb-lock").write_text("lock")
    (db_dir / "notes.json").write_text("{}")

    ds = ASELMDBDataset(str(db_dir), new_energy_unit="Ha")
    assert len(ds) == 2
    assert len(ds.dbs) == 2
    assert len(ds.data_paths) == 2
    assert {Path(p).name for p in ds.data_paths} == {"a.aselmdb", "b.aselmdb"}

    bad = db_dir / "corrupt.aselmdb"
    bad.write_text("not-a-real-aselmdb")
    with pytest.raises(ValueError, match="Failed to connect"):
        ASELMDBDataset(str(db_dir), new_energy_unit="Ha")


def test_aselmdb_dataset_loads_energy_forces_and_qs(tmp_path):
    db_path = tmp_path / "toy.aselmdb"
    energy_ev, forces_ev = _write_toy_aselmdb(db_path)

    ds = ASELMDBDataset(str(db_path), new_energy_unit="Ha")
    assert len(ds) == 1
    assert "E" in ds and "Fa" in ds and "Q" in ds and "S" in ds
    e = ds["E"][0]
    fa = ds["Fa"][0]
    assert e == pytest.approx(energy_ev / ase.units.Ha)
    np.testing.assert_allclose(fa, forces_ev / ase.units.Ha)
    assert ds["Q"][0] == 0
    assert ds["S"][0] == 0
    assert ds["N"][0] == 2


def test_aselmdb_dataset_loads_dipole_as_m2(tmp_path):
    """Annotate stores ASE ``dipole``; Datahub exposes standard ``M2`` only."""
    db_path = tmp_path / "dipole.aselmdb"
    dipole = np.array([0.11, -0.22, 0.33])
    _write_toy_aselmdb(db_path, dipole=dipole)

    ds = ASELMDBDataset(str(db_path), new_energy_unit="Ha")
    assert "M2" in ds
    assert "dipole" not in ds
    assert "charge" not in ds and "spin" not in ds
    np.testing.assert_allclose(ds["M2"][0], dipole)


def test_aselmdb_declared_m2_when_first_row_lacks_dipole(tmp_path):
    """Declared / schema registration must not depend on the first row alone."""
    from enerzyme.data.datahub import ASELMDB_METADATA_PROPERTIES_KEY

    db_path = tmp_path / "sparse_dipole.aselmdb"
    dipole = np.array([0.11, -0.22, 0.33])
    _write_toy_aselmdb(db_path, energy_ev=-1.0, index=0)  # no dipole
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-2.0,
        forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        dipole=dipole,
    )
    with connect(str(db_path)) as db:
        db.write(atoms, data={"charge": 0, "spin": 1, "index": 1}, index=1)

    # Without help, first-row probe misses M2.
    ds_probe = ASELMDBDataset(str(db_path), new_energy_unit="Ha")
    assert "M2" not in ds_probe

    # Declared Datahub targets register M2 even when row 0 lacks dipole.
    ds_declared = ASELMDBDataset(
        str(db_path),
        new_energy_unit="Ha",
        declared_properties=["E", "Fa", "M2"],
    )
    assert "M2" in ds_declared
    np.testing.assert_allclose(ds_declared["M2"][1], dipole)

    # Writer schema likewise registers M2 without probing row 0.
    with connect(str(db_path)) as db:
        meta = dict(db.metadata or {})
        meta[ASELMDB_METADATA_PROPERTIES_KEY] = ["Ra", "Za", "N", "Q", "S", "E", "Fa", "M2"]
        db.metadata = meta
    ds_schema = ASELMDBDataset(str(db_path), new_energy_unit="Ha")
    assert "M2" in ds_schema
    np.testing.assert_allclose(ds_schema["M2"][1], dipole)


def test_aselmdb_dataset_handles_none_row_data(tmp_path, monkeypatch):
    """Rows with data=None / null must still expose calculator-backed E/Fa."""
    from ase_db_backends.aselmdb import LMDBDatabase

    db_path = tmp_path / "none_data.aselmdb"
    energy_ev, forces_ev = _write_toy_aselmdb(db_path)

    orig_get_row = LMDBDatabase._get_row
    probed = {"done": False}

    def _patched_get_row(self, id, *args, **kwargs):
        row = orig_get_row(self, id, *args, **kwargs)
        if not probed["done"]:
            row._data = None  # ASE AtomsRow.data then raises TypeError
            probed["done"] = True
        return row

    monkeypatch.setattr(LMDBDatabase, "_get_row", _patched_get_row)
    ds = ASELMDBDataset(str(db_path), new_energy_unit="Ha")

    assert "E" in ds and "Fa" in ds
    assert ds.properties_from_info == set()
    assert ds["E"][0] == pytest.approx(energy_ev / ase.units.Ha)
    np.testing.assert_allclose(ds["Fa"][0], forces_ev / ase.units.Ha)


def test_singledatahub_loads_m2_from_aselmdb(tmp_path):
    """HDF5 build must include M2 via identity mapping (standard names)."""
    db_path = tmp_path / "dipole.aselmdb"
    dipole = np.array([1.0, 2.0, 3.0])
    _write_toy_aselmdb(db_path, dipole=dipole)

    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=str(db_path),
        data_format="aselmdb",
        preload=False,
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
        targets={"E": "E", "Fa": "Fa", "M2": "M2"},
        neighbor_list="",
        compressed=False,
    )
    assert "M2" in hub.data
    np.testing.assert_allclose(hub.data["M2"][0], dipole)


def test_singledatahub_aselmdb_ignores_negative_gradient(tmp_path):
    """ASELMDB Fa is already physical forces; negative_gradient must not flip them."""
    from enerzyme.data.transform import NegativeGradientTransform

    db_path = tmp_path / "toy.aselmdb"
    forces_ev = np.array([[0.25, -0.1, 0.0], [-0.25, 0.1, 0.0]])
    _write_toy_aselmdb(db_path, forces_ev=forces_ev)
    expected_fa = forces_ev / ase.units.Ha

    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=str(db_path),
        data_format="aselmdb",
        preload=False,
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
        targets={"E": "E", "Fa": "Fa"},
        neighbor_list="",
        compressed=False,
        preprocessings={"negative_gradient": True},
        global_transforms={"negative_gradient": True},
    )
    assert not any(isinstance(s, NegativeGradientTransform) for s in hub.preprocessing.scales)
    assert not any(isinstance(s, NegativeGradientTransform) for s in hub.global_transform.scales)
    np.testing.assert_allclose(hub.data["Fa"][0, :2], expected_fa)
    hub.file.close()


def test_singledatahub_rejects_pickle_style_aselmdb_maps(tmp_path):
    """Pickle aliases (E: energy, Ra: coord) must fail loudly for aselmdb."""
    db_path = tmp_path / "toy.aselmdb"
    _write_toy_aselmdb(db_path)

    with pytest.raises(ValueError, match="identity maps"):
        SingleDataHub(
            dump_dir=str(tmp_path / "out"),
            data_path=str(db_path),
            data_format="aselmdb",
            preload=False,
            features={"Ra": "coord", "Za": "atom_type", "N": "N", "Q": "total_chrg"},
            targets={"E": "energy", "Fa": "grad"},
            neighbor_list="",
            compressed=False,
        )


def test_singledatahub_raises_on_missing_molecular_source(tmp_path):
    """Declared molecular fields that no row can supply must fail while loading."""
    db_path = tmp_path / "toy.aselmdb"
    _write_toy_aselmdb(db_path)  # no dipole on any row

    with pytest.raises(Exception, match=r"dipole"):
        SingleDataHub(
            dump_dir=str(tmp_path / "out"),
            data_path=str(db_path),
            data_format="aselmdb",
            preload=False,
            features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
            # Declared M2 is registered, then get_dipole_moment fails on the toy row.
            targets={"E": "E", "Fa": "Fa", "M2": "M2"},
            neighbor_list="",
            compressed=False,
        )


def test_singledatahub_raises_on_unknown_custom_source(tmp_path):
    """Custom targets absent from row data still raise KeyError (not silent skip)."""
    db_path = tmp_path / "toy.aselmdb"
    _write_toy_aselmdb(db_path)

    with pytest.raises(KeyError, match="Requested field 'foo'"):
        SingleDataHub(
            dump_dir=str(tmp_path / "out"),
            data_path=str(db_path),
            data_format="aselmdb",
            preload=False,
            features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
            targets={"E": "E", "Fa": "Fa", "foo": "foo"},
            neighbor_list="",
            compressed=False,
        )


@pytest.mark.parametrize(
    "path_kind",
    ["directory", "glob"],
)
def test_singledatahub_loads_aselmdb_directory_or_glob(tmp_path, path_kind):
    """SingleDataHub must accept multi-DB folder layouts, not only a single file."""
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    _write_toy_aselmdb(db_dir / "a.aselmdb", energy_ev=-1.0, index=0)
    _write_toy_aselmdb(db_dir / "b.aselmdb", energy_ev=-2.0, index=1)

    if path_kind == "directory":
        data_path = str(db_dir)
    else:
        data_path = str(db_dir / "*.aselmdb")

    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=data_path,
        data_format="aselmdb",
        preload=False,
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
        targets={"E": "E", "Fa": "Fa"},
        neighbor_list="",
        compressed=False,
    )
    assert hub.n_datapoint == 2
    assert hub.data["E"].shape[0] == 2


def test_singledatahub_preload_hash_includes_aselmdb_args_when_set(tmp_path):
    """Empty connect/select args and unset data_format must not change the hash;
    non-empty values must, so stale pre_transformed.hdf5 is not reused."""
    db_path = tmp_path / "toy.aselmdb"
    _write_toy_aselmdb(db_path)
    common = dict(
        data_path=str(db_path),
        preload=False,
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
        targets={"E": "E", "Fa": "Fa"},
        neighbor_list="",
        compressed=False,
    )
    n_hubs = 0

    def _hash(**kwargs):
        nonlocal n_hubs
        n_hubs += 1
        hub = SingleDataHub(dump_dir=str(tmp_path / f"out_{n_hubs}"), **common, **kwargs)
        h = hub.hash
        hub.file.close()
        return h

    baseline = _hash()
    assert _hash(data_format=None, connect_args={}, select_args={}) == baseline

    with_format = _hash(data_format="aselmdb")
    with_select = _hash(select_args={"limit": 1})
    with_connect = _hash(connect_args={"readonly": False})
    assert with_format != baseline
    assert with_select != baseline
    assert with_connect != baseline
    assert len({with_format, with_select, with_connect}) == 3


def test_resolve_keep_stdout_accepts_legacy_keep_output():
    from enerzyme.qm.qm_driver import _resolve_keep_stdout

    assert _resolve_keep_stdout(False, {}) is False
    assert _resolve_keep_stdout(True, {}) is True

    kwargs = {"keep_output": True}
    assert _resolve_keep_stdout(False, kwargs) is True
    assert "keep_output" not in kwargs

    kwargs = {"keep_output": False}
    assert _resolve_keep_stdout(False, kwargs) is False

    kwargs = {"keep_output": False}
    assert _resolve_keep_stdout(True, kwargs) is True
