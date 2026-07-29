"""Minimal tests for ASE LMDB dataset helpers."""
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
    assert _get_single_aselmdb_data_path(str(f1)) == [str(f1)]
    found = set(_get_single_aselmdb_data_path(str(tmp_path)))
    assert str(f1) in found and str(f2) in found


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
