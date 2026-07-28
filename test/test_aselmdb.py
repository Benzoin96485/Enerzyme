"""Minimal tests for ASE LMDB dataset helpers."""
import ase.units
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from enerzyme.data.datahub import ASELMDBDataset, _get_single_aselmdb_data_path


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
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    energy_ev = -1.0
    forces_ev = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces_ev)
    with connect(str(db_path)) as db:
        db.write(atoms, data={"charge": 0, "spin": 1, "index": 0}, index=0)

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
