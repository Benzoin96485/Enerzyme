"""Integration smoke for L3-COMT-style AL datahub / annotate with ASELMDB (PR #80).

Uses the tiny fixtures under ``example/L3-COMT-aselmdb-smoke`` (no checkpoints, no full DB).
TeraChem is not required: annotate write path is exercised with a FakeQMDriver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import ase.units
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.db import connect

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "L3-COMT-aselmdb-smoke"
SCRIPTS = EXAMPLE / "scripts"
FIXTURES = EXAMPLE / "fixtures"

sys.path.insert(0, str(SCRIPTS))
from pickle_to_aselmdb import convert  # noqa: E402


@pytest.fixture
def tiny_aselmdb(tmp_path: Path) -> Path:
    out = tmp_path / "fragments_tiny.aselmdb"
    n = convert(FIXTURES / "fragments_tiny.pkl", out)
    assert n == 3
    return out


def test_pickle_to_aselmdb_roundtrip_fields(tiny_aselmdb: Path):
    from enerzyme.data.datahub import ASELMDBDataset

    ds = ASELMDBDataset(str(tiny_aselmdb), new_energy_unit="Ha")
    assert len(ds) == 3
    assert "E" in ds and "Fa" in ds and "Q" in ds and "S" in ds
    e0 = float(ds["E"][0])
    fa0 = np.asarray(ds["Fa"][0])
    assert np.isfinite(e0)
    assert fa0.shape[1] == 3
    # Energy should be back in Ha (ASE stored eV)
    assert abs(e0) > 1.0  # Hartree-scale fragment energy


def test_datahub_yaml_loads_aselmdb(tiny_aselmdb: Path, tmp_path: Path):
    from enerzyme.data.datahub import DataHub

    cfg_path = EXAMPLE / "config" / "train_aselmdb.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    datahub_cfg = cfg["Datahub"]
    datahub_cfg["data_path"] = str(tiny_aselmdb)
    # Resolve relative atomic_energy path
    ae = FIXTURES / "atomic_energy.csv"
    datahub_cfg["transforms"]["atomic_energy"] = str(ae)
    datahub_cfg["global_transforms"]["atomic_energy"] = str(ae)

    hub = DataHub(dump_dir=str(tmp_path / "out"), **datahub_cfg)
    default = hub.datahubs["default"]
    assert default.n_datapoint == 3
    assert "E" in default.targets.keys() or "E" in default.data
    assert default.n_datapoint == len(default.features["Ra"]) or default.n_datapoint == 3


def test_annotate_config_matches_new_api():
    with open(EXAMPLE / "config" / "annotate.yaml") as f:
        cfg = yaml.safe_load(f)
    qmd = cfg["QMDriver"]
    assert qmd["engine"].lower() == "terachem"
    assert "output_file" in qmd
    assert qmd["output_file"].endswith(".aselmdb")
    assert "template_input_file" in qmd
    assert "pickle_name" not in qmd
    assert "bs" not in qmd  # basis lives in template


def test_legacy_campaign_annotate_still_pickle_shaped():
    with open(EXAMPLE / "config" / "annotate_legacy_pickle.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "pickle_name" in cfg["QMDriver"]


def test_mock_qm_driver_writes_aselmdb(tmp_path: Path):
    """Exercise QMDriver.single_run / ASE DB write without calling TeraChem."""
    from enerzyme.data.supplier import SDFSupplier
    from enerzyme.qm.qm_driver import QMDriver

    class FakeQMDriver(QMDriver):
        def make_input(self, atoms: Atoms, tmp_dir: Path):
            path = tmp_dir / f"{atoms.info['index']}.in"
            path.write_text("run gradient\nend\n")
            return path

        def invoke_qm(self, input_file, atoms: Atoms, tmp_dir: Path):
            out = Path(input_file).with_suffix(".out")
            out.write_text("ok\n")
            return out

        def collect_results(self, input_file, atoms: Atoms, tmp_dir: Path):
            n = len(atoms)
            return {
                "E": -1.0 * ase.units.Ha,  # Ha; driver stores via SPC as given
                "Fa": np.zeros((n, 3)),
                "M2": np.zeros(3),
            }

    sdf = FIXTURES / "fragments_tiny.sdf"
    supplier = SDFSupplier(input_file=str(sdf), start=0, end=2)
    out_dir = tmp_path / "annot_out"
    tmp_dir = tmp_path / "annot_tmp"
    template = FIXTURES / "terachem_template.in"
    driver = FakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_dir),
        output_dir=str(out_dir),
        output_file="fragments.aselmdb",
        template_input_file=str(template),
        n_processes=1,
        clean_tmp=True,
    )
    # Fake collect_results returns E in Ha but SinglePointCalculator expects eV-like floats;
    # for write-path smoke we only assert DB rows exist.
    driver.run()
    db_files = list(out_dir.rglob("*.aselmdb"))
    assert db_files, f"no aselmdb under {out_dir}"
    with connect(str(db_files[0])) as db:
        assert db.count() == 2


def test_sdf_supplier_reads_fixture():
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=3)
    atoms_list = list(supplier.suppl())
    assert len(atoms_list) == 3
    assert all("charge" in a.info for a in atoms_list)
