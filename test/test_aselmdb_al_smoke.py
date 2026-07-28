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
    assert not qmd.get("pickle_name")
    assert "bs" not in qmd  # basis lives in template


def test_annotate_pickle_config_selects_pickle_name():
    with open(EXAMPLE / "config" / "annotate_pickle.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["QMDriver"]["pickle_name"] == "fragments.pkl"


def test_legacy_campaign_annotate_still_pickle_shaped():
    with open(EXAMPLE / "config" / "annotate_legacy_pickle.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "pickle_name" in cfg["QMDriver"]


from enerzyme.qm.qm_driver import QMDriver


class FakeQMDriver(QMDriver):
    """Module-level fake so multiprocessing Pool can pickle the worker."""

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
            "E": -1.0 * ase.units.Ha,
            "Fa": np.zeros((n, 3)),
            "M2": np.zeros(3),
        }


def test_mock_qm_driver_writes_aselmdb(tmp_path: Path):
    """Exercise QMDriver.single_run / ASE DB write without calling TeraChem."""
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=2)
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
    assert driver.output_format == "aselmdb"
    driver.run()
    db_files = list(out_dir.rglob("*.aselmdb"))
    assert db_files, f"no aselmdb under {out_dir}"
    assert not list(out_dir.rglob("*.pkl")), "aselmdb mode must not write pickle"
    with connect(str(db_files[0])) as db:
        assert db.count() == 2
        indices = sorted(row.get("index") for row in db.select())
        assert indices == [0, 1]


def test_mock_qm_driver_multiprocess_unique_ids(tmp_path: Path):
    """n_processes>1 must pre-reserve distinct primary keys (no lost/duplicate rows)."""
    from enerzyme.data.supplier import get_supplier

    # Use pickle supplier: Pool pickles the bound method (driver), and RDKit
    # SDMolSupplier is not picklable.
    supplier = get_supplier(
        str(FIXTURES / "fragments_tiny.pkl"),
        start=0,
        end=3,
        features={
            "Ra": "coord",
            "Za": "atom_type",
            "Q": "total_chrg",
            "S": "total_spin",
        },
    )
    driver = FakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=2,
        clean_tmp=True,
    )
    driver.run()
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    assert db_files
    with connect(str(db_files[0])) as db:
        rows = list(db.select())
        assert db.count() == 3
        ids = [row.id for row in rows]
        indices = [row.get("index") for row in rows]
        assert len(set(ids)) == 3
        assert sorted(indices) == [0, 1, 2]


def test_aselmdb_failed_qm_deletes_reservation(tmp_path: Path):
    """Failed QM after a successful write must not crash on delete (bare reserve bug)."""
    from enerzyme.data.supplier import get_supplier

    class FlakyQMDriver(QMDriver):
        def make_input(self, atoms: Atoms, tmp_dir: Path):
            path = tmp_dir / f"{atoms.info['index']}.in"
            path.write_text("run gradient\nend\n")
            return path

        def invoke_qm(self, input_file, atoms: Atoms, tmp_dir: Path):
            out = Path(input_file).with_suffix(".out")
            out.write_text("ok\n")
            return out

        def collect_results(self, input_file, atoms: Atoms, tmp_dir: Path):
            if int(atoms.info["index"]) == 1:
                raise FileNotFoundError("simulated QM failure")
            n = len(atoms)
            return {"E": -1.0 * ase.units.Ha, "Fa": np.zeros((n, 3)), "M2": np.zeros(3)}

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=3)
    driver = FlakyQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=1,
        clean_tmp=True,
    )
    driver.run()
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    assert db_files
    with connect(str(db_files[0])) as db:
        rows = list(db.select())
        assert db.count() == 2
        assert sorted(row.get("index") for row in rows) == [0, 2]


def test_sdf_supplier_reads_fixture():
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=3)
    atoms_list = list(supplier.suppl())
    assert len(atoms_list) == 3
    assert all("charge" in a.info for a in atoms_list)


def test_pickle_supplier_default_features(tmp_path: Path):
    """get_supplier(.pkl) must not require features (identity Ra/Za/Q/S default)."""
    import pickle

    from enerzyme.data.supplier import get_supplier

    pkl = tmp_path / "unlabeled.pkl"
    frames = [
        {
            "Ra": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            "Za": np.array([1, 1], dtype=int),
            "Q": 0,
            "S": 0,
        }
    ]
    with open(pkl, "wb") as f:
        pickle.dump(frames, f)

    supplier = get_supplier(str(pkl))
    atoms_list = list(supplier.suppl())
    assert len(atoms_list) == 1
    assert len(atoms_list[0]) == 2
    assert atoms_list[0].info["charge"] == 0
    assert atoms_list[0].info["spin"] == 1


def test_pickle_supplier_custom_features():
    """Remapped keys (annotate/training pickle schema) via features=."""
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(
        str(FIXTURES / "fragments_tiny.pkl"),
        features={
            "Ra": "coord",
            "Za": "atom_type",
            "Q": "total_chrg",
            "S": "total_spin",
        },
    )
    atoms_list = list(supplier.suppl())
    assert len(atoms_list) == 3
    assert all("charge" in a.info for a in atoms_list)


def test_load_from_sdf_and_datahub(tmp_path: Path):
    from enerzyme.data.datahub import DataHub, load_from_sdf

    sdf = FIXTURES / "fragments_tiny.sdf"
    raw = load_from_sdf(str(sdf))
    assert set(raw) >= {"Ra", "Za", "N", "Q", "S"}
    assert len(raw["Ra"]) == 3
    assert all(np.asarray(za).ndim == 1 for za in raw["Za"])

    hub = DataHub(
        dump_dir=str(tmp_path / "sdf_hub"),
        data_path=str(sdf),
        data_format="sdf",
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q"},
        targets={},
        preload=False,
        neighbor_list="",
    )
    default = hub.datahubs["default"]
    assert default.n_datapoint == 3
    assert "Ra" in default.data and "Za" in default.data and "Q" in default.data



def test_resolve_annotate_output_modes():
    from enerzyme.qm.qm_driver import _resolve_annotate_output

    assert _resolve_annotate_output("fragments.aselmdb", None) == ("aselmdb", "fragments.aselmdb")
    assert _resolve_annotate_output(None, None) == ("aselmdb", "dataset.aselmdb")
    assert _resolve_annotate_output("fragments.pkl", None) == ("pickle", "fragments.pkl")
    assert _resolve_annotate_output("x.aselmdb", "fragments.pkl") == ("pickle", "fragments.pkl")
    assert _resolve_annotate_output(None, "fragments") == ("pickle", "fragments.pkl")


def test_mock_qm_driver_writes_pickle(tmp_path: Path):
    from enerzyme.data.supplier import SDFSupplier
    from enerzyme.qm.qm_driver import QMDriver
    import pickle

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
                "E": -1.0 * ase.units.Ha,
                "Fa": np.zeros((n, 3)),
                "M2": np.zeros(3),
            }

    sdf = FIXTURES / "fragments_tiny.sdf"
    supplier = SDFSupplier(input_file=str(sdf), start=0, end=2)
    driver = FakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        template_input_file=str(FIXTURES / "terachem_template.in"),
        pickle_name="fragments.pkl",
        n_processes=1,
        clean_tmp=True,
    )
    assert driver.output_format == "pickle"
    driver.run()
    pkl = list((tmp_path / "annot_out").rglob("fragments.pkl"))
    assert len(pkl) == 1
    with open(pkl[0], "rb") as f:
        data = pickle.load(f)
    assert len(data) == 2
    assert set(data[0]) >= {"energy", "grad", "dipole", "coord", "atom_type", "total_chrg", "total_spin", "index"}
    assert abs(data[0]["energy"] - (-1.0)) < 1e-8  # Ha
