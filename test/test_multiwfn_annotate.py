"""Unit tests for Multiwfn annotate post-processing (no real TeraChem / Multiwfn)."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import ase.units
import numpy as np
import pytest
from ase import Atoms
from ase.db import connect

from enerzyme.qm.multiwfn import (
    build_multiwfn_stdin,
    chg_dirname,
    parse_chg_file,
    parse_multiwfn_config,
)
from enerzyme.qm.qm_driver import QMDriver

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "example" / "L3-COMT-aselmdb-smoke" / "fixtures"

_FAKE_MULTIWFN = r'''#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

molden = Path(sys.argv[1])
sleep_s = float(os.environ.get("FAKE_MW_SLEEP", "0"))
if sleep_s > 0:
    time.sleep(sleep_s)
n = 2
if molden.is_file():
    for line in molden.read_text().splitlines():
        if line.startswith("NATOMS"):
            n = int(line.split()[1])
            break
log = os.environ.get("FAKE_MW_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(f"mw_end {molden.stem} {time.time():.6f}\n")
with open(f"{molden.stem}.chg", "w") as fh:
    for i in range(n):
        fh.write(f"H    0.000000  0.000000  {float(i):.6f}  {0.01 * (i + 1):.10f}\n")
'''


class MoldenFakeQMDriver(QMDriver):
    """Writes a stub molden so the Multiwfn queue has a valid input."""

    def make_input(self, atoms: Atoms, tmp_dir: Path):
        path = tmp_dir / f"{atoms.info['index']}.in"
        path.write_text("run gradient\nend\n")
        return path

    def invoke_qm(self, input_file, atoms: Atoms, tmp_dir: Path):
        log = os.environ.get("FAKE_TC_LOG")
        if log:
            with open(log, "a") as fh:
                fh.write(f"tc_start {atoms.info['index']} {time.time():.6f}\n")
        out = Path(input_file).with_suffix(".out")
        out.write_text("ok\n")
        return out

    def collect_results(self, input_file, atoms: Atoms, tmp_dir: Path):
        index = Path(input_file).stem
        n = len(atoms)
        molden = tmp_dir / f"scr_{index}" / f"{index}.molden"
        molden.parent.mkdir(parents=True, exist_ok=True)
        molden.write_text(f"NATOMS {n}\n[Molden Format]\nfake\n")
        return {
            "E": -1.0 * ase.units.Ha,
            "Fa": np.zeros((n, 3)),
            "M2": np.array([0.1, 0.2, 0.3]),
            "molden_file": molden,
        }


@pytest.fixture
def fake_multiwfn(tmp_path: Path, monkeypatch):
    bindir = tmp_path / "mwbin"
    bindir.mkdir()
    exe = bindir / "Multiwfn_noGUI"
    exe.write_text(_FAKE_MULTIWFN)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return exe


def _mw_cfg(**kwargs):
    cfg = {
        "enabled": True,
        "executable": "Multiwfn_noGUI",
        "charge_method": "1.2cm5",
        "n_threads": 2,
        "n_processes": 1,
        "keep_chg": True,
    }
    cfg.update(kwargs)
    return cfg


def test_build_stdin_12cm5_and_cm5():
    text = build_multiwfn_stdin("1.2cm5")
    assert text.splitlines() == ["7", "-16", "1", "y", "0", "q"]
    assert build_multiwfn_stdin("cm5").splitlines() == ["7", "16", "1", "y", "0", "q"]
    assert build_multiwfn_stdin("hirshfeld").splitlines()[1] == "1"
    assert build_multiwfn_stdin("adch").splitlines()[1] == "11"
    assert build_multiwfn_stdin("mulliken").splitlines() == ["7", "5", "1", "y", "0", "0", "q"]
    assert build_multiwfn_stdin("lowdin").splitlines() == ["7", "6", "", "y", "0", "q"]
    assert build_multiwfn_stdin("mbis").splitlines()[1] == "20"


def test_build_stdin_menu_inputs_override():
    custom = ["7", "12", "0", "q"]
    assert build_multiwfn_stdin("1.2cm5", menu_inputs=custom).splitlines() == custom


def test_build_stdin_unknown_method():
    with pytest.raises(ValueError, match="Unknown Multiwfn charge_method"):
        build_multiwfn_stdin("hirshfeld-i")


def test_parse_chg_file(tmp_path: Path):
    chg = tmp_path / "0.chg"
    chg.write_text(
        "O    0.0  0.0  0.0  -0.5\n"
        "H    0.0  0.0  1.0   0.25\n"
        "H    0.0  0.0  2.0   0.25\n"
    )
    qa = parse_chg_file(chg, n_atoms=3)
    np.testing.assert_allclose(qa, [-0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="2 atoms"):
        parse_chg_file(chg, n_atoms=2)


def test_chg_dirname_12cm5():
    assert chg_dirname("1.2cm5") == "chrg-12CM5"
    assert chg_dirname("adch") == "chrg-adch"


def test_parse_multiwfn_config_disabled_by_default():
    assert parse_multiwfn_config(None).enabled is False
    assert parse_multiwfn_config({"enabled": False}).enabled is False
    cfg = parse_multiwfn_config({"charge_method": "cm5"})
    assert cfg.enabled is True
    assert cfg.charge_method == "cm5"


def test_aselmdb_writes_qa(tmp_path: Path, fake_multiwfn):
    from enerzyme.data.datahub import ASELMDB_METADATA_PROPERTIES_KEY
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=2)
    driver = MoldenFakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=1,
        clean_tmp=True,
        keep_molden=True,
        multiwfn_config=_mw_cfg(),
    )
    driver.run()
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    assert db_files
    with connect(str(db_files[0])) as db:
        assert "Qa" in db.metadata.get(ASELMDB_METADATA_PROPERTIES_KEY, [])
        assert db.count() == 2
        for row in db.select():
            atoms = row.toatoms()
            qa = atoms.get_charges()
            assert qa.shape == (len(atoms),)
            np.testing.assert_allclose(qa[0], 0.01)
    chg_files = list((tmp_path / "annot_out").rglob("chrg-12CM5/*.chg"))
    assert len(chg_files) == 2


def test_pickle_writes_qa_and_chrg_alias(tmp_path: Path, fake_multiwfn):
    import pickle

    from enerzyme.data.supplier import SDFSupplier

    supplier = SDFSupplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=2)
    driver = MoldenFakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        template_input_file=str(FIXTURES / "terachem_template.in"),
        pickle_name="fragments.pkl",
        pickle_fields={"E": "energy", "Fa": "grad", "M2": "dipole", "Qa": "chrg",
                       "Ra": "coord", "Za": "atom_type", "Q": "total_chrg", "S": "total_spin"},
        n_processes=1,
        clean_tmp=True,
        multiwfn_config=_mw_cfg(),
    )
    driver.run()
    pkl = list((tmp_path / "annot_out").rglob("fragments.pkl"))
    assert pkl
    with open(pkl[0], "rb") as fh:
        data = pickle.load(fh)
    assert len(data) == 2
    assert "chrg" in data[0]
    assert data[0]["chrg"].shape[0] == data[0]["coord"].shape[0]


def test_multiwfn_does_not_block_next_terachem(tmp_path: Path, fake_multiwfn, monkeypatch):
    from enerzyme.data.supplier import get_supplier

    tc_log = tmp_path / "tc.log"
    mw_log = tmp_path / "mw.log"
    monkeypatch.setenv("FAKE_TC_LOG", str(tc_log))
    monkeypatch.setenv("FAKE_MW_LOG", str(mw_log))
    monkeypatch.setenv("FAKE_MW_SLEEP", "0.4")

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=2)
    driver = MoldenFakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=1,
        clean_tmp=True,
        keep_molden=True,
        multiwfn_config=_mw_cfg(),
    )
    driver.run()

    tc = {}
    for line in tc_log.read_text().splitlines():
        _, idx, ts = line.split()
        tc[int(idx)] = float(ts)
    mw = {}
    for line in mw_log.read_text().splitlines():
        _, idx, ts = line.split()
        mw[int(idx)] = float(ts)
    assert 0 in tc and 1 in tc and 0 in mw
    assert tc[1] < mw[0], (
        "TeraChem for structure 1 must start before Multiwfn for structure 0 finishes"
    )


def test_aselmdb_resume_skips_qm_runs_multiwfn_only(tmp_path: Path, fake_multiwfn):
    from enerzyme.data.supplier import get_supplier

    sdf = FIXTURES / "fragments_tiny.sdf"
    common = dict(
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=1,
        clean_tmp=True,
        keep_molden=True,
    )
    call_count = {"n": 0}

    class CountingDriver(MoldenFakeQMDriver):
        def collect_results(self, input_file, atoms: Atoms, tmp_dir: Path):
            call_count["n"] += 1
            return super().collect_results(input_file, atoms, tmp_dir)

    CountingDriver(
        supplier=get_supplier(str(sdf), start=0, end=1),
        multiwfn_config={"enabled": False},
        **common,
    ).run()
    assert call_count["n"] == 1

    call_count["n"] = 0
    CountingDriver(
        supplier=get_supplier(str(sdf), start=0, end=1),
        multiwfn_config=_mw_cfg(),
        **common,
    ).run()
    assert call_count["n"] == 0, "resume must not re-run TeraChem when molden exists"
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    with connect(str(db_files[0])) as db:
        atoms = db.get(index=0).toatoms()
        assert atoms.get_charges() is not None
        assert len(atoms.get_charges()) == len(atoms)


def test_multiprocess_qm_and_multiwfn(tmp_path: Path, fake_multiwfn):
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=3)
    driver = MoldenFakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=2,
        clean_tmp=True,
        keep_molden=True,
        multiwfn_config=_mw_cfg(n_processes=2),
    )
    driver.run()
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    with connect(str(db_files[0])) as db:
        assert db.count() == 3
        for row in db.select():
            assert row.toatoms().get_charges() is not None


def test_disabled_multiwfn_does_not_require_binary(tmp_path: Path):
    """enabled:false must match the pre-Multiwfn annotate path (no executable)."""
    from enerzyme.data.supplier import get_supplier

    supplier = get_supplier(str(FIXTURES / "fragments_tiny.sdf"), start=0, end=1)
    driver = MoldenFakeQMDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        output_file="fragments.aselmdb",
        template_input_file=str(FIXTURES / "terachem_template.in"),
        n_processes=1,
        clean_tmp=True,
        multiwfn_config={"enabled": False},
    )
    driver.run()
    db_files = list((tmp_path / "annot_out").rglob("*.aselmdb"))
    assert db_files
    with connect(str(db_files[0])) as db:
        atoms = db.get(index=0).toatoms()
        assert "charges" not in (atoms.calc.results or {})
    assert not list((tmp_path / "annot_out").rglob("moldens/*.molden")), (
        "keep_molden=false and Multiwfn off must not stage moldens"
    )


def test_pickle_resume_uses_existing_chg_without_rerunning_qm(tmp_path: Path, fake_multiwfn):
    """If Multiwfn wrote .chg but pickle merge never ran, resume must not re-run TeraChem."""
    import pickle

    from enerzyme.data.supplier import SDFSupplier

    sdf = FIXTURES / "fragments_tiny.sdf"
    call_count = {"n": 0}

    class CountingDriver(MoldenFakeQMDriver):
        def collect_results(self, input_file, atoms: Atoms, tmp_dir: Path):
            call_count["n"] += 1
            return super().collect_results(input_file, atoms, tmp_dir)

    common = dict(
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        template_input_file=str(FIXTURES / "terachem_template.in"),
        pickle_name="fragments.pkl",
        dump_single_run=True,
        n_processes=1,
        clean_tmp=True,
        keep_molden=False,
        multiwfn_config=_mw_cfg(),
    )
    CountingDriver(supplier=SDFSupplier(str(sdf), start=0, end=1), **common).run()
    assert call_count["n"] == 1

    single_runs = list((tmp_path / "annot_out").rglob("single_run/0.pkl"))
    assert single_runs
    with open(single_runs[0], "rb") as fh:
        cached = pickle.load(fh)
    cached.pop("Qa", None)
    with open(single_runs[0], "wb") as fh:
        pickle.dump(cached, fh)
    for molden in (tmp_path / "annot_out").rglob("moldens/*.molden"):
        molden.unlink()
    chg_files = list((tmp_path / "annot_out").rglob("chrg-12CM5/0.chg"))
    assert chg_files, "resume fixture requires the Multiwfn .chg to survive"

    call_count["n"] = 0
    CountingDriver(supplier=SDFSupplier(str(sdf), start=0, end=1), **common).run()
    assert call_count["n"] == 0, "existing .chg must skip TeraChem even without a molden"

    pkl = list((tmp_path / "annot_out").rglob("fragments.pkl"))
    with open(pkl[0], "rb") as fh:
        data = pickle.load(fh)
    expected = parse_chg_file(chg_files[0])
    np.testing.assert_allclose(data[0]["Qa"], expected)


def test_stale_chg_cleared_when_terachem_reruns(tmp_path: Path, fake_multiwfn):
    """A leftover .chg must not survive a fresh TeraChem run for the same index."""
    import pickle

    from enerzyme.data.supplier import SDFSupplier

    sdf = FIXTURES / "fragments_tiny.sdf"
    common = dict(
        tmp_dir=str(tmp_path / "annot_tmp"),
        output_dir=str(tmp_path / "annot_out"),
        template_input_file=str(FIXTURES / "terachem_template.in"),
        pickle_name="fragments.pkl",
        dump_single_run=True,
        n_processes=1,
        clean_tmp=True,
        keep_molden=False,
        multiwfn_config=_mw_cfg(),
    )
    driver = MoldenFakeQMDriver(
        supplier=SDFSupplier(str(sdf), start=0, end=1),
        **common,
    )
    driver.run()

    out_root = tmp_path / "annot_out" / "fragments_tiny_0_1"
    chg_path = out_root / "chrg-12CM5" / "0.chg"
    assert chg_path.is_file()
    n_atoms = len(SDFSupplier(str(sdf), start=0, end=1).suppl().__next__())
    stale = np.full(n_atoms, 9.99)
    with open(chg_path, "w") as fh:
        for i, q in enumerate(stale):
            fh.write(f"H  0.0  0.0  {float(i):.6f}  {q:.10f}\n")

    for path in out_root.rglob("single_run/*.pkl"):
        path.unlink()
    for path in out_root.rglob("moldens/*.molden"):
        path.unlink()

    MoldenFakeQMDriver(supplier=SDFSupplier(str(sdf), start=0, end=1), **common).run()
    pkl = list(out_root.rglob("fragments.pkl"))
    with open(pkl[0], "rb") as fh:
        data = pickle.load(fh)
    qa = np.asarray(data[0]["Qa"], dtype=float)
    assert not np.allclose(qa, 9.99), "stale .chg must not be merged after TeraChem rerun"
    np.testing.assert_allclose(qa[0], 0.01)
