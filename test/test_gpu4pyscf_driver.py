"""GPU4PySCF / Electrolites annotate interface. No GPU libraries required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from enerzyme.qm.electrolites_plugin import (
    ElectrolitesPlugin,
    ElectrolitesPluginError,
    apply_electrolites,
    parse_electrolites,
)
from enerzyme.qm.gpu4pyscf import (
    STOCK_CPCM,
    Gpu4PyscfError,
    build_job_spec,
    energy_ha_to_ev,
    grad_ha_bohr_to_forces_ev_a,
    load_gpu4pyscf_settings,
)
from enerzyme.qm.qm_driver import GPU4PySCFDriver

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "enerzyme"
    / "qm"
    / "templates"
    / "gpu4pyscf_wb97mv.yaml"
)
CPCM_TEMPLATE = TEMPLATE.with_name("gpu4pyscf_wb97mv_cpcm.yaml")


def _water(**info) -> Atoms:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.239, 0.927, 0.0]],
    )
    payload = {"index": 7, "charge": 0, "spin": 1}
    payload.update(info)
    atoms.info.update(payload)
    return atoms


def _driver(tmp_path: Path, template: Path, **kwargs) -> GPU4PySCFDriver:
    supplier = type("S", (), {"name": "water"})()
    return GPU4PySCFDriver(
        supplier=supplier,
        tmp_dir=str(tmp_path / "tmp"),
        output_dir=str(tmp_path / "out"),
        template_input_file=str(template),
        n_processes=1,
        clean_tmp=False,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _reset_plugin_process_state():
    import enerzyme.qm.electrolites_plugin as plugin

    plugin._PROCESS_PLUGIN = None
    yield
    plugin._PROCESS_PLUGIN = None


def test_template_loads_stock_direct_scf():
    settings = load_gpu4pyscf_settings(TEMPLATE)
    assert settings.xc == "wb97m-v"
    assert settings.basis == "def2-tzvpd"
    assert settings.cart is False
    assert settings.pcm is None
    assert settings.electrolites == ElectrolitesPlugin(enabled=False)
    assert settings.precision_env["FASTNLC_MODE"] == "plain"
    assert settings.precision_env["FASTNLC_RCUT"] == "inf"
    assert "gpu4pyscf" not in sys.modules


def test_cpcm_template_is_stock_swig():
    settings = load_gpu4pyscf_settings(CPCM_TEMPLATE)
    assert settings.pcm["method"] == "C-PCM"
    assert settings.pcm["vdw_scale"] == pytest.approx(STOCK_CPCM["vdw_scale"])
    assert settings.pcm["lebedev_order"] == 29
    assert settings.pcm["eps"] == pytest.approx(80.0)


def test_parse_electrolites_forms():
    assert parse_electrolites(False).enabled is False
    assert parse_electrolites("all").modules is None
    assert parse_electrolites("fastk, fastj").modules == ("fastk", "fastj")
    assert parse_electrolites(["fastk"]).modules == ("fastk",)
    spec = parse_electrolites({"enabled": True, "modules": "all"})
    assert spec.enabled is True and spec.modules is None
    assert parse_electrolites({"enabled": False, "modules": "fastk"}).enabled is False


def test_apply_electrolites_refuses_a_second_choice(monkeypatch):
    class FakeElectrolites:
        def __init__(self):
            self.names: list[str] = []

        def patched(self):
            return tuple(self.names)

        def patch_all(self):
            self.names = ["fastk", "fastj"]
            return tuple(self.names)

        def patch(self, *names):
            self.names = list(names)
            return tuple(self.names)

    monkeypatch.setitem(sys.modules, "electrolites", FakeElectrolites())
    first = apply_electrolites(parse_electrolites("fastk"))
    assert first == ("fastk",)
    assert apply_electrolites(parse_electrolites("fastk")) == ("fastk",)
    with pytest.raises(ElectrolitesPluginError, match="process-global"):
        apply_electrolites(parse_electrolites(False))


def test_stock_refuses_an_already_patched_process(monkeypatch):
    class AlreadyPatched:
        def patched(self):
            return ("fastk",)

    monkeypatch.setitem(sys.modules, "electrolites", AlreadyPatched())
    with pytest.raises(ElectrolitesPluginError, match="already patched"):
        apply_electrolites(parse_electrolites(False))


def test_job_spec_uses_multiplicity_and_epsilon_override():
    settings = load_gpu4pyscf_settings(TEMPLATE)
    spec = build_job_spec(_water(charge=-1, spin=3, epsilon=7.3), settings, settings.electrolites)
    assert spec["charge"] == -1
    assert spec["multiplicity"] == 3
    assert spec["pyscf_spin"] == 2
    assert spec["pcm"]["eps"] == pytest.approx(7.3)
    assert spec["pcm"]["surface_discretization_method"] == "SWIG"
    vacuum = build_job_spec(_water(epsilon="vacuum"), settings, parse_electrolites(True))
    assert vacuum["pcm"] is None
    assert vacuum["electrolites"]["enabled"] is True


def test_unknown_template_key_and_bad_pcm(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("xc: hf\nbasis: sto-3g\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(Gpu4PyscfError, match="unrecognized"):
        load_gpu4pyscf_settings(bad)
    pcm = tmp_path / "pcm.yaml"
    pcm.write_text(
        "xc: wb97m-v\nbasis: def2-tzvpd\npcm:\n  method: SMD\n  eps: 80\n",
        encoding="utf-8",
    )
    with pytest.raises(Gpu4PyscfError, match="C-PCM"):
        load_gpu4pyscf_settings(pcm)


def test_driver_round_trip_units_without_gpu(tmp_path: Path, monkeypatch):
    import os

    captured = {}

    def fake_runner(spec):
        captured["spec"] = spec
        return {
            "ok": True,
            "scf_converged": True,
            "scf_cycles": 4,
            "energy_ha": -1.0,
            "grad_ha_bohr": [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
            "dipole_au": [0.0, 0.0, 1.0],
            "electrolites_patched": [],
            "error": None,
        }

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    driver = _driver(tmp_path, TEMPLATE, electrolites="fastk,fastj", runner=fake_runner, n_gpus=2)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    atoms = _water()
    inp = driver.make_input(atoms, scratch)
    spec = json.loads(inp.read_text(encoding="utf-8"))
    assert spec["electrolites"]["modules"] == ["fastk", "fastj"]
    assert spec["xc"] == "wb97m-v"
    out = driver.invoke_qm(inp, atoms, scratch)
    assert out.is_file()
    assert captured["spec"]["index"] == 7
    package = driver.collect_results(inp, atoms, scratch)
    assert package["E"] == pytest.approx(energy_ha_to_ev(-1.0))
    np.testing.assert_allclose(
        package["Fa"],
        grad_ha_bohr_to_forces_ev_a([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(package["M2"], np.array([0.0, 0.0, Bohr]))
    assert package["E"] == pytest.approx(-Hartree)
    # The main process is slot 0. Pool workers use their process identity.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_driver_collect_requires_converged_scf(tmp_path: Path):
    driver = _driver(tmp_path, TEMPLATE)

    def failed(_spec):
        return {"scf_converged": False, "energy_ha": -1.0, "error": "SCF did not converge"}

    driver.runner = failed
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    atoms = _water()
    inp = driver.make_input(atoms, scratch)
    driver.invoke_qm(inp, atoms, scratch)
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        driver.collect_results(inp, atoms, scratch)


def test_annotate_dispatches_gpu4pyscf(tmp_path: Path):
    from enerzyme.annotate import QMAnnotate
    from enerzyme.qm.qm_driver import GPU4PySCFDriver

    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    cfg = tmp_path / "ann.yaml"
    cfg.write_text(
        "Supplier:\n"
        f"  path: {xyz}\n"
        "QMDriver:\n"
        "  engine: gpu4pyscf\n"
        f"  template_input_file: {TEMPLATE}\n"
        "  electrolites: false\n"
        "  n_processes: 1\n",
        encoding="utf-8",
    )
    job = QMAnnotate(str(cfg), str(tmp_path / "t"), str(tmp_path / "o"))
    assert isinstance(job.driver, GPU4PySCFDriver)
    assert job.driver.plugin.enabled is False
    assert "gpu4pyscf" not in sys.modules
