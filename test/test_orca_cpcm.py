"""ORCA 6.1.1 C-PCM input, fixture parsing, retry, and resume."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from enerzyme.qm.orca_cpcm import (
    FORBIDDEN_SOLVENT_TOKENS,
    MAX_ATTEMPTS,
    ORCA_FUNCTIONAL,
    ORCA_RETRY_MAXITER,
    OrcaCpcmError,
    classify_retry,
    compare_vacuum_to_omol,
    finite_difference_force,
    parse_engrad,
    parse_orca_output,
    prepare_pilot_jobs,
    render_orca_input,
    run_job,
    summarize_pilot,
    vacuum_gate_from_comparisons,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orca_cpcm"
NUMBERS = [8, 1, 1]
POSITIONS = [
    [0.0, 0.0, 0.0],
    [0.957, 0.0, 0.0],
    [-0.239, 0.927, 0.0],
]


def _input(**kwargs):
    payload = dict(
        numbers=NUMBERS,
        positions=POSITIONS,
        charge=0,
        multiplicity=1,
        epsilon="vacuum",
        nprocs=4,
        maxcore_mb=1000,
        nbo=False,
    )
    payload.update(kwargs)
    return render_orca_input(**payload)


def test_vacuum_input_has_omol_keywords_and_no_cpcm():
    text = _input(epsilon="vacuum")
    assert f"! {ORCA_FUNCTIONAL} def2-TZVPD" in text
    assert "EnGrad" in text
    assert "RIJCOSX" in text
    assert "DEFGRID3" in text
    assert "ALLPOP" in text
    assert "NormalConv" in text
    assert "maxiter 300" in text
    assert "%cpcm" not in text.lower()
    for token in FORBIDDEN_SOLVENT_TOKENS:
        assert token not in text.lower()
    assert "* xyz 0 1" in text


def test_cpcm_block_only_adds_epsilon_for_arbitrary_dielectrics():
    for eps in (1.5, 7.3, 80.0):
        text = _input(epsilon=eps)
        assert "%cpcm" in text
        assert f"epsilon {eps:g}" in text
        assert "smd" not in text.lower()
        assert "draco" not in text.lower()
        assert "cds" not in text.lower()
        assert "EnGrad" in text


def test_forbidden_solvent_tokens_rejected():
    from enerzyme.qm.orca_cpcm import _assert_no_forbidden_solvent

    with pytest.raises(OrcaCpcmError):
        _assert_no_forbidden_solvent("! wB97M-V CPCM(water)\n")
    with pytest.raises(OrcaCpcmError):
        _assert_no_forbidden_solvent("%cpcm\n  smd true\nend\n")


def test_parse_vacuum_fixture_energy_forces_lowdin_dipole_scf():
    out = (FIXTURES / "success_vacuum.out").read_text()
    engrad = (FIXTURES / "success.engrad").read_text()
    parsed = parse_orca_output(out, engrad)
    assert parsed.orca_version == "6.1.1"
    assert parsed.terminated_normally
    assert parsed.scf_converged is True
    assert parsed.scf_cycles == 12
    assert parsed.energy_ha == pytest.approx(-76.123456789012)
    assert parsed.lowdin_charges == pytest.approx([-0.512345, 0.256172, 0.256173])
    assert parsed.dipole_au == pytest.approx([0.1, 0.0, 0.0])
    assert parsed.cpcm_epsilon is None
    assert parsed.has_cartesian_gradient
    assert not parsed.numerical_gradient
    energy, grad = parse_engrad(engrad)
    assert energy == pytest.approx(parsed.energy_ha)
    forces = -grad * Hartree / Bohr
    np.testing.assert_allclose(parsed.forces_ev_a, forces, rtol=1e-12, atol=0.0)


def test_parse_cpcm_fixture_records_epsilon_and_analytic_gradient():
    out = (FIXTURES / "success_cpcm80.out").read_text()
    engrad = (FIXTURES / "cpcm80.engrad").read_text()
    parsed = parse_orca_output(out, engrad)
    assert parsed.cpcm_epsilon == pytest.approx(80.0)
    assert "GAUSSIAN" in (parsed.cpcm_surface or "").upper()
    assert parsed.cpcm_feps and parsed.cpcm_feps.upper() == "CPCM"
    assert parsed.cpcm_smd is False
    assert parsed.cpcm_draco is False
    assert parsed.cpcm_gradient_mentioned
    assert parsed.has_cartesian_gradient
    assert not parsed.numerical_gradient


def test_parse_scf_failure_is_retryable():
    parsed = parse_orca_output((FIXTURES / "fail_scf.out").read_text())
    assert parsed.scf_converged is False
    assert parsed.retryable
    assert not parsed.terminated_normally


def test_parse_mpi_segfault_in_propint_is_retryable():
    text = (
        "ORCA finished by error termination in PROPINT\n"
        "mpirun noticed that process rank 11 exited on signal 11 (Segmentation fault).\n"
        "THE SCF HAS CONVERGED\n"
    )
    parsed = parse_orca_output(text)
    assert parsed.infra_failure
    assert parsed.retryable
    assert classify_retry(parsed) == "InfraRetry"


def test_parse_startup_abort_is_retryable():
    text = (
        "ORCA finished by error termination in Startup\n"
        "[nid004482:1469418] Signal: Segmentation fault (11)\n"
    )
    parsed = parse_orca_output(text)
    assert parsed.infra_failure
    assert parsed.retryable
    assert classify_retry(parsed) == "InfraRetry"


def test_banner_smd_draco_mentions_are_not_enabled():
    banner = (
        "   Dagmar Lenk            : GEPOL surface, SMD, ORCA-2-JSON\n"
        "   DOCKER, DRACO, updates on ESD\n"
    )
    parsed = parse_orca_output(banner + (FIXTURES / "success_cpcm80.out").read_text())
    assert parsed.cpcm_smd is False
    assert parsed.cpcm_draco is False
    abort = parse_orca_output("ORCA finished by error termination in Startup\n")
    assert abort.terminated_normally is False
    assert abort.error


def _tiny_manifest(epsilon="vacuum"):
    geom_id = "omol25:data0000.aselmdb:1"
    geom = {
        "geometry_id": geom_id,
        "parent_id": "graph:test|iso:natural|Q0|S1",
        "split": "train",
        "roles": ["pilot"],
        "charge": 0,
        "spin": 1,
        "n_atoms": 3,
        "n_basis": 58,
        "composition": "H2O",
        "numbers": NUMBERS,
        "positions": POSITIONS,
        "vacuum_energy_ev": -76.123456789012 * Hartree,
        "vacuum_forces_ev_a": (-np.array([[0.01, 0.0, 0.0], [-0.005, 0.0, 0.0], [-0.005, 0.0, 0.0]]) * Hartree / Bohr).tolist(),
    }
    return {
        "geometries": [geom],
        "pilot": {
            "geometry_ids": [geom_id],
            "jobs": [{"job_id": f"{geom_id}|pilot|eps={epsilon}", "geometry_id": geom_id, "epsilon": epsilon}],
        },
    }


def _write_basis(path: Path) -> Path:
    path.write_text("$DATA\nOXYGEN\n\n", encoding="utf-8")
    return path


class _ScriptedOrca:
    def __init__(self, responses: list[tuple[str, str | None]]):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, job_dir: Path, orca_bin: str) -> subprocess.CompletedProcess:
        out_name, engrad_name = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        (job_dir / "orca.out").write_text((FIXTURES / out_name).read_text(), encoding="utf-8")
        if engrad_name:
            (job_dir / "orca.engrad").write_text((FIXTURES / engrad_name).read_text(), encoding="utf-8")
        rc = 0 if "success" in out_name else 1
        return SimpleNamespace(returncode=rc)


def test_prepare_and_resume_skip_successful_job(tmp_path):
    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    root = tmp_path / "orca"
    array_map = prepare_pilot_jobs(
        _tiny_manifest("vacuum"),
        root,
        nprocs=4,
        basis_src=basis,
        require_n=1,
    )
    assert array_map["n_jobs"] == 1
    job_dir = root / array_map["jobs"][0]["relpath"]
    inp = (job_dir / "orca.inp").read_text()
    assert "%cpcm" not in inp.lower()
    assert (job_dir / "input.sha256").is_file()
    runner = _ScriptedOrca([("success_vacuum.out", "success.engrad")])
    status = run_job(job_dir, execute=runner, orca_bin="/fake/orca", skip_fd=True)
    assert status["status"] == "success"
    assert runner.calls == 1
    again = run_job(job_dir, execute=runner, orca_bin="/fake/orca", skip_fd=True)
    assert again["status"] == "success"
    assert runner.calls == 1


def test_retry_uses_slowconv_maxiter_600_and_fresh_guess(tmp_path):
    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    root = tmp_path / "orca"
    array_map = prepare_pilot_jobs(
        _tiny_manifest(1.5),
        root,
        nprocs=4,
        basis_src=basis,
        require_n=1,
    )
    job_dir = root / array_map["jobs"][0]["relpath"]
    first = (job_dir / "orca.inp").read_text()
    assert "epsilon 1.5" in first
    assert "NormalConv" in first
    runner = _ScriptedOrca(
        [
            ("fail_scf.out", None),
            ("success_cpcm80.out", "cpcm80.engrad"),
        ]
    )
    status = run_job(job_dir, execute=runner, orca_bin="/fake/orca", skip_fd=True)
    assert runner.calls == 2
    assert status["attempt"] == MAX_ATTEMPTS
    retry_inp = (job_dir / "orca.inp").read_text()
    assert "SlowConv" in retry_inp
    assert f"maxiter {ORCA_RETRY_MAXITER}" in retry_inp
    assert "Guess PModel" in retry_inp
    assert "wB97M-V" in retry_inp
    assert "DEFGRID3" in retry_inp
    assert "epsilon 1.5" in retry_inp
    assert "smd" not in retry_inp.lower()
    assert (job_dir / "attempts" / "01" / "orca.out").is_file()
    # Fixture epsilon is 80, requested is 1.5, so the second pass should fail the epsilon gate.
    assert status["status"] == "failed"
    assert "epsilon" in (status["error"] or "").lower()


def test_retry_success_when_fixture_epsilon_matches(tmp_path):
    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    root = tmp_path / "orca"
    array_map = prepare_pilot_jobs(
        _tiny_manifest(80.0),
        root,
        nprocs=4,
        basis_src=basis,
        require_n=1,
    )
    job_dir = root / array_map["jobs"][0]["relpath"]
    runner = _ScriptedOrca(
        [
            ("fail_scf.out", None),
            ("success_cpcm80.out", "cpcm80.engrad"),
        ]
    )
    status = run_job(job_dir, execute=runner, orca_bin="/fake/orca", skip_fd=True)
    assert status["status"] == "success"
    assert status["cpcm_epsilon"] == pytest.approx(80.0)
    assert status["analytic_gradient"] is True
    assert status["lowdin_charges"]


def test_finite_difference_formula_and_vacuum_omol_compare():
    delta = 1.0e-3
    analytic = 1.234
    e0 = -76.0
    e_plus = e0 - (analytic * delta) / Hartree
    e_minus = e0 + (analytic * delta) / Hartree
    assert finite_difference_force(e_plus, e_minus, delta) == pytest.approx(analytic, rel=1e-9, abs=1e-10)

    meta = {
        "epsilon": "vacuum",
        "geometry_id": "g1",
        "vacuum_energy_ev": 1.0,
        "vacuum_forces_ev_a": [[0.1, 0.0, 0.0]],
    }
    close = compare_vacuum_to_omol(
        {"energy_ev": 1.0 + 1e-5, "forces_ev_a": [[0.1, 0.0, 0.0]]},
        meta,
    )
    assert close["above_recompute_noise"] is False
    shifted = compare_vacuum_to_omol(
        {"energy_ev": 1.0 + 1.0, "forces_ev_a": [[0.1, 0.0, 0.0]]},
        meta,
    )
    assert shifted["possible_state_change"] is True
    assert shifted["above_recompute_noise"] is True
    oss = compare_vacuum_to_omol(
        {"energy_ev": 1.0 + 0.08, "forces_ev_a": [[0.7, 0.0, 0.0]]},
        {**meta, "s_squared": 0.734},
    )
    assert oss["closed_shell"] is False
    closed = compare_vacuum_to_omol(
        {"energy_ev": 1.0 + 1e-5, "forces_ev_a": [[0.1, 0.0, 0.0]]},
        {**meta, "s_squared": 0.0},
    )
    assert closed["closed_shell"] is True
    gate = vacuum_gate_from_comparisons([oss, closed])
    assert gate["recommend_recompute_all_vacuum"] is False
    assert len(gate["vacuum_vs_omol_open_shell"]) == 1


def test_summarize_resume_and_stop_expand(tmp_path):
    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    root = tmp_path / "orca"
    array_map = prepare_pilot_jobs(
        _tiny_manifest("vacuum"),
        root,
        nprocs=4,
        basis_src=basis,
        require_n=1,
    )
    job_dir = root / array_map["jobs"][0]["relpath"]
    status = run_job(
        job_dir,
        execute=_ScriptedOrca([("success_vacuum.out", "success.engrad")]),
        orca_bin="/fake/orca",
        skip_fd=True,
    )
    report = summarize_pilot(root, report_path=tmp_path / "report.json", write_default_report=False)
    assert report["n_jobs"] == 1
    assert report["n_success"] == 1
    assert report["first_pass_rate"] == 1.0
    assert report["stop_expand"] is False
    assert status["orca_version"] == "6.1.1"
    assert (tmp_path / "report.json").is_file()
