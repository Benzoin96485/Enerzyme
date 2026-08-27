"""240-job diagnostic subset, residual pairing, and zero / x-only / Stage A grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.units import Bohr, Hartree

from enerzyme.data.dc_uma_cpcm.constants import (
    DIAGNOSTIC_BASELINES,
    DIAGNOSTIC_EVAL_SLICES,
    DIAGNOSTIC_SEEDS,
    EVAL_EPSILONS,
    N_DIAG_JOBS,
    N_DIAG_TRAIN_LABELS,
    N_DIAG_VAL_LABELS,
    TRAIN_EPSILONS,
)
from enerzyme.data.dc_uma_cpcm.diagnostic import (
    DiagnosticError,
    build_residual_records,
    decide_go_nogo,
    diagnostic_job_specs,
    iter_diagnostic_labels,
    pair_residual,
    record_from_job,
    split_records,
    vacuum_source_allowed,
    write_residual_aselmdb,
    write_residual_jsonl,
)
from enerzyme.qm.orca_cpcm import prepare_diagnostic_jobs, run_job
from enerzyme.tasks.cpcm_diagnostic import (
    ToyCpcmModel,
    collate_residual_batch,
    recommend_accumulate_grad_batches,
    run_seed_grid,
    train_baseline,
)
from test_orca_cpcm import NUMBERS, POSITIONS, _ScriptedOrca, _write_basis

ROOT = Path(__file__).resolve().parents[1]
DIAG_CFG = ROOT / "enerzyme" / "config" / "dc_uma_cpcm_diagnostic.yaml"


def _geom(gid="g0", parent="p0", split="train", charge=0, n_atoms=3):
    forces = (-np.array([[0.01, 0.0, 0.0], [-0.005, 0.0, 0.0], [-0.005, 0.0, 0.0]]) * Hartree / Bohr).tolist()
    return {
        "geometry_id": gid,
        "parent_id": parent,
        "split": split,
        "roles": [split, "diagnostic"],
        "charge": charge,
        "spin": 1,
        "n_atoms": n_atoms,
        "numbers": NUMBERS,
        "positions": POSITIONS,
        "vacuum_energy_ev": -76.0 * Hartree,
        "vacuum_forces_ev_a": forces,
    }


def _label(geom, eps, role):
    return {
        "label_id": f"{geom['geometry_id']}|role={role}|eps={eps:g}",
        "geometry_id": geom["geometry_id"],
        "parent_id": geom["parent_id"],
        "split": geom["split"],
        "role": role,
        "epsilon": float(eps),
    }


def _tiny_diagnostic_manifest():
    g_train = _geom("g_train", "p_train", "train")
    g_val = _geom("g_val", "p_val", "validation")
    labels = [_label(g_train, 4.0, "train"), _label(g_val, 8.0, "validation")]
    return {
        "geometries": [g_train, g_val],
        "labels": labels,
        "diagnostic": {
            "train_parent_ids": ["p_train"],
            "validation_parent_ids": ["p_val"],
            "n_labels": 2,
            "jobs": [
                {
                    "job_id": lab["label_id"],
                    "geometry_id": lab["geometry_id"],
                    "parent_id": lab["parent_id"],
                    "epsilon": lab["epsilon"],
                    "role": lab["role"],
                    "split": lab["split"],
                }
                for lab in labels
            ],
        },
    }


def test_diagnostic_config_has_three_baselines_seeds_and_slices():
    assert N_DIAG_JOBS == 240
    assert N_DIAG_TRAIN_LABELS == 128
    assert N_DIAG_VAL_LABELS == 112
    cfg = yaml.safe_load(DIAG_CFG.read_text())
    assert cfg["build_params"]["adapter_mode"] == "stage_a"
    assert cfg["trainer"]["baselines"] == list(DIAGNOSTIC_BASELINES)
    assert cfg["trainer"]["seeds"] == list(DIAGNOSTIC_SEEDS)
    assert cfg["trainer"]["eval_slices"] == list(DIAGNOSTIC_EVAL_SLICES)
    assert DIAG_CFG.read_text().count("/home/") == 0
    assert DIAG_CFG.read_text().count("/global/") == 0
    assert "50000" not in DIAG_CFG.read_text()


def test_diagnostic_job_specs_and_held_out_epsilon():
    manifest = _tiny_diagnostic_manifest()
    specs = diagnostic_job_specs(manifest, require_n=2)
    assert [s["role"] for s in specs] == ["train", "validation"]
    assert specs[0]["epsilon"] in TRAIN_EPSILONS
    assert specs[1]["epsilon"] in EVAL_EPSILONS
    assert specs[1]["epsilon"] not in TRAIN_EPSILONS
    labels = iter_diagnostic_labels(manifest)
    assert len(labels) == 2


def test_diagnostic_job_specs_reject_open_shell():
    manifest = _tiny_diagnostic_manifest()
    manifest["geometries"][0]["s_squared"] = 0.73
    with pytest.raises(DiagnosticError, match="open-shell"):
        diagnostic_job_specs(manifest, require_n=2)


def test_pair_residual_is_raw_cpcm_minus_vacuum():
    residual = pair_residual(
        energy_cpcm_ev=10.0,
        forces_cpcm_ev_a=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        energy_vac_ev=7.0,
        forces_vac_ev_a=[[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert residual["dE"] == pytest.approx(3.0)
    np.testing.assert_allclose(residual["dFa"], [[0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_vacuum_gate_blocks_omol_subtraction_when_pilot_says_recompute():
    assert vacuum_source_allowed(None)["allowed"] is True
    blocked = vacuum_source_allowed({"recommend_recompute_all_vacuum": True, "stop_expand": False})
    assert blocked["allowed"] is False
    stopped = vacuum_source_allowed({"stop_expand": True})
    assert stopped["allowed"] is False


def test_prepare_diagnostic_jobs_and_build_residuals(tmp_path):
    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    manifest = _tiny_diagnostic_manifest()
    root = tmp_path / "orca"
    array_map = prepare_diagnostic_jobs(
        manifest, root, nprocs=4, basis_src=basis, require_n=2
    )
    assert array_map["n_jobs"] == 2
    assert array_map["subset"] == "diagnostic"
    assert array_map["jobs"][0]["relpath"].startswith("jobs/00_")
    runner = _ScriptedOrca([("success_cpcm80.out", "cpcm80.engrad")])
    for item in array_map["jobs"]:
        status = run_job(root / item["relpath"], execute=runner, orca_bin="/fake/orca", skip_fd=True)
        geom = manifest["geometries"][0 if item["role"] == "train" else 1]
        status["status"] = "success"
        status["energy_ev"] = float(geom["vacuum_energy_ev"]) - (0.4 if item["role"] == "train" else 0.2)
        status["forces_ev_a"] = (np.asarray(geom["vacuum_forces_ev_a"]) + 0.01).tolist()
        status["cpcm_epsilon"] = float(item["epsilon"])
        (root / item["relpath"] / "status.json").write_text(
            json.dumps(status),
            encoding="utf-8",
        )
    records = build_residual_records(manifest, root, require_n=2, vacuum_gate=None)
    assert len(records) == 2
    split = split_records(records)
    assert len(split["train"]) == 1
    assert len(split["validation"]) == 1
    assert split["probe"] == []
    assert split["test"] == []
    assert split["train"][0]["dE"] == pytest.approx(-0.4)
    assert split["validation"][0]["held_out_epsilon"] is True
    db = tmp_path / "train.aselmdb"
    assert write_residual_aselmdb(split["train"], db) == 1
    write_residual_jsonl(records, tmp_path / "residuals.jsonl")
    assert (tmp_path / "residuals.jsonl").is_file()


def test_split_records_keeps_probe_and_test():
    rows = [
        {"role": "train", "label_id": "a"},
        {"role": "validation", "label_id": "b"},
        {"role": "probe", "label_id": "c"},
        {"role": "test", "label_id": "d"},
    ]
    split = split_records(rows)
    assert [r["label_id"] for r in split["train"]] == ["a"]
    assert [r["label_id"] for r in split["validation"]] == ["b"]
    assert [r["label_id"] for r in split["probe"]] == ["c"]
    assert [r["label_id"] for r in split["test"]] == ["d"]


def test_build_full_residual_subset(tmp_path):
    from enerzyme.qm.orca_cpcm import prepare_full_jobs

    basis = _write_basis(tmp_path / "def2-tzvpd.bas")
    geom = {
        "geometry_id": "omol25:data0000.aselmdb:1",
        "parent_id": "graph:t|iso:natural|Q0|S1",
        "split": "train",
        "roles": ["train"],
        "charge": 0,
        "spin": 1,
        "n_atoms": 3,
        "n_basis": 58,
        "composition": "H2O",
        "numbers": [8, 1, 1],
        "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        "vacuum_energy_ev": -76.0,
        "vacuum_forces_ev_a": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "s_squared": 0.0,
    }
    label = {
        "label_id": "omol25:data0000.aselmdb:1|role=train|eps=80",
        "geometry_id": geom["geometry_id"],
        "parent_id": geom["parent_id"],
        "role": "train",
        "epsilon": 80.0,
    }
    manifest = {"geometries": [geom], "labels": [label]}
    root = tmp_path / "orca_full"
    array_map = prepare_full_jobs(manifest, root, nprocs=4, basis_src=basis, require_n=1)
    assert array_map["n_jobs"] == 1
    item = array_map["jobs"][0]
    status = {
        "status": "success",
        "energy_ev": -76.5,
        "forces_ev_a": [[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
        "cpcm_epsilon": 80.0,
    }
    (root / item["relpath"] / "status.json").write_text(json.dumps(status), encoding="utf-8")
    records = build_residual_records(
        manifest, root, require_n=1, vacuum_gate=None, subset="full"
    )
    assert len(records) == 1
    assert records[0]["dE"] == pytest.approx(-0.5)
    assert records[0]["role"] == "train"


def _synthetic_records(n_train=8, n_val=6):
    records = []
    rng = np.random.default_rng(0)
    train_eps = list(TRAIN_EPSILONS)
    val_eps = list(EVAL_EPSILONS)
    for i in range(n_train):
        eps = train_eps[i % len(train_eps)]
        n = 3
        pos = np.array(POSITIONS, dtype=np.float64) + 0.01 * i
        scale = np.log(eps)
        dE = -0.05 * scale
        dFa = np.zeros((n, 3))
        dFa[0, 0] = 0.02 * scale
        dFa[1, 0] = -0.01 * scale
        dFa[2, 0] = -0.01 * scale
        e_vac = -10.0 - 0.1 * i
        records.append(
            {
                "label_id": f"train-{i}",
                "geometry_id": f"gt{i // 2}",
                "parent_id": f"pt{i // 4}",
                "role": "train",
                "split": "train",
                "charge": 0 if i % 2 == 0 else 1,
                "spin": 1,
                "n_atoms": n,
                "size_bin": "6-15",
                "dielectric": float(eps),
                "train_epsilon": True,
                "held_out_epsilon": False,
                "numbers": NUMBERS,
                "positions": pos.tolist(),
                "dE": float(dE),
                "dFa": dFa.tolist(),
                "E_cpcm": float(e_vac + dE),
                "F_cpcm": dFa.tolist(),
                "E_vac": e_vac,
                "F_vac": np.zeros((n, 3)).tolist(),
            }
        )
    for i in range(n_val):
        eps = val_eps[i % len(val_eps)]
        n = 3
        pos = np.array(POSITIONS, dtype=np.float64) + 0.2 + 0.01 * i
        scale = np.log(eps)
        dE = -0.05 * scale
        dFa = np.zeros((n, 3))
        dFa[0, 0] = 0.02 * scale
        noise = 0.001 * rng.normal()
        records.append(
            {
                "label_id": f"val-{i}",
                "geometry_id": f"gv{i // 2}",
                "parent_id": f"pv{i // 3}",
                "role": "validation",
                "split": "validation",
                "charge": -1 if i % 3 == 2 else 0,
                "spin": 1,
                "n_atoms": n,
                "size_bin": "6-15",
                "dielectric": float(eps),
                "train_epsilon": eps in TRAIN_EPSILONS,
                "held_out_epsilon": eps not in TRAIN_EPSILONS,
                "numbers": NUMBERS,
                "positions": pos.tolist(),
                "dE": float(dE + noise),
                "dFa": dFa.tolist(),
                "E_cpcm": float(-11.0 + dE),
                "F_cpcm": dFa.tolist(),
                "E_vac": -11.0,
                "F_vac": np.zeros((n, 3)).tolist(),
            }
        )
    return records


def torch_device():
    import torch

    return torch.device("cpu")


def test_collate_and_zero_baseline_is_identically_zero():
    recs = _synthetic_records(n_train=2, n_val=2)
    batch = collate_residual_batch(recs[:2], torch_device())
    assert batch["dielectric"].shape == (2,)
    assert batch["Ra"].requires_grad
    model = ToyCpcmModel(adapter_mode="zero")
    model.eval()
    out = model.get_output(
        Ra=batch["Ra"],
        Za=batch["Za"],
        batch_seg=batch["batch_seg"],
        dielectric=batch["dielectric"],
        Q=batch["Q"],
        S=batch["S"],
    )
    assert float(out["dE"].abs().sum()) == 0.0
    assert float(out["dFa"].abs().sum()) == 0.0


def test_three_baselines_three_seeds_and_three_slices(tmp_path):
    records = _synthetic_records(n_train=8, n_val=6)
    payload = run_seed_grid(
        lambda baseline, seed: ToyCpcmModel(adapter_mode=baseline),
        records,
        out_root=tmp_path / "runs",
        max_epochs=12,
        patience=6,
        batch_size=2,
        max_steps=40,
        accumulate_grad_batches=1,
        lr=0.05,
    )
    assert len(payload["runs"]) == 9
    baselines = {run["baseline"] for run in payload["runs"]}
    seeds = {run["seed"] for run in payload["runs"]}
    assert baselines == set(DIAGNOSTIC_BASELINES)
    assert seeds == set(DIAGNOSTIC_SEEDS)
    for run in payload["runs"]:
        for name in DIAGNOSTIC_EVAL_SLICES:
            assert name in run["slices"]
            assert run["slices"][name]["n"] >= 0
    zero = next(run for run in payload["runs"] if run["baseline"] == "zero")
    assert zero["n_steps"] == 0
    gate = payload["gate"]
    assert gate["unseen_claim"] == "adapter-unseen parent"
    assert "stop_remaining_orca" in gate
    assert set(gate["slices"]) == set(DIAGNOSTIC_EVAL_SLICES)


def test_go_nogo_geometry_memorization_and_force_degradation():
    def _run(baseline, seed, dE, dF, fa):
        return {
            "baseline": baseline,
            "seed": seed,
            "slices": {
                "train_epsilon": {"dE_mae": dE, "dFa_mae": dF, "Fa_mae": fa, "joint": dE + dF},
                "held_out_epsilon": {"dE_mae": dE, "dFa_mae": dF, "Fa_mae": fa, "joint": dE + dF},
                "held_out_parent": {"dE_mae": dE, "dFa_mae": dF, "Fa_mae": fa, "joint": dE + dF},
            },
        }

    runs = []
    for seed in DIAGNOSTIC_SEEDS:
        runs.append(_run("zero", seed, 0.20, 0.10, 0.10))
        runs.append(_run("x_only", seed, 0.18, 0.09, 0.12))
        runs.append(_run("stage_a", seed, 0.19, 0.095, 0.20))
    gate = decide_go_nogo(runs)
    assert gate["go"] is False
    assert gate["stop_remaining_orca"] is True
    assert (
        "stage_a_geometry_memorization_held_out_epsilon" in gate["flags"]
        or "stage_a_total_force_degradation" in gate["flags"]
    )

    good = []
    for seed in DIAGNOSTIC_SEEDS:
        good.append(_run("zero", seed, 0.40, 0.30, 0.30))
        good.append(_run("x_only", seed, 0.25, 0.20, 0.22))
        good.append(_run("stage_a", seed, 0.08, 0.06, 0.07))
    ok = decide_go_nogo(good)
    assert ok["go"] is True
    assert ok["stop_remaining_orca"] is False
    assert ok["flags"] == []


def test_recommend_accum_does_not_assume_batch_32():
    assert recommend_accumulate_grad_batches(None) == 1
    rec = recommend_accumulate_grad_batches(8.0, device_mem_gb=40.0, microbatch=1)
    assert rec >= 1
    assert rec <= 8


def test_record_from_job_uses_geometry_vacuum():
    geom = _geom()
    spec = {"job_id": "x", "epsilon": 16.0, "role": "train"}
    status = {
        "status": "success",
        "energy_ev": float(geom["vacuum_energy_ev"]) - 0.5,
        "forces_ev_a": (np.asarray(geom["vacuum_forces_ev_a"]) + 0.02).tolist(),
    }
    rec = record_from_job(spec, status, geom)
    assert rec["dE"] == pytest.approx(-0.5)
    assert rec["train_epsilon"] is True
    np.testing.assert_allclose(np.asarray(rec["dFa"]), 0.02)


def test_train_zero_skips_optimizer(tmp_path):
    records = _synthetic_records(n_train=4, n_val=3)
    split = split_records(records)
    report = train_baseline(
        ToyCpcmModel(adapter_mode="zero"),
        split["train"],
        split["validation"],
        baseline="zero",
        seed=DIAGNOSTIC_SEEDS[0],
        out_dir=tmp_path / "zero",
        max_epochs=2,
    )
    assert report["trained"] is False
    assert report["n_steps"] == 0
    assert report["slices"]["held_out_parent"]["n"] >= 1
