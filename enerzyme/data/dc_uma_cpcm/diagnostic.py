"""240-job diagnostic subset, residual labels, and three-slice evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from .constants import (
    ASPIRATIONAL_ENERGY_MAE_EV,
    ASPIRATIONAL_FORCE_MAE_EV_A,
    STAGE_B_CHARGED_ENERGY_IMPROVE,
    STAGE_B_FORCE_DEGRADE_MAX,
    DEFAULT_EXPERIMENTS,
    DIAGNOSTIC_BASELINES,
    DIAGNOSTIC_EVAL_SLICES,
    DIAGNOSTIC_SEEDS,
    EVAL_EPSILONS,
    N_DIAG_JOBS,
    N_DIAG_TRAIN_LABELS,
    N_DIAG_VAL_LABELS,
    SCHEMA_VERSION,
    TRAIN_EPSILONS,
    UNSEEN_CLAIM,
)
from .index import is_closed_shell
from .sample import size_bin

DEFAULT_DIAGNOSTIC_ORCA = Path(DEFAULT_EXPERIMENTS) / "orca" / "diagnostic"
DEFAULT_DIAGNOSTIC_DATA = Path(DEFAULT_EXPERIMENTS) / "manifests" / "diagnostic"
DEFAULT_DIAGNOSTIC_RUNS = Path(DEFAULT_EXPERIMENTS) / "runs" / "diagnostic"
DEFAULT_FULL_ORCA = Path(DEFAULT_EXPERIMENTS) / "orca" / "full"
DEFAULT_FULL_DATA = Path(DEFAULT_EXPERIMENTS) / "manifests" / "full"
DEFAULT_FULL_RUNS = Path(DEFAULT_EXPERIMENTS) / "runs" / "full"
HELD_OUT_EPSILONS = tuple(eps for eps in EVAL_EPSILONS if eps not in TRAIN_EPSILONS)


class DiagnosticError(RuntimeError):
    """Invalid diagnostic subset, residual pairing, or go/no-go inputs."""


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise DiagnosticError(f"{name} must be finite and non-empty")
    return arr


def diagnostic_parent_ids(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    diag = manifest.get("diagnostic") or {}
    train = set(diag.get("train_parent_ids") or [])
    val = set(diag.get("validation_parent_ids") or [])
    return train, val


def iter_diagnostic_labels(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    train_ids, val_ids = diagnostic_parent_ids(manifest)
    parent_ids = train_ids | val_ids
    labels = []
    for lab in manifest.get("labels") or []:
        if lab.get("parent_id") not in parent_ids:
            continue
        if lab.get("role") not in ("train", "validation"):
            continue
        labels.append(dict(lab))
    jobs = (manifest.get("diagnostic") or {}).get("jobs")
    if jobs:
        by_id = {lab["label_id"]: lab for lab in labels}
        ordered = []
        for job in jobs:
            lab = by_id.get(job.get("job_id"))
            if lab is None:
                continue
            merged = dict(lab)
            merged.update({k: job[k] for k in job if k not in merged})
            ordered.append(merged)
        if ordered:
            labels = ordered
    return labels


def geometry_table(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {geom["geometry_id"]: geom for geom in manifest["geometries"]}


def diagnostic_job_specs(
    manifest: Mapping[str, Any],
    *,
    require_n: int | None = N_DIAG_JOBS,
) -> list[dict[str, Any]]:
    geoms = geometry_table(manifest)
    specs = []
    for lab in iter_diagnostic_labels(manifest):
        gid = lab["geometry_id"]
        if gid not in geoms:
            raise DiagnosticError(f"diagnostic geometry {gid} missing from manifest")
        spec = dict(lab)
        spec["job_id"] = lab.get("job_id") or lab["label_id"]
        spec["geometry"] = geoms[gid]
        if not is_closed_shell(geoms[gid].get("s_squared")):
            raise DiagnosticError(
                f"diagnostic geometry {gid} is open-shell (S^2={geoms[gid].get('s_squared')}); "
                "this stage is closed-shell only"
            )
        specs.append(spec)
    if require_n is not None and len(specs) != require_n:
        raise DiagnosticError(f"expected {require_n} diagnostic jobs, found {len(specs)}")
    return specs


def vacuum_source_allowed(pilot_report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Refuse OMol-historical vacuum subtraction when the pilot says to recompute."""
    if not pilot_report:
        return {
            "allowed": True,
            "source": "omol_historical",
            "reason": "no pilot report provided; tests and dry-runs may pair OMol vacuum",
        }
    recompute = bool(pilot_report.get("recommend_recompute_all_vacuum"))
    stop = bool(pilot_report.get("stop_expand"))
    if stop:
        return {
            "allowed": False,
            "source": None,
            "reason": "pilot stop_expand is true; do not label or train the 240-job set",
        }
    if recompute:
        return {
            "allowed": False,
            "source": "fresh_orca_vacuum",
            "reason": (
                "pilot found vacuum bias or electronic-state switching; "
                "do not subtract OMol historical vacuum from C-PCM"
            ),
        }
    return {
        "allowed": True,
        "source": "omol_historical",
        "reason": "pilot vacuum comparison is within recompute noise",
    }


def pair_residual(
    *,
    energy_cpcm_ev: float,
    forces_cpcm_ev_a: Any,
    energy_vac_ev: float,
    forces_vac_ev_a: Any,
) -> dict[str, Any]:
    e_cpcm = float(energy_cpcm_ev)
    e_vac = float(energy_vac_ev)
    f_cpcm = _as_float_array(forces_cpcm_ev_a, name="F_cpcm")
    f_vac = _as_float_array(forces_vac_ev_a, name="F_vac")
    if f_cpcm.shape != f_vac.shape:
        raise DiagnosticError(f"force shape {f_cpcm.shape} != vacuum {f_vac.shape}")
    if not np.isfinite(e_cpcm) or not np.isfinite(e_vac):
        raise DiagnosticError("C-PCM and vacuum energies must be finite")
    dE = e_cpcm - e_vac
    dF = f_cpcm - f_vac
    return {
        "E_cpcm": e_cpcm,
        "F_cpcm": np.asarray(f_cpcm, dtype=np.float64).copy(),
        "E_vac": e_vac,
        "F_vac": np.asarray(f_vac, dtype=np.float64).copy(),
        "dE": float(dE),
        "dFa": dF,
    }


def record_from_job(
    spec: Mapping[str, Any],
    status: Mapping[str, Any],
    geom: Mapping[str, Any],
) -> dict[str, Any]:
    if status.get("status") != "success":
        raise DiagnosticError(f"job {spec.get('job_id')} is not success")
    if not is_closed_shell(geom.get("s_squared")):
        raise DiagnosticError(
            f"job {spec.get('job_id')} geometry {geom.get('geometry_id')} is open-shell; "
            "this stage is closed-shell only"
        )
    residual = pair_residual(
        energy_cpcm_ev=status["energy_ev"],
        forces_cpcm_ev_a=status["forces_ev_a"],
        energy_vac_ev=geom["vacuum_energy_ev"],
        forces_vac_ev_a=geom["vacuum_forces_ev_a"],
    )
    eps = float(spec["epsilon"])
    n_atoms = int(geom["n_atoms"])
    return {
        "label_id": spec.get("job_id") or spec.get("label_id"),
        "geometry_id": geom["geometry_id"],
        "parent_id": geom["parent_id"],
        "role": spec.get("role") or ("train" if geom.get("split") == "train" else "validation"),
        "split": geom.get("split"),
        "charge": int(geom["charge"]),
        "spin": int(geom["spin"]),
        "n_atoms": n_atoms,
        "size_bin": size_bin(n_atoms),
        "dielectric": eps,
        "train_epsilon": eps in TRAIN_EPSILONS,
        "held_out_epsilon": eps not in TRAIN_EPSILONS,
        "numbers": list(geom["numbers"]),
        "positions": np.asarray(geom["positions"], dtype=np.float64).tolist(),
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in residual.items()},
    }


def _job_specs_for_subset(
    manifest: Mapping[str, Any],
    subset: str,
) -> list[dict[str, Any]]:
    subset = str(subset).strip().lower()
    if subset == "diagnostic":
        return diagnostic_job_specs(manifest, require_n=None)
    if subset == "full":
        from enerzyme.qm.orca_cpcm import full_label_job_specs

        return full_label_job_specs(manifest, require_n=None)
    raise DiagnosticError(f"unknown residual subset {subset!r}; expected diagnostic or full")


def build_residual_records(
    manifest: Mapping[str, Any],
    orca_root: Path | str,
    *,
    require_n: int | None = N_DIAG_JOBS,
    require_success: bool = True,
    vacuum_gate: Mapping[str, Any] | None = None,
    subset: str = "diagnostic",
) -> list[dict[str, Any]]:
    from enerzyme.qm.orca_cpcm import ARRAY_MAP_NAME, STATUS_NAME, load_json

    gate = vacuum_source_allowed(vacuum_gate)
    if not gate["allowed"]:
        raise DiagnosticError(gate["reason"])
    root = Path(orca_root)
    array_map = load_json(root / ARRAY_MAP_NAME)
    specs = {item["job_id"]: item for item in _job_specs_for_subset(manifest, subset)}
    records = []
    missing = []
    failed = []
    for item in array_map["jobs"]:
        job_id = item["job_id"]
        spec = specs.get(job_id)
        if spec is None:
            continue
        status_path = root / item["relpath"] / STATUS_NAME
        if not status_path.is_file():
            missing.append(job_id)
            continue
        status = load_json(status_path)
        if status.get("status") != "success":
            failed.append(job_id)
            continue
        records.append(record_from_job(spec, status, spec["geometry"]))
    if require_success and (missing or failed):
        raise DiagnosticError(
            f"{subset} residuals incomplete: missing={len(missing)} failed={len(failed)}"
        )
    if require_n is not None and len(records) != require_n:
        raise DiagnosticError(f"expected {require_n} residual records, found {len(records)}")
    return records


def split_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {"train": [], "validation": [], "probe": [], "test": []}
    for rec in records:
        role = rec.get("role")
        if role in out:
            out[role].append(dict(rec))
    return out


def write_residual_aselmdb(records: Sequence[Mapping[str, Any]], db_path: Path | str) -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    n = 0
    with connect(str(db_path)) as db:
        for index, rec in enumerate(records):
            atoms = Atoms(
                numbers=rec["numbers"],
                positions=np.asarray(rec["positions"], dtype=np.float64),
            )
            dFa = np.asarray(rec["dFa"], dtype=np.float64)
            atoms.calc = SinglePointCalculator(atoms, energy=float(rec["dE"]), forces=dFa)
            atoms.info.update(
                {
                    "charge": int(rec["charge"]),
                    "spin": int(rec["spin"]),
                    "dielectric": float(rec["dielectric"]),
                    "name": rec.get("label_id"),
                }
            )
            data = {
                "charge": int(rec["charge"]),
                "spin": int(rec["spin"]),
                "index": index,
                "dielectric": float(rec["dielectric"]),
                "dE": float(rec["dE"]),
                "dFa": dFa.tolist(),
                "E_cpcm": float(rec["E_cpcm"]),
                "F_cpcm": np.asarray(rec["F_cpcm"], dtype=np.float64).tolist(),
                "E_vac": float(rec["E_vac"]),
                "F_vac": np.asarray(rec["F_vac"], dtype=np.float64).tolist(),
                "label_id": rec.get("label_id"),
                "geometry_id": rec.get("geometry_id"),
                "parent_id": rec.get("parent_id"),
                "role": rec.get("role"),
                "split": rec.get("split"),
                "size_bin": rec.get("size_bin"),
                "train_epsilon": bool(rec.get("train_epsilon")),
                "held_out_epsilon": bool(rec.get("held_out_epsilon")),
            }
            db.write(atoms, data=data, index=index)
            n += 1
    return n


def write_residual_jsonl(records: Sequence[Mapping[str, Any]], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return len(records)


def load_residual_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def slice_name_for(record: Mapping[str, Any]) -> dict[str, bool]:
    role = record.get("role")
    return {
        "train_epsilon": bool(record.get("train_epsilon")) and role == "validation",
        "held_out_epsilon": bool(record.get("held_out_epsilon")) and role == "validation",
        "held_out_parent": role == "validation",
    }


def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def metrics_from_predictions(
    records: Sequence[Mapping[str, Any]],
    pred_dE: Sequence[float],
    pred_dFa: Sequence[Any],
    *,
    pred_E: Sequence[float] | None = None,
    pred_Fa: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if len(records) != len(pred_dE) or len(records) != len(pred_dFa):
        raise DiagnosticError("prediction length does not match records")
    rows = []
    for i, rec in enumerate(records):
        n = int(rec["n_atoms"])
        dE_t = float(rec["dE"])
        dE_p = float(pred_dE[i])
        dF_t = np.asarray(rec["dFa"], dtype=np.float64).reshape(-1)
        dF_p = np.asarray(pred_dFa[i], dtype=np.float64).reshape(-1)
        row = {
            "label_id": rec.get("label_id"),
            "parent_id": rec.get("parent_id"),
            "geometry_id": rec.get("geometry_id"),
            "role": rec.get("role"),
            "charge": int(rec["charge"]),
            "size_bin": rec.get("size_bin") or size_bin(n),
            "dielectric": float(rec["dielectric"]),
            "n_atoms": n,
            "train_epsilon": bool(rec.get("train_epsilon")),
            "held_out_epsilon": bool(rec.get("held_out_epsilon")),
            "dE_true": dE_t,
            "dE_pred": dE_p,
            "dE_err": dE_p - dE_t,
            "dE_err_per_atom": (dE_p - dE_t) / max(n, 1),
            "dFa_mae": _mae(dF_p, dF_t),
            "dFa_rmse": _rmse(dF_p, dF_t),
        }
        if pred_E is not None and pred_Fa is not None:
            e_t = float(rec["E_cpcm"])
            e_p = float(pred_E[i])
            f_t = np.asarray(rec["F_cpcm"], dtype=np.float64).reshape(-1)
            f_p = np.asarray(pred_Fa[i], dtype=np.float64).reshape(-1)
            row["E_err"] = e_p - e_t
            row["Fa_mae"] = _mae(f_p, f_t)
            row["Fa_rmse"] = _rmse(f_p, f_t)
        rows.append(row)
    return {"n": len(rows), "rows": rows, "summary": summarize_rows(rows)}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "dE_mae": None,
            "dE_mae_per_atom": None,
            "dFa_mae": None,
            "E_mae": None,
            "Fa_mae": None,
            "joint": None,
        }
    dE = np.asarray([abs(r["dE_err"]) for r in rows], dtype=np.float64)
    dEa = np.asarray([abs(r["dE_err_per_atom"]) for r in rows], dtype=np.float64)
    dF = np.asarray([r["dFa_mae"] for r in rows], dtype=np.float64)
    out = {
        "n": len(rows),
        "dE_mae": float(dE.mean()),
        "dE_mae_per_atom": float(dEa.mean()),
        "dFa_mae": float(dF.mean()),
        "joint": float(dEa.mean() + dF.mean()),
    }
    if "E_err" in rows[0]:
        out["E_mae"] = float(np.mean([abs(r["E_err"]) for r in rows]))
        out["Fa_mae"] = float(np.mean([r["Fa_mae"] for r in rows]))
    return out


def subgroup_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for slice_name in DIAGNOSTIC_EVAL_SLICES:
        if slice_name == "train_epsilon":
            chosen = [r for r in rows if r.get("role") == "validation" and r.get("train_epsilon")]
        elif slice_name == "held_out_epsilon":
            chosen = [r for r in rows if r.get("role") == "validation" and r.get("held_out_epsilon")]
        else:
            chosen = [r for r in rows if r.get("role") == "validation"]
        tables[slice_name] = summarize_rows(chosen)

    # Bucket the split the caller evaluated (validation or test), not only
    # validation rows. Diagnostic slices above stay validation-only.
    tables["by_charge"] = {
        k: summarize_rows(v) for k, v in sorted(_bucket(list(rows), "charge").items())
    }
    tables["by_size"] = {
        k: summarize_rows(v) for k, v in sorted(_bucket(list(rows), "size_bin").items())
    }
    tables["by_epsilon"] = {
        k: summarize_rows(v) for k, v in sorted(_bucket(list(rows), "dielectric").items())
    }
    return tables


def _bucket(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for row in rows:
        out.setdefault(str(row.get(key)), []).append(row)
    return out


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 20260817,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return {"mean": float(arr.mean()), "low": lo, "high": hi, "n": int(arr.size)}


def seed_interval(values: Sequence[float]) -> dict[str, float | None]:
    arr = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not arr:
        return {"mean": None, "low": None, "high": None, "n": 0, "std": None}
    return {
        "mean": float(np.mean(arr)),
        "low": float(min(arr)),
        "high": float(max(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def _metric(run: Mapping[str, Any], slice_name: str, key: str) -> float | None:
    slices = run.get("slices") or {}
    payload = slices.get(slice_name) or {}
    value = payload.get(key)
    return None if value is None else float(value)


def decide_go_nogo(
    runs: Sequence[Mapping[str, Any]],
    *,
    aspirational_energy: float = ASPIRATIONAL_ENERGY_MAE_EV,
    aspirational_force: float = ASPIRATIONAL_FORCE_MAE_EV_A,
) -> dict[str, Any]:
    """Go/no-go from 3 baselines × 3 seeds and the three validation slices.

    Hard stops (do not submit remaining ORCA jobs):

    * Stage A does not beat zero-residual on adapter-unseen-parent residual
      energy or force (seed-mean interval includes no improvement).
    * Stage A does not beat x-only on held-out ε (geometry memorization).
    * Stage A total-force MAE is worse than zero (vacuum UMA) on held-out
      parents.
    """
    by_base: dict[str, list] = {name: [] for name in DIAGNOSTIC_BASELINES}
    for run in runs:
        by_base.setdefault(run["baseline"], []).append(run)

    def _collect(baseline: str, slice_name: str, key: str) -> dict[str, float | None]:
        vals = [_metric(run, slice_name, key) for run in by_base.get(baseline, [])]
        return seed_interval([v for v in vals if v is not None])

    comparisons = {}
    flags = []
    for slice_name in DIAGNOSTIC_EVAL_SLICES:
        for key in ("dE_mae", "dFa_mae", "Fa_mae", "joint"):
            comparisons[f"{slice_name}.{key}"] = {
                baseline: _collect(baseline, slice_name, key)
                for baseline in DIAGNOSTIC_BASELINES
            }

    def _beats(winner: str, loser: str, slice_name: str, key: str) -> bool | None:
        w = _collect(winner, slice_name, key)
        l = _collect(loser, slice_name, key)
        if w["mean"] is None or l["mean"] is None:
            return None
        # Stable gain: Stage A mean is better and its worst seed is still
        # no worse than the other baseline's mean.
        return w["mean"] < l["mean"] and (w["high"] is not None and w["high"] <= l["mean"])

    parent_e = _beats("stage_a", "zero", "held_out_parent", "dE_mae")
    parent_f = _beats("stage_a", "zero", "held_out_parent", "dFa_mae")
    held_eps_e = _beats("stage_a", "x_only", "held_out_epsilon", "dE_mae")
    held_eps_f = _beats("stage_a", "x_only", "held_out_epsilon", "dFa_mae")
    force_deg = False
    sa_fa = _collect("stage_a", "held_out_parent", "Fa_mae")
    z_fa = _collect("zero", "held_out_parent", "Fa_mae")
    if sa_fa["mean"] is not None and z_fa["mean"] is not None:
        force_deg = sa_fa["mean"] > z_fa["mean"] and (
            sa_fa["low"] is None or sa_fa["low"] > z_fa["mean"]
        )

    geometry_memorization = held_eps_e is False and held_eps_f is False
    no_parent_gain = parent_e is False or parent_f is False
    if geometry_memorization:
        flags.append("stage_a_geometry_memorization_held_out_epsilon")
    if no_parent_gain:
        flags.append("stage_a_no_stable_gain_held_out_parent")
    if force_deg:
        flags.append("stage_a_total_force_degradation")

    sa_parent = _collect("stage_a", "held_out_parent", "dE_mae")
    sa_parent_f = _collect("stage_a", "held_out_parent", "dFa_mae")
    aspirational = {
        "energy_mae_ev": aspirational_energy,
        "force_mae_ev_a": aspirational_force,
        "energy_met": sa_parent["mean"] is not None and sa_parent["mean"] <= aspirational_energy,
        "force_met": sa_parent_f["mean"] is not None and sa_parent_f["mean"] <= aspirational_force,
        "note": "aspirational targets, not a hard diagnostic gate",
    }
    go = not flags
    return {
        "schema": "dc_uma_cpcm_diagnostic_gate_v1",
        "unseen_claim": UNSEEN_CLAIM,
        "baselines": list(DIAGNOSTIC_BASELINES),
        "seeds": list(DIAGNOSTIC_SEEDS),
        "slices": list(DIAGNOSTIC_EVAL_SLICES),
        "n_runs": len(runs),
        "comparisons": comparisons,
        "stage_a_beats_zero_parent_energy": parent_e,
        "stage_a_beats_zero_parent_force": parent_f,
        "stage_a_beats_x_only_held_out_epsilon_energy": held_eps_e,
        "stage_a_beats_x_only_held_out_epsilon_force": held_eps_f,
        "total_force_degradation": force_deg,
        "geometry_memorization": geometry_memorization,
        "flags": flags,
        "go": go,
        "stop_remaining_orca": not go,
        "aspirational": aspirational,
        "notes": [
            "Validation parents are adapter-unseen parent only; they may have been in UMA pretraining.",
            "Go requires Stage A to beat zero on held-out parents and x-only on held-out ε, without total-force degradation.",
        ],
    }


def attach_lowdin_charges(
    records: Sequence[Mapping[str, Any]],
    orca_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Copy residual records and fill ``lowdin_charges`` from ORCA status when missing."""
    relpath_by_id: dict[str, str] = {}
    root = Path(orca_root) if orca_root is not None else None
    load_json = None
    status_name = "status.json"
    if root is not None:
        from enerzyme.qm.orca_cpcm import ARRAY_MAP_NAME, STATUS_NAME, load_json as _load_json

        load_json = _load_json
        status_name = STATUS_NAME
        map_path = root / ARRAY_MAP_NAME
        if map_path.is_file():
            array_map = load_json(map_path)
            for job in array_map.get("jobs") or []:
                if job.get("job_id") is not None and job.get("relpath") is not None:
                    relpath_by_id[str(job["job_id"])] = str(job["relpath"])
    out = []
    for rec in records:
        row = dict(rec)
        if row.get("lowdin_charges") is not None:
            out.append(row)
            continue
        label_id = str(row.get("label_id") or "")
        if root is not None and load_json is not None and label_id in relpath_by_id:
            status_path = root / relpath_by_id[label_id] / status_name
            if status_path.is_file():
                status = load_json(status_path)
                lowdin = status.get("lowdin_charges")
                if lowdin is not None:
                    row["lowdin_charges"] = lowdin
        out.append(row)
    return out


def _test_by_charge(run: Mapping[str, Any]) -> Mapping[str, Any]:
    slices = run.get("test_slices") or {}
    by_c = slices.get("by_charge")
    if by_c:
        return by_c
    return (run.get("slices") or {}).get("by_charge") or {}


def charged_dE_mae(run: Mapping[str, Any]) -> float | None:
    """Residual energy MAE over anion and cation test rows (n-weighted)."""
    by_c = _test_by_charge(run)
    n_tot = 0
    acc = 0.0
    for key in ("-1", "1"):
        row = by_c.get(key) or {}
        n = int(row.get("n") or 0)
        mae = row.get("dE_mae")
        if n <= 0 or mae is None:
            continue
        acc += float(n) * float(mae)
        n_tot += n
    if n_tot == 0:
        return None
    return acc / n_tot


def decide_stage_b_keep_drop(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    *,
    energy_improve: float = STAGE_B_CHARGED_ENERGY_IMPROVE,
    force_degrade: float = STAGE_B_FORCE_DEGRADE_MAX,
) -> dict[str, Any]:
    """Keep Stage B if charged test dE MAE improves ≥10% and full-test dFa MAE degrades ≤5%."""
    sa_charged = charged_dE_mae(stage_a)
    sb_charged = charged_dE_mae(stage_b)
    sa_f = (stage_a.get("test") or {}).get("dFa_mae")
    sb_f = (stage_b.get("test") or {}).get("dFa_mae")
    energy_ratio = None
    force_ratio = None
    if sa_charged and sb_charged is not None and sa_charged > 0:
        energy_ratio = float(sb_charged) / float(sa_charged)
    if sa_f and sb_f is not None and float(sa_f) > 0:
        force_ratio = float(sb_f) / float(sa_f)
    energy_ok = (
        sa_charged is not None
        and sb_charged is not None
        and float(sb_charged) <= float(sa_charged) * (1.0 - float(energy_improve))
    )
    force_ok = (
        sa_f is not None
        and sb_f is not None
        and float(sb_f) <= float(sa_f) * (1.0 + float(force_degrade))
    )
    keep = bool(energy_ok and force_ok)
    flags = []
    if not energy_ok:
        flags.append("charged_test_energy_mae_not_improved_10pct")
    if not force_ok:
        flags.append("full_test_force_mae_degraded_over_5pct")
    return {
        "schema": "dc_uma_cpcm_stage_b_keep_drop_v1",
        "keep": keep,
        "decision": "keep" if keep else "drop",
        "thresholds": {
            "charged_energy_improve": float(energy_improve),
            "force_degrade_max": float(force_degrade),
        },
        "stage_a_charged_test_dE_mae": sa_charged,
        "stage_b_charged_test_dE_mae": sb_charged,
        "charged_energy_mae_ratio": energy_ratio,
        "stage_a_test_dFa_mae": None if sa_f is None else float(sa_f),
        "stage_b_test_dFa_mae": None if sb_f is None else float(sb_f),
        "force_mae_ratio": force_ratio,
        "energy_ok": energy_ok,
        "force_ok": force_ok,
        "flags": flags,
        "next_step": (
            "Train the remaining two seeds (20260817, 20260819)."
            if keep
            else "Drop Stage B after the 1-seed gate; do not expand to 3 seeds."
        ),
    }


def expected_diagnostic_counts() -> dict[str, Any]:
    return {
        "jobs": N_DIAG_JOBS,
        "train_labels": N_DIAG_TRAIN_LABELS,
        "validation_labels": N_DIAG_VAL_LABELS,
        "seeds": len(DIAGNOSTIC_SEEDS),
        "baselines": len(DIAGNOSTIC_BASELINES),
        "eval_slices": len(DIAGNOSTIC_EVAL_SLICES),
        "schema_version": SCHEMA_VERSION,
    }
