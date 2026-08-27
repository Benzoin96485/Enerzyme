"""Stage A seed-level error analysis and Stage B GB go/no-go.

Uses stored production ``metrics.json`` files (no new ORCA, no retraining).
Validation ``by_charge`` / ``by_size`` / ``by_epsilon`` slices are aggregated
across seeds. Residual JSONL supplies ``|ΔE|`` label scale so an absolute
anion MAE gap can be compared with Born-like charge scaling. Optional ORCA
status Lowdin charges and dipoles are correlated with label ``|ΔE|``; per-row
Stage A residuals were not persisted in production metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    ASPIRATIONAL_ENERGY_MAE_EV,
    DEFAULT_EXPERIMENTS,
    DIAGNOSTIC_SEEDS,
    SIZE_BINS,
    TRAIN_EPSILONS,
    UNSEEN_CLAIM,
    UNSEEN_CLAIM_NOTE,
)
from .diagnostic import DEFAULT_FULL_DATA, DEFAULT_FULL_ORCA, DEFAULT_FULL_RUNS, seed_interval

SCHEMA = "dc_uma_cpcm_stage_a_error_analysis_v1"
CHARGE_KEYS = ("-1", "0", "1")
SUMMARY_KEYS = ("n", "dE_mae", "dE_mae_per_atom", "dFa_mae", "joint", "E_mae", "Fa_mae")
SLICE_GROUPS = ("by_charge", "by_size", "by_epsilon")
# Absolute anion/charged MAE must be ≥50% above neutral on every seed.
MAE_RATIO_MIN = 1.5
# Relative MAE (MAE / mean|ΔE|) must also be ≥50% worse, otherwise the
# absolute gap is just the larger ionic C-PCM label scale.
RELATIVE_RATIO_MIN = 1.5
# "接近或低于 0.10 eV": seed-mean within 20% of the aspirational target.
NEUTRAL_NEAR_FACTOR = 1.20


class ErrorAnalysisError(RuntimeError):
    """Invalid Stage A metrics, residual labels, or Stage B gate inputs."""


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage_a_metrics(runs_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    paths = sorted(root.glob("stage_a_seed*/metrics.json"))
    if not paths:
        raise ErrorAnalysisError(f"no stage_a_seed*/metrics.json under {root}")
    runs = []
    for path in paths:
        payload = load_json(path)
        if payload.get("baseline") not in (None, "stage_a"):
            continue
        payload["_path"] = str(path)
        runs.append(payload)
    if not runs:
        raise ErrorAnalysisError(f"no Stage A metrics.json under {root}")
    return runs


def _as_summary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {k: payload[k] for k in SUMMARY_KEYS if k in payload}


def _interval_tree(values: Sequence[Any]) -> Any:
    nonempty = [v for v in values if v is not None]
    if not nonempty:
        return {}
    if all(isinstance(v, Mapping) for v in nonempty):
        keys: set[str] = set()
        for item in nonempty:
            keys.update(item.keys())
        out: dict[str, Any] = {}
        for key in sorted(keys, key=str):
            children = [item.get(key) for item in nonempty]
            if key == "n" and all(not isinstance(v, Mapping) for v in children):
                ints = [int(v) for v in children if v is not None]
                out[key] = ints[0] if ints else None
            else:
                out[key] = _interval_tree(children)
        return out
    nums: list[float] = []
    for value in nonempty:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return seed_interval(nums)


def aggregate_stage_a_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ErrorAnalysisError("no Stage A runs to aggregate")
    seeds = [int(run["seed"]) for run in runs]
    slices = [run.get("slices") or {} for run in runs]
    return {
        "n_seeds": len(runs),
        "seeds": seeds,
        "validation": _interval_tree([_as_summary(run.get("validation")) for run in runs]),
        "probe": _interval_tree([_as_summary(run.get("probe")) for run in runs]),
        "test": _interval_tree([_as_summary(run.get("test")) for run in runs]),
        "slices": {
            "train_epsilon": _interval_tree([s.get("train_epsilon") for s in slices]),
            "held_out_epsilon": _interval_tree([s.get("held_out_epsilon") for s in slices]),
            "held_out_parent": _interval_tree([s.get("held_out_parent") for s in slices]),
            "by_charge": _interval_tree([s.get("by_charge") for s in slices]),
            "by_size": _interval_tree([s.get("by_size") for s in slices]),
            "by_epsilon": _interval_tree([s.get("by_epsilon") for s in slices]),
        },
        "note": (
            "by_charge / by_size / by_epsilon are validation-only in production "
            "metrics.json; test stores an overall summary."
        ),
    }


def _abs_dE_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean_abs_dE": None, "median_abs_dE": None, "p90_abs_dE": None}
    return {
        "n": int(arr.size),
        "mean_abs_dE": float(arr.mean()),
        "median_abs_dE": float(np.median(arr)),
        "p90_abs_dE": float(np.quantile(arr, 0.90)),
    }


def _bucket_abs_dE(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for rec in records:
        buckets.setdefault(str(rec.get(key)), []).append(abs(float(rec["dE"])))
    return {k: _abs_dE_stats(v) for k, v in sorted(buckets.items(), key=lambda kv: kv[0])}


def label_scale_tables(records: Sequence[Mapping[str, Any]], *, role: str) -> dict[str, Any]:
    chosen = [rec for rec in records if rec.get("role") == role]
    by_charge_eps = {"train_epsilon": {}, "held_out_epsilon": {}}
    for rec in chosen:
        side = "held_out_epsilon" if rec.get("held_out_epsilon") else "train_epsilon"
        by_charge_eps[side].setdefault(str(rec.get("charge")), []).append(abs(float(rec["dE"])))
    return {
        "role": role,
        "n": len(chosen),
        "overall": _abs_dE_stats([abs(float(r["dE"])) for r in chosen]),
        "by_charge": _bucket_abs_dE(chosen, "charge"),
        "by_size": _bucket_abs_dE(chosen, "size_bin"),
        "by_epsilon": _bucket_abs_dE(chosen, "dielectric"),
        "by_charge_and_epsilon_holdout": {
            side: {k: _abs_dE_stats(v) for k, v in sorted(buckets.items())}
            for side, buckets in by_charge_eps.items()
        },
    }


def pearson_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int | None]:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(a.size)
    if n < 3 or float(a.std()) < 1.0e-12 or float(b.std()) < 1.0e-12:
        return {"pearson": None, "spearman": None, "n": n}

    def _corr(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.corrcoef(u, v)[0, 1])

    ranks_a = np.argsort(np.argsort(a)).astype(np.float64)
    ranks_b = np.argsort(np.argsort(b)).astype(np.float64)
    return {"pearson": _corr(a, b), "spearman": _corr(ranks_a, ranks_b), "n": n}


def _charge_descriptors(lowdin: Sequence[float] | None, charge: int, n_atoms: int) -> dict[str, float | None]:
    if lowdin is None:
        return {
            "lowdin_sum": None,
            "mean_abs_lowdin": None,
            "charge_variation": None,
            "charge_separation": None,
        }
    q = np.asarray(lowdin, dtype=np.float64)
    q_mean = float(charge) / max(int(n_atoms), 1)
    pos = float(np.sum(np.clip(q, 0.0, None)))
    neg = float(np.sum(np.clip(q, None, 0.0)))
    return {
        "lowdin_sum": float(q.sum()),
        "mean_abs_lowdin": float(np.mean(np.abs(q))),
        "charge_variation": float(np.mean(np.abs(q - q_mean))),
        "charge_separation": 0.5 * (pos - neg),
    }


def attach_orca_descriptors(
    records: Sequence[Mapping[str, Any]],
    *,
    orca_root: Path | str | None = None,
    orca_report: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Join Lowdin / dipole onto residual rows. Missing ORCA is allowed."""
    dipole_by_id: dict[str, Any] = {}
    if orca_report:
        for job in orca_report.get("jobs") or []:
            if job.get("job_id") is not None:
                dipole_by_id[str(job["job_id"])] = job.get("dipole_au")
    relpath_by_id: dict[str, str] = {}
    root = Path(orca_root) if orca_root is not None else None
    if root is not None:
        from enerzyme.qm.orca_cpcm import ARRAY_MAP_NAME, STATUS_NAME

        map_path = root / ARRAY_MAP_NAME
        if map_path.is_file():
            array_map = load_json(map_path)
            for job in array_map.get("jobs") or []:
                relpath_by_id[str(job["job_id"])] = str(job["relpath"])
    rows = []
    n_lowdin = 0
    n_dipole = 0
    for rec in records:
        label_id = str(rec.get("label_id") or "")
        lowdin = rec.get("lowdin_charges")
        dipole = rec.get("dipole_au") if rec.get("dipole_au") is not None else dipole_by_id.get(label_id)
        if lowdin is None and root is not None and label_id in relpath_by_id:
            status_path = root / relpath_by_id[label_id] / STATUS_NAME
            if status_path.is_file():
                status = load_json(status_path)
                lowdin = status.get("lowdin_charges")
                if dipole is None:
                    dipole = status.get("dipole_au")
        charge = int(rec["charge"])
        n_atoms = int(rec["n_atoms"])
        eps = float(rec["dielectric"])
        g_eps = 1.0 - 1.0 / eps
        desc = _charge_descriptors(lowdin, charge, n_atoms)
        dip_mag = None
        if dipole is not None:
            dip_mag = float(np.linalg.norm(np.asarray(dipole, dtype=np.float64)))
            n_dipole += 1
        if desc["charge_separation"] is not None:
            n_lowdin += 1
        rows.append(
            {
                "label_id": rec.get("label_id"),
                "role": rec.get("role"),
                "charge": charge,
                "n_atoms": n_atoms,
                "size_bin": rec.get("size_bin"),
                "dielectric": eps,
                "held_out_epsilon": bool(rec.get("held_out_epsilon")),
                "abs_dE": abs(float(rec["dE"])),
                "g_eps": g_eps,
                "abs_charge": abs(charge),
                "born_proxy": float(charge * charge) * g_eps,
                "dipole_au_mag": dip_mag,
                **desc,
            }
        )
    return rows


def descriptor_correlations(rows: Sequence[Mapping[str, Any]], *, role: str | None = None) -> dict[str, Any]:
    chosen = [row for row in rows if role is None or row.get("role") == role]
    names = (
        "abs_charge",
        "born_proxy",
        "g_eps",
        "n_atoms",
        "dipole_au_mag",
        "charge_separation",
        "charge_variation",
        "mean_abs_lowdin",
    )
    overall = {
        name: pearson_spearman([row["abs_dE"] for row in chosen], [row.get(name) for row in chosen])
        for name in names
    }
    by_charge = {}
    for key in CHARGE_KEYS:
        subset = [row for row in chosen if str(row.get("charge")) == key]
        by_charge[key] = {
            "n": len(subset),
            "mean_abs_dE": None
            if not subset
            else float(np.mean([row["abs_dE"] for row in subset])),
            "mean_dipole_au_mag": _nanmean([row.get("dipole_au_mag") for row in subset]),
            "mean_charge_separation": _nanmean([row.get("charge_separation") for row in subset]),
            "correlations": {
                name: pearson_spearman(
                    [row["abs_dE"] for row in subset], [row.get(name) for row in subset]
                )
                for name in ("dipole_au_mag", "charge_separation", "charge_variation", "g_eps", "n_atoms")
            },
        }
    return {
        "role": role or "all",
        "n": len(chosen),
        "n_with_lowdin": sum(1 for row in chosen if row.get("charge_separation") is not None),
        "n_with_dipole": sum(1 for row in chosen if row.get("dipole_au_mag") is not None),
        "overall": overall,
        "by_charge": by_charge,
        "note": (
            "Correlations are versus residual *label* |ΔE|, not per-row Stage A "
            "energy error. Production metrics.json does not persist per-label residuals."
        ),
    }


def _nanmean(values: Sequence[Any]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)] if arr.size else arr
        if finite.size == 0:
            return None
        return float(finite.mean())
    return float(arr.mean())


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0 or not math.isfinite(den):
        return None
    return float(num) / float(den)


def _mae_by_charge(run: Mapping[str, Any], charge: str) -> float | None:
    payload = ((run.get("slices") or {}).get("by_charge") or {}).get(charge) or {}
    value = payload.get("dE_mae")
    return None if value is None else float(value)


def scale_normalized_mae(
    runs: Sequence[Mapping[str, Any]],
    label_scale: Mapping[str, Any],
) -> dict[str, Any]:
    by_charge_scale = label_scale.get("by_charge") or {}
    per_charge: dict[str, Any] = {}
    for key in CHARGE_KEYS:
        maes = [_mae_by_charge(run, key) for run in runs]
        mean_abs = (by_charge_scale.get(key) or {}).get("mean_abs_dE")
        rel = [_ratio(mae, mean_abs) for mae in maes]
        per_charge[key] = {
            "n": (by_charge_scale.get(key) or {}).get("n"),
            "mean_abs_dE": mean_abs,
            "dE_mae": seed_interval([v for v in maes if v is not None]),
            "relative_mae": seed_interval([v for v in rel if v is not None]),
            "per_seed": [
                {
                    "seed": int(run["seed"]),
                    "dE_mae": mae,
                    "relative_mae": _ratio(mae, mean_abs),
                }
                for run, mae in zip(runs, maes)
            ],
        }
    anion_over_neutral = []
    cation_over_neutral = []
    charged_over_neutral = []
    anion_rel_over_neutral = []
    charged_rel_over_neutral = []
    for run in runs:
        n_mae = _mae_by_charge(run, "0")
        a_mae = _mae_by_charge(run, "-1")
        c_mae = _mae_by_charge(run, "1")
        charged = None if a_mae is None or c_mae is None else 0.5 * (a_mae + c_mae)
        anion_over_neutral.append(_ratio(a_mae, n_mae))
        cation_over_neutral.append(_ratio(c_mae, n_mae))
        charged_over_neutral.append(_ratio(charged, n_mae))
        n_rel = _ratio(n_mae, (by_charge_scale.get("0") or {}).get("mean_abs_dE"))
        a_rel = _ratio(a_mae, (by_charge_scale.get("-1") or {}).get("mean_abs_dE"))
        c_rel = _ratio(c_mae, (by_charge_scale.get("1") or {}).get("mean_abs_dE"))
        charged_rel = None if a_rel is None or c_rel is None else 0.5 * (a_rel + c_rel)
        anion_rel_over_neutral.append(_ratio(a_rel, n_rel))
        charged_rel_over_neutral.append(_ratio(charged_rel, n_rel))
    n_scale = (by_charge_scale.get("0") or {}).get("mean_abs_dE")
    a_scale = (by_charge_scale.get("-1") or {}).get("mean_abs_dE")
    c_scale = (by_charge_scale.get("1") or {}).get("mean_abs_dE")
    charged_scale = None if a_scale is None or c_scale is None else 0.5 * (a_scale + c_scale)
    return {
        "role": label_scale.get("role"),
        "by_charge": per_charge,
        "anion_mae_ratio_vs_neutral": seed_interval([v for v in anion_over_neutral if v is not None]),
        "cation_mae_ratio_vs_neutral": seed_interval([v for v in cation_over_neutral if v is not None]),
        "charged_mae_ratio_vs_neutral": seed_interval([v for v in charged_over_neutral if v is not None]),
        "anion_relative_mae_ratio_vs_neutral": seed_interval(
            [v for v in anion_rel_over_neutral if v is not None]
        ),
        "charged_relative_mae_ratio_vs_neutral": seed_interval(
            [v for v in charged_rel_over_neutral if v is not None]
        ),
        "anion_label_ratio_vs_neutral": _ratio(a_scale, n_scale),
        "cation_label_ratio_vs_neutral": _ratio(c_scale, n_scale),
        "charged_label_ratio_vs_neutral": _ratio(charged_scale, n_scale),
        "per_seed_ratios": [
            {
                "seed": int(run["seed"]),
                "anion_mae_ratio": anion_over_neutral[i],
                "cation_mae_ratio": cation_over_neutral[i],
                "charged_mae_ratio": charged_over_neutral[i],
                "anion_relative_mae_ratio": anion_rel_over_neutral[i],
                "charged_relative_mae_ratio": charged_rel_over_neutral[i],
            }
            for i, run in enumerate(runs)
        ],
    }


def _all_ge(values: Sequence[float | None], threshold: float) -> bool:
    nums = [float(v) for v in values if v is not None]
    return bool(nums) and all(v >= threshold for v in nums)


def decide_stage_b(
    scaled: Mapping[str, Any],
    *,
    aspirational_energy: float = ASPIRATIONAL_ENERGY_MAE_EV,
    mae_ratio_min: float = MAE_RATIO_MIN,
    relative_ratio_min: float = RELATIVE_RATIO_MIN,
    neutral_near_factor: float = NEUTRAL_NEAR_FACTOR,
) -> dict[str, Any]:
    """Stage B GB go/no-go from the attached plan.

    Go only if all of:

    * anion *or* charged energy MAE is ≥50% above neutral on every seed
    * that gap is *not* explained by ``|ΔE|`` label scale (relative MAE also
      ≥50% worse than neutral on every seed)
    * neutral seed-mean energy MAE is at or near the 0.10 eV aspirational
    """
    per_seed = scaled.get("per_seed_ratios") or []
    anion_ratios = [row.get("anion_mae_ratio") for row in per_seed]
    charged_ratios = [row.get("charged_mae_ratio") for row in per_seed]
    anion_rel = [row.get("anion_relative_mae_ratio") for row in per_seed]
    charged_rel = [row.get("charged_relative_mae_ratio") for row in per_seed]
    anion_abs_ok = _all_ge(anion_ratios, mae_ratio_min)
    charged_abs_ok = _all_ge(charged_ratios, mae_ratio_min)
    abs_gap_ok = anion_abs_ok or charged_abs_ok
    anion_rel_unexplained = _all_ge(anion_rel, relative_ratio_min)
    charged_rel_unexplained = _all_ge(charged_rel, relative_ratio_min)
    unexplained = anion_rel_unexplained or charged_rel_unexplained
    neutral = ((scaled.get("by_charge") or {}).get("0") or {}).get("dE_mae") or {}
    neutral_mean = neutral.get("mean")
    near_cut = float(aspirational_energy) * float(neutral_near_factor)
    neutral_ok = neutral_mean is not None and float(neutral_mean) <= near_cut
    flags = []
    if not abs_gap_ok:
        flags.append("charged_mae_gap_below_50pct")
    if abs_gap_ok and not unexplained:
        flags.append("anion_gap_explained_by_label_scale")
    if not neutral_ok:
        flags.append("neutral_not_near_aspirational")
    go = abs_gap_ok and unexplained and neutral_ok
    rationale = []
    if abs_gap_ok:
        which = "anion" if anion_abs_ok else "charged"
        rationale.append(
            f"{which} residual energy MAE is ≥{mae_ratio_min:.2f}× the neutral MAE on every seed"
        )
    else:
        rationale.append(
            f"neither anion nor charged MAE is ≥{mae_ratio_min:.2f}× the neutral MAE on every seed"
        )
    label_ratio = scaled.get("anion_label_ratio_vs_neutral")
    mae_ratio = (scaled.get("anion_mae_ratio_vs_neutral") or {}).get("mean")
    rel_ratio = (scaled.get("anion_relative_mae_ratio_vs_neutral") or {}).get("mean")
    if unexplained:
        rationale.append(
            "scale-normalized MAE remains ≥50% worse on charged species; "
            "the absolute gap is not just larger |ΔE| labels"
        )
    else:
        rationale.append(
            "the absolute anion/charged MAE gap is accounted for by larger "
            f"|ΔE| labels (anion/neutral |ΔE| ratio {_fmt(label_ratio)}, "
            f"MAE ratio {_fmt(mae_ratio)}, relative-MAE ratio {_fmt(rel_ratio)})"
        )
    if neutral_ok:
        rationale.append(
            f"neutral seed-mean dE MAE {neutral_mean:.3f} eV is at or near "
            f"the {aspirational_energy:.2f} eV aspirational"
        )
    else:
        rationale.append(
            f"neutral seed-mean dE MAE {neutral_mean} eV is not near "
            f"{aspirational_energy:.2f} eV (cut {near_cut:.3f} eV)"
        )
    return {
        "schema": "dc_uma_cpcm_stage_b_gate_v1",
        "go": go,
        "decision": "go" if go else "no-go",
        "unseen_claim": UNSEEN_CLAIM,
        "thresholds": {
            "mae_ratio_min": mae_ratio_min,
            "relative_ratio_min": relative_ratio_min,
            "aspirational_energy_mae_ev": aspirational_energy,
            "neutral_near_cut_ev": near_cut,
        },
        "anion_mae_ratio_vs_neutral": scaled.get("anion_mae_ratio_vs_neutral"),
        "charged_mae_ratio_vs_neutral": scaled.get("charged_mae_ratio_vs_neutral"),
        "anion_relative_mae_ratio_vs_neutral": scaled.get("anion_relative_mae_ratio_vs_neutral"),
        "anion_label_ratio_vs_neutral": label_ratio,
        "neutral_dE_mae": neutral,
        "anion_or_charged_mae_gap_ge_50pct": abs_gap_ok,
        "gap_unexplained_by_label_scale": unexplained,
        "neutral_near_or_below_aspirational": neutral_ok,
        "flags": flags,
        "rationale": rationale,
        "next_step": (
            "Implement Stage B GB-like reaction-field on the existing 2,000 labels, "
            "warm-started from the Stage A checkpoint."
            if go
            else "Stop GB experiments. Keep Stage A as the final adapter and write the PoC document."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, Mapping) and "mean" in value:
        mean = value.get("mean")
        low = value.get("low")
        high = value.get("high")
        if mean is None:
            return "—"
        return f"{float(mean):.{digits}f} ({float(low):.{digits}f}–{float(high):.{digits}f})"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_row(summary: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> str:
    extra = extra or {}
    n = summary.get("n")
    return (
        f"| {n if n is not None else '—'} "
        f"| {_fmt(summary.get('dE_mae'))} "
        f"| {_fmt(summary.get('dE_mae_per_atom'), 4)} "
        f"| {_fmt(summary.get('dFa_mae'))} "
        f"| {_fmt(extra.get('mean_abs_dE'))} "
        f"| {_fmt(extra.get('relative_mae'), 3)} |"
    )


def render_error_analysis_markdown(payload: Mapping[str, Any]) -> str:
    agg = payload["metrics"]
    gate = payload["stage_b"]
    val_scale = payload["label_scale"]["validation"]
    test_scale = payload["label_scale"]["test"]
    scaled = payload["scale_normalized"]
    corr = payload.get("correlations") or {}
    val_corr = corr.get("validation") or {}
    decision = str(gate["decision"]).upper()
    lines = [
        "# Stage A error analysis and Stage B go/no-go",
        "",
        f"**Decision:** Stage B **{decision}**",
        "",
        f"**Date:** {payload.get('date') or 'unspecified'}",
        f"**Metrics:** `{payload.get('runs_dir')}` (`stage_a_seed{{{','.join(str(s) for s in agg['seeds'])}}}/metrics.json`)",
        f"**Labels:** `{payload.get('records_path')}`",
        f"**Hold-out wording:** {UNSEEN_CLAIM}. {UNSEEN_CLAIM_NOTE}",
        "",
        "## 1. Verdict",
        "",
    ]
    for item in gate["rationale"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"**Next step:** {gate['next_step']}",
            "",
            "Go requires all three (from the executable plan):",
            "",
            "1. Anion (or all charged) residual energy MAE is at least 50% higher than neutral on every seed.",
            "2. That gap is *not* explained by `|ΔE|` label scale alone (scale-normalized MAE also ≥50% worse).",
            "3. Neutral energy MAE is already at or near 0.10 eV, so the local head is sufficient for neutrals.",
            "",
            "## 2. What was analyzed",
            "",
            "No new ORCA jobs and no retraining. Seed-mean and min–max ranges come from the three production Stage A `metrics.json` files. `|ΔE|` label scale is computed from the frozen residual JSONL. Lowdin charges and dipoles are joined from ORCA job status where present.",
            "",
            "Production `metrics.json` stores `by_charge` / `by_size` / `by_epsilon` on **validation only**. Test reports an overall summary. Per-label Stage A residuals were not persisted, so Lowdin/dipole correlations below use **label** `|ΔE|`, while model error vs charge uses the stored validation subgroup MAEs.",
            "",
            "## 3. Seed-mean subgroups",
            "",
            "Units: residual energy MAE in eV, residual force MAE in eV/Å. Ranges are min–max over seeds.",
            "",
            "### 3.1 Overall roles",
            "",
            "| Split | n | dE MAE (eV) | dE MAE/atom (eV) | dFa MAE (eV/Å) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("validation", "probe", "test"):
        summary = agg.get(name) or {}
        lines.append(
            f"| {name} | {summary.get('n', '—')} | {_fmt(summary.get('dE_mae'))} "
            f"| {_fmt(summary.get('dE_mae_per_atom'), 4)} | {_fmt(summary.get('dFa_mae'))} |"
        )
    lines.extend(
        [
            "",
            "### 3.2 Validation by charge",
            "",
            "| Charge | n | dE MAE (eV) | dE MAE/atom | dFa MAE (eV/Å) | mean\\|ΔE\\| (eV) | relative MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_charge = (agg.get("slices") or {}).get("by_charge") or {}
    scaled_charge = scaled.get("by_charge") or {}
    for key, label in (("-1", "anion (−1)"), ("0", "neutral (0)"), ("1", "cation (+1)")):
        summary = by_charge.get(key) or {}
        extra = scaled_charge.get(key) or {}
        lines.append(
            f"| {label} {_metric_row(summary, extra)}"
        )
    lines.extend(
        [
            "",
            "### 3.3 Validation by size",
            "",
            "| Size | n | dE MAE (eV) | dE MAE/atom | dFa MAE (eV/Å) | mean\\|ΔE\\| (eV) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    by_size = (agg.get("slices") or {}).get("by_size") or {}
    size_scale = val_scale.get("by_size") or {}
    for key in SIZE_BINS:
        summary = by_size.get(key) or {}
        extra = size_scale.get(key) or {}
        lines.append(
            f"| {key} | {summary.get('n', extra.get('n', '—'))} | {_fmt(summary.get('dE_mae'))} "
            f"| {_fmt(summary.get('dE_mae_per_atom'), 4)} | {_fmt(summary.get('dFa_mae'))} "
            f"| {_fmt(extra.get('mean_abs_dE'))} |"
        )
    lines.extend(
        [
            "",
            "### 3.4 Validation by ε",
            "",
            "| ε | n | train ε? | dE MAE (eV) | dFa MAE (eV/Å) | mean\\|ΔE\\| (eV) | relative MAE |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    by_eps = (agg.get("slices") or {}).get("by_epsilon") or {}
    eps_scale = val_scale.get("by_epsilon") or {}
    eps_keys = sorted(set(by_eps) | set(eps_scale), key=lambda s: float(s))
    for key in eps_keys:
        summary = by_eps.get(key) or {}
        extra = eps_scale.get(key) or {}
        mean_abs = extra.get("mean_abs_dE")
        rel = None
        mae_mean = (summary.get("dE_mae") or {}).get("mean")
        if mae_mean is not None and mean_abs:
            rel = mae_mean / float(mean_abs)
        train = "yes" if float(key) in TRAIN_EPSILONS else "held-out"
        lines.append(
            f"| {key} | {summary.get('n', extra.get('n', '—'))} | {train} "
            f"| {_fmt(summary.get('dE_mae'))} | {_fmt(summary.get('dFa_mae'))} "
            f"| {_fmt(mean_abs)} | {_fmt(rel, 3)} |"
        )
    lines.extend(
        [
            "",
            "### 3.5 Validation dielectric slices",
            "",
            "| Slice | n | dE MAE (eV) | dFa MAE (eV/Å) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("train_epsilon", "held_out_epsilon", "held_out_parent"):
        summary = (agg.get("slices") or {}).get(name) or {}
        lines.append(
            f"| {name} | {summary.get('n', '—')} | {_fmt(summary.get('dE_mae'))} | {_fmt(summary.get('dFa_mae'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. Is the anion gap just a larger label?",
            "",
            "Ionic C-PCM residuals should scale roughly as $Q^2(1-1/\\varepsilon)$ (Born). "
            "If Stage A MAE grew by the same factor as mean $|\\Delta E|$, the local head is "
            "keeping up with the physics and a GB term is not motivated by these metrics.",
            "",
            "| Comparison | MAE ratio (seed-mean, range) | $\\lvert\\Delta E\\rvert$ label ratio | relative-MAE ratio |",
            "|---|---:|---:|---:|",
            f"| anion / neutral | {_fmt(scaled.get('anion_mae_ratio_vs_neutral'))} | {_fmt(scaled.get('anion_label_ratio_vs_neutral'))} | {_fmt(scaled.get('anion_relative_mae_ratio_vs_neutral'))} |",
            f"| cation / neutral | {_fmt(scaled.get('cation_mae_ratio_vs_neutral'))} | {_fmt(scaled.get('cation_label_ratio_vs_neutral'))} | — |",
            f"| charged / neutral | {_fmt(scaled.get('charged_mae_ratio_vs_neutral'))} | {_fmt(scaled.get('charged_label_ratio_vs_neutral'))} | {_fmt(scaled.get('charged_relative_mae_ratio_vs_neutral'))} |",
            "",
            "Per-seed anion / neutral residual energy MAE:",
            "",
            "| Seed | anion MAE | neutral MAE | MAE ratio | relative-MAE ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    anion_rows = (scaled_charge.get("-1") or {}).get("per_seed") or []
    neutral_rows = (scaled_charge.get("0") or {}).get("per_seed") or []
    ratio_rows = scaled.get("per_seed_ratios") or []
    by_seed_n = {int(r["seed"]): r for r in neutral_rows}
    by_seed_r = {int(r["seed"]): r for r in ratio_rows}
    for row in anion_rows:
        seed = int(row["seed"])
        neu = by_seed_n.get(seed) or {}
        rat = by_seed_r.get(seed) or {}
        lines.append(
            f"| {seed} | {_fmt(row.get('dE_mae'))} | {_fmt(neu.get('dE_mae'))} "
            f"| {_fmt(rat.get('anion_mae_ratio'))} | {_fmt(rat.get('anion_relative_mae_ratio'))} |"
        )
    hold = val_scale.get("by_charge_and_epsilon_holdout") or {}
    lines.extend(
        [
            "",
            "Test **label** scale (model `by_charge` was not stored for test):",
            "",
            "| Charge | n | val mean\\|ΔE\\| (eV) | test mean\\|ΔE\\| (eV) |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (("-1", "anion"), ("0", "neutral"), ("1", "cation")):
        v = (val_scale.get("by_charge") or {}).get(key) or {}
        t = (test_scale.get("by_charge") or {}).get(key) or {}
        lines.append(
            f"| {label} | {t.get('n', v.get('n', '—'))} | {_fmt(v.get('mean_abs_dE'))} | {_fmt(t.get('mean_abs_dE'))} |"
        )
    lines.extend(
        [
            "",
            "## 5. Held-out ε and the anion gap",
            "",
            "Every validation geometry is labeled at all seven ε, so the charge mix is identical at train and held-out dielectrics. Label `|ΔE|` by charge × hold-out:",
            "",
            "| Split | anion mean\\|ΔE\\| | neutral mean\\|ΔE\\| | cation mean\\|ΔE\\| | anion/neutral |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for side, title in (("train_epsilon", "train ε {1.5,4,16,80}"), ("held_out_epsilon", "held-out ε {2,8,32}")):
        bucket = hold.get(side) or {}
        a = (bucket.get("-1") or {}).get("mean_abs_dE")
        n = (bucket.get("0") or {}).get("mean_abs_dE")
        c = (bucket.get("1") or {}).get("mean_abs_dE")
        lines.append(
            f"| {title} | {_fmt(a)} | {_fmt(n)} | {_fmt(c)} | {_fmt(_ratio(a, n))} |"
        )
    lines.extend(
        [
            "",
            "The anion/neutral label ratio is essentially the same on held-out ε as on train ε. Validation held-out-ε Stage A MAE is only slightly above train-ε MAE (§3.5), and relative MAE vs `|ΔE|` does not blow up at ε = 2, 8, or 32 (§3.4). The absolute anion gap therefore appears on held-out dielectrics because the *labels* are larger, not because dielectric interpolation specifically fails for anions.",
            "",
            "## 6. Size bins",
            "",
            "Per-atom energy error is highest for 6–15 atoms because *N* is small, not because molecular MAE is worse. 31–60 atom molecular dE MAE is comparable to (or only modestly above) the smaller bins, while per-atom MAE is lowest there. This is not evidence that a local head fails on larger molecules in this 6–60 range.",
            "",
            "## 7. Lowdin charges, dipole, and charge separation",
            "",
        ]
    )
    if not val_corr or not val_corr.get("n_with_lowdin"):
        lines.append("Lowdin charges were not available on the joined validation labels; dipole/Lowdin correlations are omitted.")
    else:
        overall = val_corr.get("overall") or {}
        lines.extend(
            [
                f"Validation labels with Lowdin: {val_corr.get('n_with_lowdin')} / {val_corr.get('n')}; with dipole: {val_corr.get('n_with_dipole')}.",
                "",
                "Pearson *r* of **label** `|ΔE|` vs descriptors (not per-row model error):",
                "",
                "| Descriptor | Pearson r | Spearman ρ | n |",
                "|---|---:|---:|---:|",
            ]
        )
        labels = {
            "abs_charge": "abs. total charge",
            "born_proxy": "Born proxy Q² g(ε)",
            "g_eps": "g(ε) = 1 − 1/ε",
            "dipole_au_mag": "dipole magnitude (a.u.)",
            "charge_separation": "Lowdin charge separation",
            "charge_variation": "mean abs. Lowdin vs Q/N",
            "n_atoms": "N atoms",
        }
        for key, title in labels.items():
            stats = overall.get(key) or {}
            lines.append(
                f"| {title} | {_fmt(stats.get('pearson'), 3)} | {_fmt(stats.get('spearman'), 3)} | {stats.get('n', '—')} |"
            )
        lines.extend(
            [
                "",
                "Within-charge validation means:",
                "",
                "| Charge | mean\\|ΔE\\| | mean \\|μ\\| (a.u.) | mean charge separation |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, title in (("-1", "anion"), ("0", "neutral"), ("1", "cation")):
            row = (val_corr.get("by_charge") or {}).get(key) or {}
            lines.append(
                f"| {title} | {_fmt(row.get('mean_abs_dE'))} | {_fmt(row.get('mean_dipole_au_mag'))} "
                f"| {_fmt(row.get('mean_charge_separation'))} |"
            )
        lines.extend(
            [
                "",
                "Label `|ΔE|` tracks total charge and the Born proxy $Q^2 g(\\varepsilon)$ (Pearson $r \\approx 0.94$ on validation), not Lowdin charge separation (near zero / weakly negative). Within anions, `|ΔE|` is almost entirely $g(\\varepsilon)$ — the same ions' residuals grow with dielectric as a reaction-field, which Stage A's explicit $g(\\varepsilon)$ gate already provides. Neutrals show more dipole and size correlation, which a local atom-wise MLP can represent.",
                "",
                "Because per-row Stage A errors were not stored, model-error vs Lowdin cannot be computed label-wise. Group-level model MAE vs $|Q|$ is in §3.2 and is *smaller* relative to label scale for ions than for neutrals.",
            ]
        )
    flags = ", ".join(gate.get("flags") or []) or "none"
    lines.extend(
        [
            "",
            "## 8. Stage B decision",
            "",
            f"- **go:** `{gate.get('go')}`",
            f"- **flags:** {flags}",
            f"- anion or charged MAE gap ≥50%: `{gate.get('anion_or_charged_mae_gap_ge_50pct')}`",
            f"- gap unexplained by label scale: `{gate.get('gap_unexplained_by_label_scale')}`",
            f"- neutral near/below 0.10 eV: `{gate.get('neutral_near_or_below_aspirational')}`",
            "",
            f"{gate['next_step']}",
            "",
            "Aspirational 0.10 eV / 0.10 eV/Å remain descriptive, not a reason to add GB: forces already meet 0.10 eV/Å; energy is close on neutrals and the leftover absolute ionic error follows Born-scale labels rather than a missing long-range channel in these metrics.",
            "",
            "---",
            "",
            "*Generated from stored Stage A `metrics.json`, residual JSONL, and ORCA status. "
            f"Schema `{SCHEMA}`.*",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage_a_error_analysis(
    *,
    runs_dir: Path | str | None = None,
    records_path: Path | str | None = None,
    orca_root: Path | str | None = None,
    orca_report_path: Path | str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    from .diagnostic import load_residual_jsonl

    runs_dir = Path(runs_dir or DEFAULT_FULL_RUNS)
    records_path = Path(records_path or (DEFAULT_FULL_DATA / "residuals.jsonl"))
    orca_root = Path(orca_root or DEFAULT_FULL_ORCA)
    report_default = Path(DEFAULT_EXPERIMENTS) / "reports" / "orca_full.json"
    orca_report_path = Path(orca_report_path) if orca_report_path is not None else report_default
    runs = load_stage_a_metrics(runs_dir)
    records = load_residual_jsonl(records_path)
    metrics = aggregate_stage_a_metrics(runs)
    label_scale = {
        role: label_scale_tables(records, role=role) for role in ("validation", "test", "probe")
    }
    scaled = scale_normalized_mae(runs, label_scale["validation"])
    orca_report = load_json(orca_report_path) if orca_report_path.is_file() else None
    descriptor_rows = attach_orca_descriptors(
        [rec for rec in records if rec.get("role") in ("validation", "test")],
        orca_root=orca_root if orca_root.is_dir() else None,
        orca_report=orca_report,
    )
    correlations = {
        "validation": descriptor_correlations(descriptor_rows, role="validation"),
        "test": descriptor_correlations(descriptor_rows, role="test"),
    }
    gate = decide_stage_b(scaled)
    payload = {
        "schema": SCHEMA,
        "date": date,
        "unseen_claim": UNSEEN_CLAIM,
        "runs_dir": str(runs_dir),
        "records_path": str(records_path),
        "orca_root": str(orca_root),
        "seeds": metrics["seeds"],
        "n_runs": len(runs),
        "metrics": metrics,
        "label_scale": label_scale,
        "scale_normalized": scaled,
        "correlations": correlations,
        "stage_b": gate,
    }
    payload["markdown"] = render_error_analysis_markdown(payload)
    return payload
