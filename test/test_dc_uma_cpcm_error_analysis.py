"""Stage A three-seed error analysis and Stage B GB go/no-go."""

from __future__ import annotations

import json

import numpy as np
import pytest

from enerzyme.data.dc_uma_cpcm.error_analysis import (
    MAE_RATIO_MIN,
    RELATIVE_RATIO_MIN,
    SCHEMA,
    aggregate_stage_a_metrics,
    attach_orca_descriptors,
    decide_stage_b,
    descriptor_correlations,
    label_scale_tables,
    pearson_spearman,
    render_error_analysis_markdown,
    scale_normalized_mae,
)


def _summary(*, n=126, dE_mae=0.19, dE_mae_per_atom=0.016, dFa_mae=0.036, joint=0.05):
    return {
        "n": n,
        "dE_mae": dE_mae,
        "dE_mae_per_atom": dE_mae_per_atom,
        "dFa_mae": dFa_mae,
        "joint": joint,
        "E_mae": dE_mae,
        "Fa_mae": dFa_mae + 0.002,
    }


def _run(seed: int, anion_mae: float, neutral_mae: float, cation_mae: float) -> dict:
    return {
        "schema": "dc_uma_cpcm_full_run_v1",
        "baseline": "stage_a",
        "seed": seed,
        "validation": _summary(n=504, dE_mae=0.12, dE_mae_per_atom=0.008, dFa_mae=0.027, joint=0.035),
        "probe": _summary(n=96, dE_mae=0.04, dE_mae_per_atom=0.004, dFa_mae=0.010, joint=0.014),
        "test": _summary(n=504, dE_mae=0.11, dE_mae_per_atom=0.007, dFa_mae=0.026, joint=0.033),
        "slices": {
            "train_epsilon": _summary(n=288, dE_mae=0.11),
            "held_out_epsilon": _summary(n=216, dE_mae=0.13),
            "held_out_parent": _summary(n=504, dE_mae=0.12),
            "by_charge": {
                "-1": _summary(n=126, dE_mae=anion_mae),
                "0": _summary(n=252, dE_mae=neutral_mae),
                "1": _summary(n=126, dE_mae=cation_mae),
            },
            "by_size": {
                "6-15": _summary(n=168, dE_mae=0.12, dE_mae_per_atom=0.016),
                "16-30": _summary(n=168, dE_mae=0.10, dE_mae_per_atom=0.005),
                "31-60": _summary(n=168, dE_mae=0.13, dE_mae_per_atom=0.004),
            },
            "by_epsilon": {
                "1.5": _summary(n=72, dE_mae=0.08),
                "80.0": _summary(n=72, dE_mae=0.13),
            },
        },
    }


def _records(*, anion_abs=1.8, neutral_abs=0.4, cation_abs=2.0, role="validation"):
    records = []
    for charge, scale, n in ((-1, anion_abs, 4), (0, neutral_abs, 8), (1, cation_abs, 4)):
        for i in range(n):
            eps = 80.0 if i % 2 else 1.5
            lowdin = [float(charge) / 3.0] * 3
            records.append(
                {
                    "label_id": f"{role}-Q{charge}-{i}",
                    "role": role,
                    "charge": charge,
                    "n_atoms": 3,
                    "size_bin": "6-15",
                    "dielectric": eps,
                    "train_epsilon": eps == 1.5,
                    "held_out_epsilon": eps != 1.5,
                    "dE": -scale if eps == 80.0 else -0.3 * scale,
                    "lowdin_charges": lowdin,
                    "dipole_au": [0.2 * abs(charge) + 0.1 * scale, 0.0, 0.0],
                }
            )
    return records


def test_pearson_spearman_perfect_line():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 2.0, 4.0, 6.0, 8.0]
    stats = pearson_spearman(x, y)
    assert stats["n"] == 5
    assert stats["pearson"] == pytest.approx(1.0)
    assert stats["spearman"] == pytest.approx(1.0)


def test_label_scale_by_charge_and_held_out_epsilon():
    tables = label_scale_tables(_records(), role="validation")
    assert tables["n"] == 16
    assert tables["by_charge"]["-1"]["n"] == 4
    assert tables["by_charge"]["0"]["mean_abs_dE"] == pytest.approx(0.26)  # 0.5*(0.4+0.12)
    anion = tables["by_charge_and_epsilon_holdout"]["held_out_epsilon"]["-1"]
    assert anion["n"] == 2
    assert anion["mean_abs_dE"] == pytest.approx(1.8)


def test_aggregate_seed_mean_and_range():
    runs = [
        _run(20260817, 0.198, 0.087, 0.104),
        _run(20260818, 0.192, 0.076, 0.097),
        _run(20260819, 0.199, 0.114, 0.090),
    ]
    agg = aggregate_stage_a_metrics(runs)
    anion = agg["slices"]["by_charge"]["-1"]["dE_mae"]
    assert anion["n"] == 3
    assert anion["mean"] == pytest.approx((0.198 + 0.192 + 0.199) / 3)
    assert anion["low"] == pytest.approx(0.192)
    assert anion["high"] == pytest.approx(0.199)
    assert agg["slices"]["by_charge"]["-1"]["n"] == 126
    assert agg["test"]["dE_mae"]["mean"] == pytest.approx(0.11)


def test_scale_explains_anion_gap_is_nogo():
    """Absolute anion MAE is 2× neutral, but |ΔE| is 4.5× — Stage B no-go."""
    runs = [
        _run(20260817, 0.20, 0.08, 0.10),
        _run(20260818, 0.19, 0.08, 0.10),
        _run(20260819, 0.21, 0.09, 0.09),
    ]
    records = _records(anion_abs=1.8, neutral_abs=0.4, cation_abs=2.0)
    scaled = scale_normalized_mae(runs, label_scale_tables(records, role="validation"))
    assert scaled["anion_mae_ratio_vs_neutral"]["mean"] > MAE_RATIO_MIN
    assert scaled["anion_label_ratio_vs_neutral"] > scaled["anion_mae_ratio_vs_neutral"]["mean"]
    gate = decide_stage_b(scaled)
    assert gate["go"] is False
    assert gate["decision"] == "no-go"
    assert gate["anion_or_charged_mae_gap_ge_50pct"] is True
    assert gate["gap_unexplained_by_label_scale"] is False
    assert "anion_gap_explained_by_label_scale" in gate["flags"]
    assert "Stop GB" in gate["next_step"]


def test_unexplained_relative_gap_is_go():
    """Similar |ΔE| across charges but anion MAE ≥1.5× — Stage B go."""
    runs = [
        _run(20260817, 0.16, 0.08, 0.09),
        _run(20260818, 0.15, 0.08, 0.09),
        _run(20260819, 0.17, 0.09, 0.10),
    ]
    records = _records(anion_abs=0.45, neutral_abs=0.40, cation_abs=0.42)
    scaled = scale_normalized_mae(runs, label_scale_tables(records, role="validation"))
    assert scaled["anion_relative_mae_ratio_vs_neutral"]["low"] >= RELATIVE_RATIO_MIN
    gate = decide_stage_b(scaled)
    assert gate["go"] is True
    assert gate["flags"] == []
    assert "Stage B" in gate["next_step"]


def test_lowdin_born_proxy_correlation():
    records = _records(anion_abs=2.0, neutral_abs=0.3, cation_abs=2.2)
    rows = attach_orca_descriptors(records)
    assert rows[0]["charge_separation"] is not None
    corr = descriptor_correlations(rows, role="validation")
    born = corr["overall"]["born_proxy"]["pearson"]
    sep = corr["overall"]["charge_separation"]["pearson"]
    assert born is not None and born > 0.8
    assert corr["n_with_lowdin"] == len(records)
    # Charge separation is nearly constant within this toy Lowdin assignment.
    assert sep is None or abs(sep) < born


def test_markdown_states_nogo_and_unseen_claim():
    runs = [
        _run(20260817, 0.20, 0.08, 0.10),
        _run(20260818, 0.19, 0.08, 0.10),
        _run(20260819, 0.21, 0.09, 0.09),
    ]
    records = _records()
    metrics = aggregate_stage_a_metrics(runs)
    scaled = scale_normalized_mae(runs, label_scale_tables(records, role="validation"))
    rows = attach_orca_descriptors(records)
    payload = {
        "schema": SCHEMA,
        "date": "2026-08-26",
        "runs_dir": "/tmp/runs/full",
        "records_path": "/tmp/residuals.jsonl",
        "metrics": metrics,
        "label_scale": {
            "validation": label_scale_tables(records, role="validation"),
            "test": label_scale_tables(records, role="test"),
        },
        "scale_normalized": scaled,
        "correlations": {"validation": descriptor_correlations(rows, role="validation")},
        "stage_b": decide_stage_b(scaled),
    }
    md = render_error_analysis_markdown(payload)
    assert "Stage B **NO-GO**" in md
    assert "adapter-unseen parent" in md
    assert "anion_gap_explained_by_label_scale" in md
    assert "by charge" in md.lower()
    assert json.dumps(payload["stage_b"]["go"]) == "false"
