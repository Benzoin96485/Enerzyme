"""Summarize archived V100 eager measurements for the refactor and main checkpoints."""

import json
from collections import Counter
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parents[1] / "test/fixtures/bench_gpu"
SYSTEMS = ("mol_batch_32x20", "bulk_256", "large_2000")
METRICS = ("infer_ms", "train_ms", "infer_peak_mb", "train_peak_mb")
PRECISIONS = ("float32", "float64")


def load(name):
    return json.loads((ARCHIVE / (name + ".json")).read_text())


def phase_zero():
    cells = {}
    for precision in PRECISIONS:
        baseline = load("baseline_eager_" + precision)
        candidate = load("initial_candidate_eager_" + precision)
        if baseline.keys() != candidate.keys():
            raise ValueError(f"The initial {precision} archives cover different models")
        for name in sorted(baseline.keys() - {"_meta"}):
            for system in SYSTEMS:
                for metric in METRICS:
                    old = baseline[name].get(system, {}).get(metric)
                    new = candidate[name].get(system, {}).get(metric)
                    delta = None if old is None or new is None else 100 * (new / old - 1)
                    cells[name, precision, system, metric] = delta, "6316675"
    return cells


def record(cells, key, percent, job):
    if key not in cells:
        raise ValueError(f"Unknown measurement {key} in {job}")
    if percent is not None:
        cells[key] = percent, job


def apply_family_and_attention(cells):
    family = load("family_eager_v100_rechecks")
    for job, cohort in family["cohorts"].items():
        for model, dtype, system, *samples, deltas in cohort["paired_systems"]:
            for metric, delta in zip(family["metric_columns"], deltas, strict=True):
                record(cells, (model, dtype, system, metric), delta, job)

    attention = load("shared_attention_eager")
    job = attention["_meta"]["job"]
    for model, rows in attention["measurements"].items():
        for measurement, runs in rows.items():
            result = runs.get("abba") or runs["initial"]
            record(cells, (*model.split("/"), *measurement.split("/")), result.get("percent"), job)

    followup = attention["followup"]
    # The geometry candidate was measured twice for fp32 MatterSim. Use the
    # documented pooled median of all four rounds on each side.
    latest = {**followup["geometry_3e65250"],
              "mattersim/float32": followup["mattersim_geometry_pooled"]["cells"]}
    for model, rows in latest.items():
        for measurement, result in rows.items():
            record(cells, (*model.split("/"), *measurement.split("/")), result["percent"], followup["metadata"]["job"])


def apply_scalar(phase, main):
    archive = load("scalar_legacy_eager_v100_rechecks")
    for job, cohort in sorted(archive["cohorts"].items()):
        for label, case in cohort["cases"].items():
            model, dtype = label.rsplit("-", 1)
            if cohort["baseline"] == "b05ea1a":
                cells = main
            elif cohort["baseline"] == "1aa0f8b" and (model, dtype, SYSTEMS[0], METRICS[0]) in phase:
                cells = phase
            else:
                # Alternate attention implementations have separate timings.
                continue
            for system in SYSTEMS:
                for metric, delta in zip(archive["metric_columns"], case[system]["delta_percent"], strict=True):
                    key = model, dtype, system, metric
                    if cells is main:
                        main.setdefault(key, (None, job))
                    record(cells, key, delta, job)


def summarize(phase, main):
    lines = [
        "# V100 eager performance",
        "",
        "The Phase 0 reference is `1aa0f8b`. The main-branch comparison uses trained checkpoints from "
        "`b05ea1a`. Both measure energy and forces during inference, a backward pass through the force "
        "loss during training, and peak allocation for each. The three systems contain 32 molecules of "
        "20 atoms, a 256-atom periodic cell, and a 2,000-atom cluster.",
        "",
        "Each revised measurement uses the paired run for that case, with its own candidate revision. "
        "The JSON archives alongside this report contain the measured times, memory, exceptions, and run order.",
        "",
        "| Reference | Within +3% | Above +3% | Unavailable | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    xpainn = load("xpainn_v100")["eager_graph_pruning"]["selected_metrics"]
    for label, percentages in (
        ("Phase 0 defaults", [value for value, _ in phase.values()]),
        ("Main checkpoints", [value for value, _ in main.values()]),
        ("XequiNet xPaiNN", [row["change_percent"] for row in xpainn.values()]),
    ):
        values = [value for value in percentages if value is not None]
        above = sum(value > 3 for value in values)
        lines.append(f"| {label} | {len(values) - above} | {above} | "
                     f"{len(percentages) - len(values)} | {len(percentages)} |")
    lines += ["", "The original xPaiNN wrapper omitted periodic image offsets. The periodic timing compares "
              "the two wrappers. Independent energy, force and weight-gradient checks use the upstream "
              "XequiNet model with the correct periodic offsets on CPU and V100."]

    slower = [(key, value, job) for key, (value, job) in main.items()
              if key[-1].endswith("_ms") and value is not None and value > 0]
    total_times = sum(key[-1].endswith("_ms") and v is not None for key, (v, _) in main.items())
    lines += ["", f"{total_times - len(slower)} of {total_times} main-checkpoint timing medians were as fast as or faster than main."]
    if slower:
        lines.append("")
    for (model, dtype, system, metric), value, job in sorted(slower):
        lines.append(f"- {model} {dtype}, {system}, {metric} +{value:.2f}% (run {job})")
    for key, (value, job) in sorted(phase.items()):
        if value is not None and value > 3:
            lines.append(f"- Phase 0 {' '.join(key)} +{value:.2f}% (run {job})")

    missing = Counter((name, dtype) for (name, dtype, _, _), (value, _) in phase.items() if value is None)
    lines += ["", "## Missing paired measurements", "",
              "Each model and precision has 12 possible comparisons. Individual exceptions are preserved "
              "in the baseline and candidate archives.", "",
              "| Preset | float32 | float64 |", "| --- | ---: | ---: |"]
    for name in sorted({name for name, _ in missing}):
        lines.append(f"| {name} | {missing[name, 'float32']} | {missing[name, 'float64']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    phase = phase_zero()
    main = {}
    apply_family_and_attention(phase)
    apply_scalar(phase, main)
    print(summarize(phase, main), end="")
