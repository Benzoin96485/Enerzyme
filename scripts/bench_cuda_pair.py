"""Measure frozen numerical parity, latency and allocated memory on one visible CUDA device."""

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from bench_cores import SYSTEM_SETS
from bench_frozen_compile import compare_frozen_runs


def assess(comparisons, threshold):
    """Apply the numerical checks before comparing medians from the paired legs."""
    cells = {}
    passed = True
    for system in comparisons[0]["requested_systems"]:
        cells[system] = {}
        for mode in comparisons[0]["modes"]:
            pairs = [comparison["comparisons"][system][mode] for comparison in comparisons]
            cell = cells[system][mode] = {"pair_status": [pair["status"] for pair in pairs]}
            cell["numerically_valid"] = all(pair["status"] == "both_validated" for pair in pairs)
            if cell["numerically_valid"]:
                for metric in ("ms", "peak_MiB"):
                    base = statistics.median(pair["baseline"][metric] for pair in pairs)
                    candidate = statistics.median(pair["candidate"][metric] for pair in pairs)
                    cell[metric] = {"baseline": base, "candidate": candidate, "delta_percent": 100 * (candidate / base - 1)}
                cell["passed"] = all(cell[metric]["candidate"] <= cell[metric]["baseline"] * (1 + threshold / 100)
                                     for metric in ("ms", "peak_MiB"))
            else:
                cell["passed"] = False
            passed &= cell["passed"]
    return {"passed": passed, "maximum_increase_percent": threshold, "cells": cells}


def run(args, directory):
    driver = Path(__file__).with_name("bench_frozen_compile.py")
    fixture = ["--main-fixture", args.main_fixture] if args.main_fixture else ["--snapshot", args.snapshot]
    order = ("baseline", "candidate") if args.order == "ab" else ("baseline", "candidate", "candidate", "baseline")
    report = {"order": list(order), "pairs": [], "worker_failures": [], "comparison_failures": [], "passed": False}
    legs = {}
    for index, side in enumerate(order):
        output, refs = (directory / f"{index}-{side}.{ext}" for ext in ("json", "pt"))
        command = [sys.executable, str(driver), "--model-tree", str(getattr(args, side).resolve()),
                   fixture[0], str(fixture[1].resolve()), "--precision", args.precision, "--device", "cuda",
                   "--systems", *args.systems, "--modes", *args.modes, "--threads", str(args.threads),
                   "--reps", str(args.reps), "--out", str(output), "--eager-grad-out", str(refs)]
        if args.execution == "eager":
            command.append("--eager")
        completed = subprocess.run(command)
        if completed.returncode:
            reason = None
            if output.is_file():
                worker = next(iter(json.loads(output.read_text()).values()))
                reason = worker.get("worker_error")
            report["worker_failures"].append({"leg": index, "side": side, "exit_code": completed.returncode,
                                              "error": reason})
        legs[side] = (output, refs)
        if index % 2:
            paths, reference_paths = zip(legs["baseline"], legs["candidate"])
            if all(path.is_file() for path in paths):
                try:
                    comparison = compare_frozen_runs(paths, reference_paths)
                    if report["pairs"]:
                        first = report["pairs"][0]
                        for key in ("device_uuid", "cuda_environment", "snapshot_sha256", "precision"):
                            if first[key] != comparison[key]:
                                raise ValueError(f"paired rounds have different {key}")
                        for revision in ("baseline", "candidate"):
                            if first["source"][revision]["model_revision"] != comparison["source"][revision]["model_revision"]:
                                raise ValueError(f"{revision} revision changed between paired rounds")
                    report["pairs"].append(comparison)
                except ValueError as exc:
                    report["comparison_failures"].append({"pair": index // 2, "error": str(exc)})
            legs.clear()
    if len(report["pairs"]) == len(order) // 2:
        report.update(assess(report["pairs"], args.threshold))
        report["passed"] &= not report["worker_failures"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline Enerzyme checkout")
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1], help="candidate checkout")
    fixtures = parser.add_mutually_exclusive_group(required=True)
    fixtures.add_argument("--snapshot", type=Path, help="pre-refactor fixture in test/fixtures/presnapshot")
    fixtures.add_argument("--main-fixture", type=Path, help="trained .pt.gz fixture in test/fixtures/main_legacy")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--execution", choices=["eager", "compile"], default="eager")
    parser.add_argument("--systems", nargs="+", choices=SYSTEM_SETS["acceptance"], default=list(SYSTEM_SETS["acceptance"]))
    parser.add_argument("--modes", nargs="+", choices=["infer", "train"], default=["infer", "train"])
    parser.add_argument("--order", choices=["ab", "abba"], default="abba", help="A = baseline, B = candidate")
    parser.add_argument("--threshold", type=float, default=3.0, help="maximum allowed time or memory increase in percent")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold < 0 or min(args.reps, args.threads) < 1:
        parser.error("threshold must be finite and nonnegative, and repetitions and threads must be positive")
    import torch

    if not torch.cuda.is_available():
        parser.error("this comparison requires a visible CUDA device")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as directory:
        report = run(args, Path(directory))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
