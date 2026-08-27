#!/usr/bin/env python3
"""240-job diagnostic C-PCM labels and zero / x-only / Stage A baselines.

Examples:

  python scripts/dc_uma_cpcm/diagnostic.py prepare-orca
  python scripts/dc_uma_cpcm/diagnostic.py run-orca --index $SLURM_ARRAY_TASK_ID
  python scripts/dc_uma_cpcm/diagnostic.py summarize-orca
  python scripts/dc_uma_cpcm/diagnostic.py build-dataset
  python scripts/dc_uma_cpcm/diagnostic.py microbench
  python scripts/dc_uma_cpcm/diagnostic.py train --baseline stage_a --seed 20260817
  python scripts/dc_uma_cpcm/diagnostic.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_EXPERIMENTS, DIAGNOSTIC_SEEDS, N_DIAG_JOBS
    from enerzyme.data.dc_uma_cpcm.diagnostic import DEFAULT_DIAGNOSTIC_DATA, DEFAULT_DIAGNOSTIC_ORCA
    from enerzyme.qm.orca_cpcm import DEFAULT_MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare-orca", help="Write 240 recoverable C-PCM job directories")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--root", type=Path, default=DEFAULT_DIAGNOSTIC_ORCA)
    prepare.add_argument("--nprocs", type=int, default=32)
    prepare.add_argument("--nbo", action="store_true")
    prepare.add_argument("--require-n", type=int, default=N_DIAG_JOBS)

    run = sub.add_parser("run-orca", help="Run or resume one diagnostic array task")
    run.add_argument("--root", type=Path, default=DEFAULT_DIAGNOSTIC_ORCA)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--skip-fd", action="store_true", default=True)

    summarize = sub.add_parser("summarize-orca", help="Collect diagnostic ORCA quality-gate report")
    summarize.add_argument("--root", type=Path, default=DEFAULT_DIAGNOSTIC_ORCA)
    summarize.add_argument(
        "--report",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "reports" / "orca_diagnostic.json",
    )

    build = sub.add_parser("build-dataset", help="Pair C-PCM jobs with OMol vacuum into residual labels")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--root", type=Path, default=DEFAULT_DIAGNOSTIC_ORCA)
    build.add_argument("--out", type=Path, default=DEFAULT_DIAGNOSTIC_DATA)
    build.add_argument(
        "--pilot-report",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "reports" / "orca_pilot.json",
    )
    build.add_argument("--allow-omol-vacuum", action="store_true")
    build.add_argument("--require-n", type=int, default=N_DIAG_JOBS)

    bench = sub.add_parser("microbench", help="GPU second-order backprop memory/time vs n_atoms")
    bench.add_argument("--report", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "reports" / "cpcm_microbench.json")
    bench.add_argument("--adapter-mode", default="stage_a")
    bench.add_argument("--microbatch", type=int, default=1)

    train = sub.add_parser("train", help="Train one baseline × seed on the diagnostic residual set")
    train.add_argument("--records", type=Path, default=DEFAULT_DIAGNOSTIC_DATA / "residuals.jsonl")
    train.add_argument("--baseline", required=True, choices=["zero", "x_only", "stage_a"])
    train.add_argument("--seed", type=int, default=DIAGNOSTIC_SEEDS[0])
    train.add_argument("--out", type=Path, default=None)
    train.add_argument("--accumulate-grad-batches", type=int, default=None)
    train.add_argument("--microbench", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "reports" / "cpcm_microbench.json")
    train.add_argument("--max-epochs", type=int, default=200)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--toy", action="store_true", help="Use ToyCpcmModel instead of UMA (tests / CPU)")

    grid = sub.add_parser("run-grid", help="Train zero / x-only / Stage A × 3 seeds and write the go/no-go report")
    grid.add_argument("--records", type=Path, default=DEFAULT_DIAGNOSTIC_DATA / "residuals.jsonl")
    grid.add_argument("--out", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "runs" / "diagnostic")
    grid.add_argument("--accumulate-grad-batches", type=int, default=None)
    grid.add_argument("--microbench", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "reports" / "cpcm_microbench.json")
    grid.add_argument("--max-epochs", type=int, default=200)
    grid.add_argument("--patience", type=int, default=20)
    grid.add_argument("--batch-size", type=int, default=1)
    grid.add_argument("--max-steps", type=int, default=None)
    grid.add_argument("--toy", action="store_true")

    report = sub.add_parser("report", help="Aggregate existing run metrics into the diagnostic go/no-go")
    report.add_argument("--runs", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "runs" / "diagnostic")
    report.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "reports" / "diagnostic_train.json",
    )
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _accum_from_args(args) -> int:
    if getattr(args, "accumulate_grad_batches", None):
        return int(args.accumulate_grad_batches)
    path = getattr(args, "microbench", None)
    if path and Path(path).is_file():
        payload = _load_json(Path(path))
        return int(payload.get("recommend_accumulate_grad_batches") or 1)
    return 1


def _build_uma(baseline: str, seed: int):
    from enerzyme.models.esen.cpcm import DCUMACPCMWrapper
    from enerzyme.models.esen.runtime import OMOL_TASK_NAME, resolve_uma_checkpoint

    set_seed = __import__("enerzyme.tasks.cpcm_diagnostic", fromlist=["set_seed"]).set_seed
    set_seed(seed)
    return DCUMACPCMWrapper(
        checkpoint_path=str(resolve_uma_checkpoint()),
        task_name=OMOL_TASK_NAME,
        frozen=True,
        use_ema=True,
        inference_settings="batch",
        strict=True,
        adapter_mode=baseline,
    )


def _build_toy(baseline: str, seed: int):
    from enerzyme.tasks.cpcm_diagnostic import ToyCpcmModel, set_seed

    set_seed(seed)
    return ToyCpcmModel(adapter_mode=baseline)


def main() -> int:
    args = _parse_args()
    if args.cmd == "prepare-orca":
        from enerzyme.qm.orca_cpcm import load_manifest, prepare_diagnostic_jobs

        manifest = load_manifest(args.manifest)
        array_map = prepare_diagnostic_jobs(
            manifest,
            args.root,
            nprocs=int(args.nprocs),
            nbo=bool(args.nbo),
            require_n=args.require_n,
        )
        print(json.dumps({"root": str(args.root), "n_jobs": array_map["n_jobs"], "subset": "diagnostic"}, indent=2))
        return 0

    if args.cmd == "run-orca":
        from enerzyme.qm.orca_cpcm import ARRAY_MAP_NAME, load_json, run_job

        array_map = load_json(Path(args.root) / ARRAY_MAP_NAME)
        jobs = {int(item["index"]): item for item in array_map["jobs"]}
        if int(args.index) not in jobs:
            print(f"array index {args.index} not in {args.root}/{ARRAY_MAP_NAME}", file=sys.stderr)
            return 2
        status = run_job(Path(args.root) / jobs[int(args.index)]["relpath"], skip_fd=bool(args.skip_fd))
        print(json.dumps({k: status.get(k) for k in ("job_id", "status", "attempt", "energy_ev", "error")}, indent=2))
        return 0 if status.get("status") == "success" else 1

    if args.cmd == "summarize-orca":
        from enerzyme.qm.orca_cpcm import summarize_diagnostic

        report = summarize_diagnostic(args.root, report_path=args.report)
        print(
            json.dumps(
                {
                    "n_jobs": report["n_jobs"],
                    "n_success": report["n_success"],
                    "first_pass_rate": report["first_pass_rate"],
                    "stop_expand": report["stop_expand"],
                },
                indent=2,
            )
        )
        return 0 if not report["stop_expand"] else 1

    if args.cmd == "build-dataset":
        from enerzyme.data.dc_uma_cpcm.diagnostic import (
            build_residual_records,
            split_records,
            vacuum_source_allowed,
            write_residual_aselmdb,
            write_residual_jsonl,
        )
        from enerzyme.qm.orca_cpcm import load_json, load_manifest

        manifest = load_manifest(args.manifest)
        pilot = None
        if args.pilot_report and Path(args.pilot_report).is_file():
            pilot = load_json(args.pilot_report)
        gate = vacuum_source_allowed(None if args.allow_omol_vacuum else pilot)
        if not gate["allowed"] and not args.allow_omol_vacuum:
            print(json.dumps(gate, indent=2), file=sys.stderr)
            return 2
        records = build_residual_records(
            manifest,
            args.root,
            require_n=args.require_n,
            vacuum_gate=None if args.allow_omol_vacuum else pilot,
        )
        split = split_records(records)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_residual_jsonl(records, out / "residuals.jsonl")
        write_residual_jsonl(split["train"], out / "train.jsonl")
        write_residual_jsonl(split["validation"], out / "validation.jsonl")
        # ASE-LMDB write-mode mmap often fails on Lustre ("Function not implemented").
        # Diagnostic training reads residuals.jsonl; aselmdb is optional convenience only.
        aselmdb = {}
        for name, rows in (("train", split["train"]), ("validation", split["validation"])):
            path = out / f"{name}.aselmdb"
            try:
                aselmdb[name] = write_residual_aselmdb(rows, path)
            except Exception as exc:  # noqa: BLE001 — Lustre/LMDB host quirks
                for leftover in (path, Path(str(path) + "-lock")):
                    if leftover.exists():
                        try:
                            leftover.unlink()
                        except OSError:
                            pass
                aselmdb[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(
            json.dumps(
                {
                    "n_records": len(records),
                    "n_train": len(split["train"]),
                    "n_valid": len(split["validation"]),
                    "out": str(out),
                    "vacuum": gate,
                    "aselmdb": aselmdb,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "microbench":
        from enerzyme.models.esen.runtime import collect_runtime_versions, ensure_pinned_fairchem
        from enerzyme.tasks.cpcm_diagnostic import microbenchmark_second_order

        ensure_pinned_fairchem()
        model = _build_uma(args.adapter_mode, 0)
        report = microbenchmark_second_order(model, microbatch=int(args.microbatch))
        report["runtime"] = collect_runtime_versions()
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("device", "recommend_accumulate_grad_batches", "rows")}, indent=2, default=str))
        return 0

    from enerzyme.data.dc_uma_cpcm.diagnostic import decide_go_nogo, load_residual_jsonl
    from enerzyme.tasks.cpcm_diagnostic import run_seed_grid, train_baseline

    build_model = _build_toy if getattr(args, "toy", False) else _build_uma

    if args.cmd == "train":
        records = load_residual_jsonl(args.records)
        from enerzyme.data.dc_uma_cpcm.diagnostic import split_records

        split = split_records(records)
        out = args.out or (Path(args.records).parent.parent / "runs" / "diagnostic" / f"{args.baseline}_seed{args.seed}")
        report = train_baseline(
            build_model(args.baseline, int(args.seed)),
            split["train"],
            split["validation"],
            baseline=args.baseline,
            seed=int(args.seed),
            out_dir=out,
            accumulate_grad_batches=_accum_from_args(args),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            max_steps=args.max_steps,
        )
        print(json.dumps({"baseline": report["baseline"], "seed": report["seed"], "best_score": report["best_score"], "slices": report["slices"]}, indent=2, default=str))
        return 0

    if args.cmd == "run-grid":
        records = load_residual_jsonl(args.records)
        payload = run_seed_grid(
            build_model,
            records,
            out_root=args.out,
            accumulate_grad_batches=_accum_from_args(args),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            max_steps=args.max_steps,
        )
        print(json.dumps({"go": payload["gate"]["go"], "flags": payload["gate"]["flags"], "report_path": payload.get("report_path")}, indent=2))
        return 0 if payload["gate"]["go"] else 1

    runs_dir = Path(args.runs)
    run_payloads = []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        run_payloads.append(_load_json(path))
    gate = decide_go_nogo(run_payloads)
    payload = {"schema": "dc_uma_cpcm_diagnostic_report_v1", "runs": run_payloads, "gate": gate}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"go": gate["go"], "flags": gate["flags"], "n_runs": len(run_payloads)}, indent=2))
    return 0 if gate["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
