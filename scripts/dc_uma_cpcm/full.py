#!/usr/bin/env python3
"""Build the 2,000-label residual set and train Stage A on the full matrix.

Examples:

  python scripts/dc_uma_cpcm/full.py build-dataset
  python scripts/dc_uma_cpcm/full.py train --baseline stage_a --seed 20260817
  python scripts/dc_uma_cpcm/full.py train --baseline stage_b --seed 20260818 --warm-start .../adapter.pt
  python scripts/dc_uma_cpcm/full.py eval --checkpoint .../adapter.pt --records .../residuals.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import (
        DEFAULT_EXPERIMENTS,
        DIAGNOSTIC_SEEDS,
        N_TOTAL_LABELS,
    )
    from enerzyme.data.dc_uma_cpcm.diagnostic import DEFAULT_FULL_DATA, DEFAULT_FULL_ORCA
    from enerzyme.qm.orca_cpcm import DEFAULT_MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-dataset", help="Pair full-matrix C-PCM jobs with OMol vacuum")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--root", type=Path, default=DEFAULT_FULL_ORCA)
    build.add_argument("--out", type=Path, default=DEFAULT_FULL_DATA)
    build.add_argument(
        "--pilot-report",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "reports" / "orca_pilot.json",
    )
    build.add_argument("--allow-omol-vacuum", action="store_true")
    build.add_argument("--require-n", type=int, default=N_TOTAL_LABELS)

    train = sub.add_parser("train", help="Train one baseline × seed on the full residual set")
    train.add_argument("--records", type=Path, default=DEFAULT_FULL_DATA / "residuals.jsonl")
    train.add_argument("--baseline", required=True, choices=["zero", "x_only", "stage_a", "stage_b"])
    train.add_argument("--seed", type=int, default=DIAGNOSTIC_SEEDS[0])
    train.add_argument("--out", type=Path, default=None)
    train.add_argument("--accumulate-grad-batches", type=int, default=None)
    train.add_argument(
        "--microbench",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "reports" / "cpcm_microbench.json",
    )
    train.add_argument("--max-epochs", type=int, default=200)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--toy", action="store_true")
    train.add_argument("--warm-start", type=Path, default=None)
    train.add_argument("--lowdin-aux-weight", type=float, default=None)
    train.add_argument("--orca-root", type=Path, default=DEFAULT_FULL_ORCA)
    train.add_argument("--stage-a-checkpoint", type=Path, default=None)

    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint on probe/test (and validation)")
    eval_p.add_argument("--records", type=Path, default=DEFAULT_FULL_DATA / "residuals.jsonl")
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.add_argument("--baseline", default="stage_a", choices=["zero", "x_only", "stage_a", "stage_b"])
    eval_p.add_argument("--seed", type=int, default=DIAGNOSTIC_SEEDS[0])
    eval_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write eval JSON (default: checkpoint dir / eval.json)",
    )
    eval_p.add_argument("--toy", action="store_true")
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
    from enerzyme.tasks.cpcm_diagnostic import set_seed

    set_seed(int(seed))
    return DCUMACPCMWrapper(
        checkpoint_path=resolve_uma_checkpoint(),
        task_name=OMOL_TASK_NAME,
        adapter_mode=baseline,
        frozen=True,
    )


def _build_toy(baseline: str, seed: int):
    from enerzyme.tasks.cpcm_diagnostic import ToyCpcmModel, set_seed

    set_seed(int(seed))
    return ToyCpcmModel(adapter_mode=baseline)


def _write_dataset(records, out: Path, gate: dict) -> dict:
    from enerzyme.data.dc_uma_cpcm.diagnostic import (
        split_records,
        write_residual_aselmdb,
        write_residual_jsonl,
    )

    split = split_records(records)
    out.mkdir(parents=True, exist_ok=True)
    write_residual_jsonl(records, out / "residuals.jsonl")
    for name in ("train", "validation", "probe", "test"):
        write_residual_jsonl(split[name], out / f"{name}.jsonl")
    aselmdb = {}
    for name in ("train", "validation", "probe", "test"):
        path = out / f"{name}.aselmdb"
        try:
            aselmdb[name] = write_residual_aselmdb(split[name], path)
        except Exception as exc:  # noqa: BLE001 — Lustre/LMDB host quirks
            for leftover in (path, Path(str(path) + "-lock")):
                if leftover.exists():
                    try:
                        leftover.unlink()
                    except OSError:
                        pass
            aselmdb[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "n_records": len(records),
        "n_train": len(split["train"]),
        "n_valid": len(split["validation"]),
        "n_probe": len(split["probe"]),
        "n_test": len(split["test"]),
        "out": str(out),
        "vacuum": gate,
        "aselmdb": aselmdb,
    }


def main() -> int:
    args = _parse_args()

    if args.cmd == "build-dataset":
        from enerzyme.data.dc_uma_cpcm.diagnostic import (
            build_residual_records,
            vacuum_source_allowed,
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
            subset="full",
        )
        payload = _write_dataset(records, Path(args.out), gate)
        print(json.dumps(payload, indent=2))
        return 0

    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_EXPERIMENTS, LOWDIN_AUX_WEIGHT
    from enerzyme.data.dc_uma_cpcm.diagnostic import (
        attach_lowdin_charges,
        decide_stage_b_keep_drop,
        load_residual_jsonl,
        split_records,
    )
    from enerzyme.tasks.cpcm_diagnostic import evaluate_records, train_baseline

    build_model = _build_toy if getattr(args, "toy", False) else _build_uma

    if args.cmd == "train":
        records = load_residual_jsonl(args.records)
        if args.baseline == "stage_b":
            records = attach_lowdin_charges(records, getattr(args, "orca_root", None))
        split = split_records(records)
        out = args.out or (
            Path(DEFAULT_EXPERIMENTS) / "runs" / "full" / f"{args.baseline}_seed{args.seed}"
        )
        aux = getattr(args, "lowdin_aux_weight", None)
        if aux is None:
            aux = LOWDIN_AUX_WEIGHT if args.baseline == "stage_b" else 0.0
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
            eval_splits={"probe": split.get("probe") or [], "test": split.get("test") or []},
            warm_start=getattr(args, "warm_start", None),
            lowdin_aux_weight=float(aux),
        )
        report["schema"] = "dc_uma_cpcm_full_run_v1"
        if args.baseline == "stage_b" and split.get("test"):
            sa_ckpt = getattr(args, "stage_a_checkpoint", None)
            if sa_ckpt is not None and Path(sa_ckpt).is_file():
                import torch

                sa_model = build_model("stage_a", int(args.seed))
                ckpt = torch.load(sa_ckpt, map_location="cpu", weights_only=False)
                if hasattr(sa_model, "adapter") and ckpt.get("adapter_state_dict"):
                    sa_model.adapter.load_state_dict(ckpt["adapter_state_dict"])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sa_model = sa_model.to(device)
                sa_test = evaluate_records(sa_model, split["test"], device)
                report["stage_b_gate"] = decide_stage_b_keep_drop(
                    {"test": sa_test["summary"], "test_slices": sa_test["slices"]},
                    {
                        "test": report.get("test") or {},
                        "test_slices": report.get("test_slices") or {},
                    },
                )
        Path(report["checkpoint"]).parent.joinpath("metrics.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        printed = {
            "baseline": report["baseline"],
            "seed": report["seed"],
            "best_score": report["best_score"],
            "n_train": report["n_train"],
            "n_valid": report["n_valid"],
            "validation": report.get("validation"),
            "probe": report.get("probe"),
            "test": report.get("test"),
            "checkpoint": report["checkpoint"],
        }
        if report.get("stage_b_gate"):
            printed["stage_b_gate"] = report["stage_b_gate"]
        print(json.dumps(printed, indent=2, default=str))
        return 0

    # eval
    import torch

    records = load_residual_jsonl(args.records)
    split = split_records(records)
    model = build_model(args.baseline, int(args.seed))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if hasattr(model, "adapter") and ckpt.get("adapter_state_dict"):
        model.adapter.load_state_dict(ckpt["adapter_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    payload = {
        "schema": "dc_uma_cpcm_full_eval_v1",
        "baseline": args.baseline,
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint),
        "splits": {},
    }
    for role in ("validation", "probe", "test"):
        rows = split.get(role) or []
        if rows:
            packed = evaluate_records(model, rows, device)
            payload["splits"][role] = {"summary": packed["summary"], "slices": packed.get("slices")}
    out = args.out or Path(args.checkpoint).with_name("eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), "splits": {k: v["summary"] for k, v in payload["splits"].items()}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
