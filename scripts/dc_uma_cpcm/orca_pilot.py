#!/usr/bin/env python3
"""Prepare, run, and summarize DC-UMA C-PCM ORCA 6.1.1 jobs.

Examples:

  python scripts/dc_uma_cpcm/orca_pilot.py prepare \\
      --manifest /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json \\
      --root /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/pilot

  python scripts/dc_uma_cpcm/orca_pilot.py run --root ... --index $SLURM_ARRAY_TASK_ID

  python scripts/dc_uma_cpcm/orca_pilot.py summarize --root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.qm.orca_cpcm import DEFAULT_MANIFEST, DEFAULT_ORCA_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="Write 24 recoverable ORCA job directories")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "pilot")
    prepare.add_argument("--nprocs", type=int, default=32)
    prepare.add_argument("--nbo", action="store_true")

    run = sub.add_parser("run", help="Run or resume one array task")
    run.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "pilot")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--skip-fd", action="store_true")

    summarize = sub.add_parser("summarize", help="Collect pilot quality-gate report")
    summarize.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "pilot")
    summarize.add_argument("--report", type=Path, default=None)
    summarize.add_argument("--src", type=Path, default=None, help="OMol train_4M root for S² lookup on old job meta")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_TRAIN_4M
    from enerzyme.qm.orca_cpcm import (
        ARRAY_MAP_NAME,
        load_json,
        load_manifest,
        prepare_pilot_jobs,
        run_job,
        summarize_pilot,
    )

    if args.cmd == "prepare":
        manifest = load_manifest(args.manifest)
        array_map = prepare_pilot_jobs(
            manifest,
            args.root,
            nprocs=int(args.nprocs),
            nbo=bool(args.nbo),
        )
        print(json.dumps({"root": str(args.root), "n_jobs": array_map["n_jobs"]}, indent=2))
        return 0

    if args.cmd == "run":
        array_map = load_json(Path(args.root) / ARRAY_MAP_NAME)
        jobs = {int(item["index"]): item for item in array_map["jobs"]}
        if int(args.index) not in jobs:
            print(f"array index {args.index} not in {args.root}/{ARRAY_MAP_NAME}", file=sys.stderr)
            return 2
        status = run_job(Path(args.root) / jobs[int(args.index)]["relpath"], skip_fd=bool(args.skip_fd))
        print(json.dumps({k: status.get(k) for k in ("job_id", "status", "attempt", "energy_ev", "error")}, indent=2))
        return 0 if status.get("status") == "success" else 1

    report = summarize_pilot(
        args.root,
        report_path=args.report,
        omol_src=args.src if args.src is not None else DEFAULT_TRAIN_4M,
    )
    print(
        json.dumps(
            {
                "n_jobs": report["n_jobs"],
                "n_success": report["n_success"],
                "first_pass_rate": report["first_pass_rate"],
                "stop_expand": report["stop_expand"],
                "vacuum_systematic_bias": report["vacuum_systematic_bias"],
                "recommend_recompute_all_vacuum": report["recommend_recompute_all_vacuum"],
            },
            indent=2,
        )
    )
    return 0 if not report["stop_expand"] else 1


if __name__ == "__main__":
    sys.exit(main())
