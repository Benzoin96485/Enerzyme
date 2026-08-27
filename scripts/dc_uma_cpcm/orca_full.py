#!/usr/bin/env python3
"""Prepare, run, and summarize the 2,000-label DC-UMA C-PCM ORCA matrix.

Examples:

  python scripts/dc_uma_cpcm/orca_full.py prepare \\
      --manifest /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json \\
      --root /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full \\
      --import-success-from /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/diagnostic

  python scripts/dc_uma_cpcm/orca_full.py run --root ... --index $SLURM_ARRAY_TASK_ID

  python scripts/dc_uma_cpcm/orca_full.py summarize --root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import N_TOTAL_LABELS
    from enerzyme.qm.orca_cpcm import DEFAULT_MANIFEST, DEFAULT_ORCA_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="Write 2,000 recoverable C-PCM job directories")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "full")
    prepare.add_argument("--nprocs", type=int, default=32)
    prepare.add_argument("--nbo", action="store_true")
    prepare.add_argument("--require-n", type=int, default=N_TOTAL_LABELS)
    prepare.add_argument(
        "--import-success-from",
        type=Path,
        default=None,
        help="Copy successful status from another array (e.g. diagnostic) by job_id",
    )

    run = sub.add_parser("run", help="Run or resume one array task")
    run.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "full")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--skip-fd", action="store_true", default=True)

    summarize = sub.add_parser("summarize", help="Collect full-matrix ORCA quality report")
    summarize.add_argument("--root", type=Path, default=DEFAULT_ORCA_ROOT / "full")
    summarize.add_argument(
        "--report",
        type=Path,
        default=Path("/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/orca_full.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.qm.orca_cpcm import (
        ARRAY_MAP_NAME,
        import_successes_from,
        load_json,
        load_manifest,
        prepare_full_jobs,
        run_job,
        summarize_full,
    )

    if args.cmd == "prepare":
        manifest = load_manifest(args.manifest)
        array_map = prepare_full_jobs(
            manifest,
            args.root,
            nprocs=int(args.nprocs),
            nbo=bool(args.nbo),
            require_n=args.require_n,
        )
        imported = None
        if args.import_success_from is not None:
            imported = import_successes_from(args.root, args.import_success_from)
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "n_jobs": array_map["n_jobs"],
                    "subset": "full",
                    "imported": imported,
                },
                indent=2,
            )
        )
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

    report = summarize_full(args.root, report_path=args.report)
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


if __name__ == "__main__":
    sys.exit(main())
