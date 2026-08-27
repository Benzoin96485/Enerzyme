#!/usr/bin/env python3
"""Aggregate Stage A three-seed metrics into the Stage B GB go/no-go report.

Does not run ORCA or retrain. Reads production metrics.json, residual JSONL,
and optional ORCA Lowdin/dipole fields.

  python scripts/dc_uma_cpcm/error_analysis.py
  python scripts/dc_uma_cpcm/error_analysis.py --out-md .../stage_a_error_analysis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_EXPERIMENTS
    from enerzyme.data.dc_uma_cpcm.diagnostic import DEFAULT_FULL_DATA, DEFAULT_FULL_ORCA, DEFAULT_FULL_RUNS

    reports = Path(DEFAULT_EXPERIMENTS) / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_FULL_RUNS)
    parser.add_argument("--records", type=Path, default=DEFAULT_FULL_DATA / "residuals.jsonl")
    parser.add_argument("--orca-root", type=Path, default=DEFAULT_FULL_ORCA)
    parser.add_argument("--orca-report", type=Path, default=reports / "orca_full.json")
    parser.add_argument("--out-md", type=Path, default=reports / "stage_a_error_analysis.md")
    parser.add_argument("--out-json", type=Path, default=reports / "stage_a_error_analysis.json")
    parser.add_argument("--date", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.data.dc_uma_cpcm.error_analysis import run_stage_a_error_analysis

    payload = run_stage_a_error_analysis(
        runs_dir=args.runs,
        records_path=args.records,
        orca_root=args.orca_root,
        orca_report_path=args.orca_report,
        date=args.date or date.today().isoformat(),
    )
    out_json = dict(payload)
    markdown = out_json.pop("markdown")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_json, indent=2, default=str) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown, encoding="utf-8")
    gate = payload["stage_b"]
    print(
        json.dumps(
            {
                "decision": gate["decision"],
                "go": gate["go"],
                "flags": gate["flags"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
