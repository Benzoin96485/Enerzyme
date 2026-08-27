#!/usr/bin/env python3
"""Build the immutable DC-UMA C-PCM parent split and 2,000-label manifest.

Reads candidate JSONL written by ``index_omol.py``. Reloads selected geometries
from train_4M to pick the two maximum heavy-atom-RMSD conformers per parent.

Example:

  python scripts/dc_uma_cpcm/sample_manifest.py \\
      --index /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests \\
      --src /global/homes/w/weiliang/m4787_cfs/data/omol25/train_4M \\
      --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_EXPERIMENTS, DEFAULT_TRAIN_4M

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "manifests")
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_TRAIN_4M))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS) / "manifests" / "dc_uma_cpcm_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-validate-full",
        action="store_true",
        help="Do not require the full 184-parent / 2000-label matrix (tests only).",
    )
    parser.add_argument(
        "--repair-from",
        type=Path,
        default=None,
        help=(
            "Replace open-shell parents in an existing frozen manifest from the "
            "pre-fixed reserve. Does not re-run farthest-point sampling."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.data.dc_uma_cpcm.constants import SEED
    from enerzyme.data.dc_uma_cpcm.index import annotate_s_squared, drop_open_shell_records
    from enerzyme.data.dc_uma_cpcm.sample import (
        annotate_manifest_s_squared,
        assign_parents,
        build_manifest,
        load_candidate_records,
        repair_closed_shell_parents,
        validate_manifest_counts,
        write_manifest,
    )

    seed = SEED if args.seed is None else int(args.seed)
    if args.repair_from is not None:
        manifest = json.loads(Path(args.repair_from).read_text(encoding="utf-8"))
        annotate_manifest_s_squared(manifest, args.src)
        manifest = repair_closed_shell_parents(manifest)
    else:
        records = load_candidate_records(args.index)
        annotate_s_squared(records, args.src)
        records, n_open = drop_open_shell_records(records)
        assigned = assign_parents(records)
        audit_path = Path(args.index) / "source_audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
        if audit is not None:
            audit = dict(audit)
            audit["n_open_shell_dropped"] = n_open
        manifest = build_manifest(assigned, src=str(args.src), seed=seed, audit=audit)
    validate_manifest_counts(manifest, strict_full_matrix=not args.skip_validate_full)
    write_manifest(manifest, args.out)
    summary = {
        "out": str(args.out),
        "seed": seed,
        "unseen_claim": manifest["unseen_claim"],
        "counts": manifest["counts"],
        "open_shell_replacements": (manifest.get("sampling") or {}).get("open_shell_replacements") or [],
    }
    summary_path = Path(args.out).with_name("manifest_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
