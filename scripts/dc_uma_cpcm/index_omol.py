#!/usr/bin/env python3
"""Stream-scan OMol25 train_4M and write candidate JSONL plus a source audit.

Example:

  python scripts/dc_uma_cpcm/index_omol.py \\
      --src /global/homes/w/weiliang/m4787_cfs/data/omol25/train_4M \\
      --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests \\
      --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from enerzyme.data.dc_uma_cpcm.constants import DEFAULT_EXPERIMENTS, DEFAULT_TRAIN_4M

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_TRAIN_4M))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_EXPERIMENTS) / "manifests")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.data.dc_uma_cpcm.index import scan_train_4m

    audit = scan_train_4m(
        args.src,
        args.out,
        workers=max(1, int(args.workers)),
        max_shards=args.max_shards,
    )
    print(json.dumps({k: audit[k] for k in ("n_shards", "n_total", "n_passed", "reject_reasons", "observed_data_ids")}, indent=2))
    print(f"wrote {args.out / 'source_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
