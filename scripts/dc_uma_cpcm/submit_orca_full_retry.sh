#!/usr/bin/env bash
# Retry failed full-matrix ORCA array tasks and chain a summarize job.
#
#   bash scripts/dc_uma_cpcm/submit_orca_full_retry.sh
#   ORCA_ROOT=... THROTTLE=8 bash scripts/dc_uma_cpcm/submit_orca_full_retry.sh

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
ORCA_ROOT="${ORCA_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full}"
THROTTLE="${THROTTLE:-8}"
SBATCH="${SBATCH:-sbatch}"

cd "$ENERZYME_ROOT"

FAILED_CSV="$(
  ORCA_ROOT="$ORCA_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ORCA_ROOT"])
amap = json.loads((root / "array_map.json").read_text())
failed = []
for item in amap["jobs"]:
    status_path = root / item["relpath"] / "status.json"
    if not status_path.is_file():
        failed.append(int(item["index"]))
        continue
    status = json.loads(status_path.read_text())
    if status.get("status") != "success":
        failed.append(int(item["index"]))
print(",".join(str(i) for i in sorted(failed)))
PY
)"

if [[ -z "$FAILED_CSV" ]]; then
  echo "No failed ORCA tasks under ${ORCA_ROOT}; nothing to retry."
  exit 0
fi

ARRAY_SPEC="${FAILED_CSV}%${THROTTLE}"
echo "Retrying ${ARRAY_SPEC} (failed indices under ${ORCA_ROOT})"

RETRY_OUT="$(
  ORCA_ROOT="$ORCA_ROOT" "$SBATCH" \
    --array="${ARRAY_SPEC}" \
    --job-name=dc-uma-orca-full-retry \
    scripts/dc_uma_cpcm/submit_orca_full.sbatch
)"
echo "$RETRY_OUT"
RETRY_ID="$(sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"$RETRY_OUT")"
if [[ -z "$RETRY_ID" ]]; then
  echo "Could not parse retry job id from sbatch output" >&2
  exit 2
fi

SUM_OUT="$(
  "$SBATCH" \
    --dependency="afterany:${RETRY_ID}" \
    scripts/dc_uma_cpcm/submit_orca_full_summary.sbatch
)"
echo "$SUM_OUT"
