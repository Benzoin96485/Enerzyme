#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J dc-uma-orca-summary
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Collect the 24-job C-PCM pilot gate after the array finishes.
#   sbatch --dependency=afterany:<array_jobid> scripts/dc_uma_cpcm/submit_orca_summary.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
ORCA_ROOT="${ORCA_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/pilot}"
cd "$ENERZYME_ROOT"

source /global/common/software/m4787/.env
conda activate barw-dev
export PYTHONPATH="${ENERZYME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/dc_uma_cpcm/orca_pilot.py summarize \
  --root "$ORCA_ROOT" \
  --report /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/orca_pilot.json
