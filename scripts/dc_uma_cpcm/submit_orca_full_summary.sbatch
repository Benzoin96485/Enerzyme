#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J dc-uma-orca-full-sum
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Collect the 2,000-label C-PCM array after it finishes.
#   sbatch --dependency=afterany:<array_jobid> scripts/dc_uma_cpcm/submit_orca_full_summary.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
ORCA_ROOT="${ORCA_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full}"
cd "$ENERZYME_ROOT"

source /global/common/software/m4787/.env
conda activate barw-dev
export PYTHONPATH="${ENERZYME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/dc_uma_cpcm/orca_full.py summarize \
  --root "$ORCA_ROOT" \
  --report /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/orca_full.json
