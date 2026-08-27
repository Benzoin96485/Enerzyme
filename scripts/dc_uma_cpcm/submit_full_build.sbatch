#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J dc-uma-full-build
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Pair 2,000 successful full-matrix ORCA jobs with OMol vacuum → JSONL residuals.
#   sbatch scripts/dc_uma_cpcm/submit_full_build.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{manifests/full,reports,runs}

source /global/common/software/m4787/.env
conda activate barw-dev
export PYTHONPATH="${ENERZYME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/dc_uma_cpcm/full.py build-dataset \
  --manifest /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json \
  --root /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full \
  --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/full \
  --pilot-report /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/orca_pilot.json \
  --require-n 2000
