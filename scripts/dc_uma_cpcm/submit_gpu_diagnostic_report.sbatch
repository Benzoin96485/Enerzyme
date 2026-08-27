#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J dc-uma-cpcm-diag-sum
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Aggregate zero / x-only / Stage A × 3 seeds into the diagnostic go/no-go.
#   sbatch --dependency=afterany:<array_jobid> scripts/dc_uma_cpcm/submit_gpu_diagnostic_report.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

python scripts/dc_uma_cpcm/diagnostic.py report \
  --runs /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/diagnostic \
  --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/diagnostic_train.json
