#!/bin/bash
#SBATCH -A m4787_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH -t 06:00:00
#SBATCH --array=0-8
#SBATCH -J dc-uma-cpcm-diag
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.err

# Three baselines × three seeds on the 240-job diagnostic residual set.
# Array index: 0-2 zero, 3-5 x_only, 6-8 stage_a.
#
#   sbatch scripts/dc_uma_cpcm/submit_gpu_diagnostic.sbatch
#
# After the array finishes:
#   python scripts/dc_uma_cpcm/diagnostic.py report

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{reports,runs/diagnostic,manifests}

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BASELINES=(zero zero zero x_only x_only x_only stage_a stage_a stage_a)
SEEDS=(20260817 20260818 20260819 20260817 20260818 20260819 20260817 20260818 20260819)
BASELINE="${BASELINES[$INDEX]}"
SEED="${SEEDS[$INDEX]}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} array=${INDEX} baseline=${BASELINE} seed=${SEED}"
nvidia-smi -L || true

RECORDS="${RECORDS:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/diagnostic/residuals.jsonl}"
if [[ ! -f "$RECORDS" ]]; then
  python scripts/dc_uma_cpcm/diagnostic.py build-dataset
  RECORDS="/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/diagnostic/residuals.jsonl"
fi

python scripts/dc_uma_cpcm/diagnostic.py train \
  --records "$RECORDS" \
  --baseline "$BASELINE" \
  --seed "$SEED" \
  --out "/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/diagnostic/${BASELINE}_seed${SEED}" \
  --microbench /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/cpcm_microbench.json
