#!/bin/bash
#SBATCH -A m4787_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -J dc-uma-cpcm-full
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.err

# Stage A × 3 seeds on the 2,000-label full residual set (train 896 / val 504).
# Array index: 0-2 → seeds 20260817 / 20260818 / 20260819.
#
#   sbatch --dependency=afterok:<build_jobid> scripts/dc_uma_cpcm/submit_gpu_full.sbatch
#   sbatch scripts/dc_uma_cpcm/submit_gpu_full.sbatch   # if residuals.jsonl already exists

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{reports,runs/full,manifests/full}

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(20260817 20260818 20260819)
SEED="${SEEDS[$INDEX]}"
BASELINE="${BASELINE:-stage_a}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} array=${INDEX} baseline=${BASELINE} seed=${SEED}"
nvidia-smi -L || true

RECORDS="${RECORDS:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/full/residuals.jsonl}"
if [[ ! -f "$RECORDS" ]]; then
  echo "missing ${RECORDS}; run submit_full_build.sbatch first" >&2
  exit 2
fi

python scripts/dc_uma_cpcm/full.py train \
  --records "$RECORDS" \
  --baseline "$BASELINE" \
  --seed "$SEED" \
  --out "/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/full/${BASELINE}_seed${SEED}" \
  --microbench /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/cpcm_microbench.json
