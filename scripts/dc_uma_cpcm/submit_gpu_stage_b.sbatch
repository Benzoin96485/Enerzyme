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
#SBATCH -J dc-uma-cpcm-stage-b
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Stage B 1-seed warm-start from Stage A seed 20260818. Expand to 3 seeds only
# if charged test dE MAE improves ≥10% and full-test dFa MAE degrades ≤5%.
#
#   sbatch scripts/dc_uma_cpcm/submit_gpu_stage_b.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{reports,runs/full}

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

SEED="${SEED:-20260818}"
BASELINE="stage_b"
STAGE_A_CKPT="${STAGE_A_CKPT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/full/stage_a_seed${SEED}/adapter.pt}"
ORCA_ROOT="${ORCA_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} baseline=${BASELINE} seed=${SEED}"
nvidia-smi -L || true

RECORDS="${RECORDS:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/full/residuals.jsonl}"
if [[ ! -f "$RECORDS" ]]; then
  echo "missing ${RECORDS}; run submit_full_build.sbatch first" >&2
  exit 2
fi
if [[ ! -f "$STAGE_A_CKPT" ]]; then
  echo "missing Stage A checkpoint ${STAGE_A_CKPT}" >&2
  exit 2
fi

python scripts/dc_uma_cpcm/full.py train \
  --records "$RECORDS" \
  --baseline "$BASELINE" \
  --seed "$SEED" \
  --warm-start "$STAGE_A_CKPT" \
  --stage-a-checkpoint "$STAGE_A_CKPT" \
  --orca-root "$ORCA_ROOT" \
  --lowdin-aux-weight 0.05 \
  --out "/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/full/${BASELINE}_seed${SEED}" \
  --microbench /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/cpcm_microbench.json
