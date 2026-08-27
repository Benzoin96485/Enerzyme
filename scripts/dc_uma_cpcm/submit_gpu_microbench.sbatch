#!/bin/bash
#SBATCH -A m4787_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH -t 02:00:00
#SBATCH -J dc-uma-cpcm-bench
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Second-order adapter backprop microbenchmark (1–60 atoms, microbatch 1).
#   sbatch scripts/dc_uma_cpcm/submit_gpu_microbench.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{reports,runs}

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-na}"
nvidia-smi -L || true

python - <<'PY'
from enerzyme.models.esen.runtime import collect_runtime_versions, ensure_pinned_fairchem
ensure_pinned_fairchem()
import json
print(json.dumps(collect_runtime_versions(), indent=2, default=str))
PY

python scripts/dc_uma_cpcm/diagnostic.py microbench \
  --report /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/cpcm_microbench.json
