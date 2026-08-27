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
#SBATCH -J dc-uma-omol-gate
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# GPU smoke + UMA OMol wrapper parity.
#   sbatch scripts/dc_uma_cpcm/submit_gpu_wrapper.sbatch

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
echo "python=$(command -v python)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "FAIRCHEM_CACHE_DIR=${FAIRCHEM_CACHE_DIR}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "UMA_CHECKPOINT=${UMA_CHECKPOINT}"
nvidia-smi -L || true

python - <<'PY'
from enerzyme.models.esen.runtime import collect_runtime_versions, ensure_pinned_fairchem
ensure_pinned_fairchem()
import json
print(json.dumps(collect_runtime_versions(), indent=2, default=str))
PY

python scripts/dc_uma_cpcm/smoke_omol_official.py \
  --device cuda \
  --report /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/omol_official_smoke.json

python -m pytest test/test_uma_registry.py test/test_uma_omol.py -q \
  --basetemp=/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/pytest
