#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=110G
#SBATCH -t 08:00:00
#SBATCH --array=0-1999%8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J dc-uma-orca-full
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%A_%a.err

# Recoverable 2,000-label C-PCM array (train/probe/validation/test).
# Jobs whose status.json is already success (including imports from diagnostic) are skipped.
#
# Infra failures (MPI segfault / bus error in Startup, GUESS, SCF, PROPINT, PROPERTIES)
# are retried in-process (fresh guess) and again on array resubmit. Keep array throttle
# at %8 to reduce node-level MPI contention; %16 saw ~1.2% first-pass infra loss.
#
#   sbatch scripts/dc_uma_cpcm/submit_orca_full.sbatch
#   sbatch --array=0-1999 scripts/dc_uma_cpcm/submit_orca_full.sbatch   # resume
#   bash scripts/dc_uma_cpcm/submit_orca_full_retry.sh                  # failed only

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
ORCA_ROOT="${ORCA_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/full}"
MANIFEST="${MANIFEST:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json}"
DIAG_ROOT="${DIAG_ROOT:-/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/orca/diagnostic}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{orca/full,reports,runs}

source /global/common/software/m4787/.env
conda activate barw-dev
export PYTHONPATH="${ENERZYME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

module use /global/common/software/m4787/ModuleFiles
module unload cray-mpich >/dev/null 2>&1 || true
module load openmpi/4.1.8
module load orca/6.1.1
export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"
export OMPI_MCA_btl_vader_single_copy_mechanism="${OMPI_MCA_btl_vader_single_copy_mechanism:-none}"
export OMPI_MCA_opal_warn_on_missing_libcuda="${OMPI_MCA_opal_warn_on_missing_libcuda:-0}"
export OMPI_MCA_hwloc_base_binding_policy="${OMPI_MCA_hwloc_base_binding_policy:-none}"
export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=none

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} array=${SLURM_ARRAY_TASK_ID:-na}"
echo "python=$(command -v python)"
echo "orca=$(command -v orca)"
echo "ORCA_ROOT=${ORCA_ROOT}"

MAP="${ORCA_ROOT}/array_map.json"
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 && ! -f "$MAP" ]]; then
  python scripts/dc_uma_cpcm/orca_full.py prepare \
    --manifest "$MANIFEST" \
    --root "$ORCA_ROOT" \
    --nprocs 32 \
    --import-success-from "$DIAG_ROOT"
fi
for _ in $(seq 1 600); do
  if [[ -f "$MAP" ]]; then
    break
  fi
  sleep 2
done
if [[ ! -f "$MAP" ]]; then
  echo "missing ${MAP}; run: python scripts/dc_uma_cpcm/orca_full.py prepare --root ${ORCA_ROOT}" >&2
  exit 2
fi

python scripts/dc_uma_cpcm/orca_full.py run \
  --root "$ORCA_ROOT" \
  --index "${SLURM_ARRAY_TASK_ID}"
