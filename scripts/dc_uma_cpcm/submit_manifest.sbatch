#!/bin/bash
#SBATCH -A m4787
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=110G
#SBATCH -t 02:00:00
#SBATCH -J dc-uma-manifest
#SBATCH -o /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.out
#SBATCH -e /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/runs/%x-%j.err

# Scan OMol25 train_4M and freeze the DC-UMA C-PCM split/label manifest.
#   sbatch scripts/dc_uma_cpcm/submit_manifest.sbatch

set -euo pipefail

ENERZYME_ROOT="${ENERZYME_ROOT:-/global/cfs/cdirs/m4787/src/Enerzyme}"
cd "$ENERZYME_ROOT"

mkdir -p /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/{manifests,reports,runs}

source /global/common/software/m4787/.env
conda activate barw-dev
# shellcheck source=/dev/null
source "${ENERZYME_ROOT}/scripts/dc_uma_cpcm/env.sh"
export PYTHONPATH="${ENERZYME_ROOT}:${PYTHONPATH}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "python=$(command -v python)"

python scripts/dc_uma_cpcm/index_omol.py \
  --src /global/homes/w/weiliang/m4787_cfs/data/omol25/train_4M \
  --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests \
  --workers 16

python scripts/dc_uma_cpcm/sample_manifest.py \
  --index /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests \
  --src /global/homes/w/weiliang/m4787_cfs/data/omol25/train_4M \
  --out /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/dc_uma_cpcm_manifest.json

cp /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/manifest_summary.json \
   /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/manifest_summary.json
cp /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/manifests/source_audit.json \
   /global/cfs/cdirs/m4787/experiments/dc_uma_cpcm/reports/source_audit.json
