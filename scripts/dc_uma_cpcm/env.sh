# Pinned fairchem / UMA environment for DC-UMA C-PCM.
# Source after activating barw-dev. Does not modify fairchem.
#
#   source /global/common/software/m4787/.env
#   conda activate barw-dev
#   source scripts/dc_uma_cpcm/env.sh

FAIRCHEM_ROOT="${FAIRCHEM_ROOT:-/global/cfs/cdirs/m4787/src/fairchem}"
FAIRCHEM_SRC="${FAIRCHEM_ROOT}/src"
export PYTHONPATH="${FAIRCHEM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-/global/cfs/cdirs/m4787/models/fairchem}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export UMA_CHECKPOINT="${UMA_CHECKPOINT:-${FAIRCHEM_CACHE_DIR}/models--facebook--UMA/snapshots/be2896459a03fcde05e20d2fcefd11f450601fce/checkpoints/uma-s-1p2.pt}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
