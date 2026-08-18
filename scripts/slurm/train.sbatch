#!/bin/bash
# Multi-node / multi-GPU SLURM batch template for Enerzyme torch DDP.
#
# Replace every <PLACEHOLDER>. Do not commit site-specific defaults
# (account, partition, qos, constraint, scratch paths, module names,
# GPUs-per-node counts).
#
# Launch contract: one process per GPU. Enerzyme wraps DDP; it does not spawn.
# --gpus-per-task=1 (or a site-equivalent GPU bind) is required so each
# process sees exactly one GPU.
#
# If your center does not support --gpus-per-task, use the site equivalent,
# e.g. --gres=gpu:<GPUS_PER_NODE>. If the center requires --constraint,
# --gres, or --exclusive, add those flags from site docs — do not hard-code
# them here.

#SBATCH --job-name=enerzyme-train
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>       # some centers use --qos= instead of or with --partition
#SBATCH --nodes=<NNODES>
#SBATCH --ntasks-per-node=<GPUS_PER_NODE>
#SBATCH --gpus-per-task=1             # one visible GPU per process
#SBATCH --cpus-per-task=<CPUS_PER_GPU>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=enerzyme-%j.out
# #SBATCH --qos=<QOS>
# #SBATCH --constraint=<CONSTRAINT>

set -euo pipefail

# Site environment (module load / conda / container): fill in locally.
# source <CONDA_PREFIX>/etc/profile.d/conda.sh
# conda activate <ENV>

# Multi-task sbatch without srun fails fast. Single-task jobs may run
# `enerzyme train` directly.
srun enerzyme train -c train.yaml -o <OUTPUT_DIR>
