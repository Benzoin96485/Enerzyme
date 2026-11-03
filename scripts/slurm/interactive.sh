#!/bin/bash
# Interactive SLURM template for Enerzyme torch DDP.
#
# Replace every <PLACEHOLDER>. Do not commit site-specific defaults
# (account, partition, qos, constraint, scratch paths, module names,
# GPUs-per-node counts).
#
# Launch contract: one process per GPU. Enerzyme wraps DDP; it does not spawn.
# --gpus-per-task=1 (or a site-equivalent GPU bind) is required so each
# process sees exactly one GPU.
#
# Workflow:
#   1. Request an allocation with salloc (resource counts only below).
#   2. Inside the allocation shell, activate your env, then srun train.
#
# Site options — fill from your center's documentation and add to salloc
# when required:
#   --account=<ACCOUNT>
#   --partition=<PARTITION>   # some centers use --qos= instead of or with --partition
#   --qos=<QOS>
#   --constraint=<CONSTRAINT>

# --- 1. Request an interactive allocation ---------------------------------
salloc \
    --nodes=<NNODES> \
    --ntasks-per-node=<GPUS_PER_NODE> \
    --gpus-per-task=1 \
    --cpus-per-task=<CPUS_PER_GPU> \
    --time=<HH:MM:SS>

# --- 2. Inside the allocation shell ---------------------------------------
# source <CONDA_PREFIX>/etc/profile.d/conda.sh
# conda activate <ENV>
#
# Multi-task allocation: do not run `enerzyme train` without srun
# (fails fast). Single-task salloc may run enerzyme train directly.
#
# srun --ntasks-per-node=<GPUS_PER_NODE> --gpus-per-task=1 \
#     enerzyme train -c train.yaml -o <OUTPUT_DIR>
