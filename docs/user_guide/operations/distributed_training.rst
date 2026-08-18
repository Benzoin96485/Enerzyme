Distributed Training
====================

Multi-GPU and multi-node training uses the **native training loop** with
**torch** ``DistributedDataParallel``. An **external launcher**
(:code:`srun` or :code:`torchrun`) creates one process per GPU and sets
rank environment variables. Enerzyme only wraps DDP; it does **not**
spawn processes.

Single-process :code:`enerzyme train` (no launcher) is unchanged: the same
native loop, plus optional rank-0 TensorBoard.

Copy-paste examples below use placeholders only
(:code:`<ACCOUNT>`, :code:`<PARTITION>`, :code:`<NNODES>`,
:code:`<GPUS_PER_NODE>`, :code:`<CPUS_PER_GPU>`, :code:`<HH:MM:SS>`,
:code:`<OUTPUT_DIR>`, :code:`<ENV>`). Fill them from **your** center's
documentation. This page never names a specific HPC site, account,
partition, QoS, constraint, scratch path, or GPUs-per-node count.

Launch contract
---------------

Who does what:

+---------------------------+--------------------------------------------------+
| Component                 | Responsibility                                   |
+===========================+==================================================+
| :code:`srun` /            | Create **one process per GPU**, set rank env     |
| :code:`torchrun`          | vars (:code:`SLURM_PROCID` / :code:`RANK`, …)    |
+---------------------------+--------------------------------------------------+
| Enerzyme                  | Detect launch mode; bind one visible GPU;        |
|                           | :code:`init_process_group` + DDP wrap; rank-0    |
|                           | writes cache / split / :code:`config.yaml` /     |
|                           | checkpoints / logs / TensorBoard                 |
+---------------------------+--------------------------------------------------+

Supported modes:

+-----------------------+--------------------------------------------+
| Scenario              | Launch                                     |
+=======================+============================================+
| Single GPU (default)  | ``enerzyme train``                         |
+-----------------------+--------------------------------------------+
| Single-node multi-GPU | ``srun --ntasks-per-node=$NGPU`` or        |
| interactive           | ``torchrun --nproc_per_node``              |
+-----------------------+--------------------------------------------+
| Multi-node batch      | ``srun --ntasks-per-node=$NGPU``           |
|                       | (sets ``SLURM_NTASKS_PER_NODE``; prefer    |
|                       | this over bare ``srun -n``)                |
+-----------------------+--------------------------------------------+

YAML (DDP is enabled by the launcher, not a :code:`lightning:` flag):

.. code-block:: yaml

    Trainer:
        find_unused_parameters: false     # true only if some params skip the loss
        ddp_timeout_minutes: 30
        tensorboard: true                 # rank0 writes dump_dir/tb
        tensorboard_log_interval: 1       # batch train_loss/lr every N optimizer steps

:code:`sbatch` that runs bare :code:`enerzyme train` with **multiple
tasks** or **multiple allocated GPUs** **fails fast** with a generic
:code:`srun` / :code:`torchrun` example instead of hanging. A single-task
single-GPU SLURM job may still run :code:`enerzyme train` without
:code:`srun`.

Obsolete keys :code:`lightning`, :code:`accelerator`, :code:`devices`,
:code:`num_nodes`, :code:`strategy`, and :code:`precision` are ignored
(with a deprecation warning). Enerzyme does **not** use Lightning's
SLURM dialect (:code:`devices` ≠ GPUs per node). If you copy a Lightning
SLURM tutorial and set :code:`devices` to the node GPU count, that key
does nothing; launch one process per GPU instead.

Repository templates (placeholders only):

- :code:`scripts/slurm/train.sbatch` — multi-node batch
- :code:`scripts/slurm/interactive.sh` — :code:`salloc` + :code:`srun`

Site-specific options
---------------------

These flags are **not** given defaults. Add them when your center requires
them; omit them when it does not:

- :code:`--account=<ACCOUNT>`
- :code:`--partition=<PARTITION>`
- :code:`--qos=<QOS>` (some centers use this instead of or together with partition)
- :code:`--constraint=<CONSTRAINT>`

GPU binding must still give **one visible GPU per process**. See
:ref:`site-gpu-binding`.

Interactive debugging
---------------------

Request an allocation with generic resource counts only. Add site flags
from the previous section as needed:

.. code-block:: bash

    salloc \
        --nodes=<NNODES> \
        --ntasks-per-node=<GPUS_PER_NODE> \
        --gpus-per-task=1 \
        --cpus-per-task=<CPUS_PER_GPU> \
        --time=<HH:MM:SS>

Inside the allocation, activate your environment, then launch with
:code:`srun` (not a bare :code:`enerzyme train` when you requested
multiple tasks):

.. code-block:: bash

    # source <CONDA_PREFIX>/etc/profile.d/conda.sh
    # conda activate <ENV>

    srun --ntasks-per-node=<GPUS_PER_NODE> --gpus-per-task=1 \
        enerzyme train -c train.yaml -o <OUTPUT_DIR>

:code:`--gpus-per-task=1` (or the site equivalent) keeps each process on
one GPU.

sbatch multi-node
-----------------

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=enerzyme-train
    #SBATCH --account=<ACCOUNT>
    #SBATCH --partition=<PARTITION>
    #SBATCH --nodes=<NNODES>
    #SBATCH --ntasks-per-node=<GPUS_PER_NODE>
    #SBATCH --gpus-per-task=1
    #SBATCH --cpus-per-task=<CPUS_PER_GPU>
    #SBATCH --time=<HH:MM:SS>
    #SBATCH --output=enerzyme-%j.out

    # source <CONDA_PREFIX>/etc/profile.d/conda.sh
    # conda activate <ENV>

    srun enerzyme train -c train.yaml -o <OUTPUT_DIR>

If the center does not support :code:`--gpus-per-task`, use a site
equivalent such as :code:`--gres=gpu:<GPUS_PER_NODE>` and keep one task
per GPU.

Under :code:`srun` Enerzyme exports torchrun-style vars
(:code:`RANK` ← :code:`SLURM_PROCID`, :code:`WORLD_SIZE` ←
:code:`SLURM_STEP_NUM_TASKS` / :code:`SLURM_NTASKS` /
:code:`ntasks-per-node * nnodes`, :code:`MASTER_ADDR` = :code:`127.0.0.1` on
single-node jobs, otherwise the first host in :code:`SLURM_NODELIST`;
do **not** rely on :code:`SLURM_LAUNCH_NODE_IPADDR`, which is often the
login/submit host under interactive :code:`salloc`) and binds one
visible GPU per process. Set :code:`MASTER_ADDR` / :code:`MASTER_PORT`
yourself only if that export is wrong for your site. :code:`torchrun`
already sets rendezvous (see next section); multi-node torchrun uses
:code:`--rdzv_endpoint`.

Single-node torchrun
--------------------

Use :code:`torchrun` off SLURM, or inside an interactive allocation when
you prefer not to use :code:`srun`:

.. code-block:: bash

    torchrun --nproc_per_node=<GPUS_PER_NODE> \
        $(which enerzyme) train -c train.yaml -o <OUTPUT_DIR>

:code:`torchrun` sets :code:`RANK`, :code:`LOCAL_RANK`, :code:`WORLD_SIZE`,
and rendezvous. You normally do **not** export :code:`MASTER_ADDR` /
:code:`MASTER_PORT` yourself unless you pin a multi-node c10d endpoint.

Batch size and learning rate
----------------------------

:code:`Trainer.batch_size` is **per GPU**.

.. math::

    \text{global batch} = \texttt{batch\_size} \times \texttt{world\_size}

Learning-rate scaling is **your** choice. A common starting point is
linear scaling with warmup (global batch relative to your tuned single-GPU
run), then check validation :code:`_judge_score`. Do not assume Enerzyme
rescales LR automatically.

Dataloader :code:`num_workers` prefers :code:`SLURM_CPUS_PER_TASK`, not
:code:`SLURM_NTASKS`. Leave :code:`num_workers: -1` unless you need to pin
it.

TensorBoard
-----------

Rank 0 writes :code:`torch.utils.tensorboard.SummaryWriter` under
:code:`<model_dir>/tb/` (not :code:`lightning_logs/`). Single-GPU and DDP
use the same path. Disable with :code:`Trainer.tensorboard: false`.

.. code-block:: bash

    tensorboard --logdir <model_dir>/tb

Logged scalars:

- **batch** (:code:`global_step` = completed optimizer steps):
  :code:`train_loss`, :code:`lr`. Written after :code:`optimizer.step()`
  and before :code:`scheduler.step()`, so :code:`lr` is the value used
  that step. :code:`Trainer.tensorboard_log_interval` (default 1) can
  skip N steps. Under DDP, batch :code:`train_loss` is **rank 0's batch**
  only (no per-step all-reduce).
- **epoch**: :code:`train_loss_epoch` (all-reduced rank mean),
  :code:`val_loss`, each :code:`Metric`, :code:`_judge_score`, and
  :code:`best_score` / :code:`patience_wait` when validation runs.
  Validation metrics are gathered globally (SSE/count, then RMSE/MAE).

Common hangs
------------

+--------------------------------------+--------------------------------------+
| Symptom / mistake                    | What to do                           |
+======================================+======================================+
| :code:`sbatch` runs bare             | Always :code:`srun enerzyme train`   |
| :code:`enerzyme train` with          | (or :code:`torchrun`). Unlaunched    |
| multiple tasks or multiple GPUs      | multi-task / multi-GPU SLURM         |
|                                      | **fails fast**.                      |
+--------------------------------------+--------------------------------------+
| Each process sees many GPUs          | Bind one GPU per task                |
|                                      | (:code:`--gpus-per-task=1` or site   |
|                                      | equivalent).                         |
+--------------------------------------+--------------------------------------+
| :code:`CUDA_VISIBLE_DEVICES` shorter | Fail fast. Match                     |
| than :code:`local_rank`              | :code:`--nproc_per_node` / tasks to  |
|                                      | the visible GPU count.               |
+--------------------------------------+--------------------------------------+
| Train loader has 0 batches           | Need ≥ :code:`batch_size` samples    |
|                                      | per rank (about :code:`batch_size` × |
|                                      | world size). Smoke-test tiny splits  |
|                                      | on one GPU.                          |
+--------------------------------------+--------------------------------------+
| All ranks write HDF5 cache / split / | Rank 0 writes; others handshake then |
| :code:`config.yaml` / shared log     | wait (no 30-minute cap on the HDF5   |
|                                      | build itself). Delete a leftover     |
|                                      | half-written cache and retry.        |
+--------------------------------------+--------------------------------------+
| NCCL timeout / invalid device        | Raise :code:`ddp_timeout_minutes`;   |
| ordinal / silent stall               | check GPU bind; with one visible GPU |
|                                      | Enerzyme sets :code:`NCCL_P2P_DISABLE`|
|                                      | and :code:`NCCL_SHM_DISABLE`.        |
|                                      | Use :code:`NCCL_DEBUG=WARN`.         |
+--------------------------------------+--------------------------------------+
| Unused-parameter DDP error on        | Set :code:`find_unused_parameters:   |
| modular / gated stacks               | true` (default is :code:`false`).    |
+--------------------------------------+--------------------------------------+

Debugging
---------

.. code-block:: bash

    export NCCL_DEBUG=WARN

- Increase :code:`Trainer.ddp_timeout_minutes` if init or the first
  validation gather is slow on a shared filesystem.
- Smoke-test with :code:`max_epochs: 2` before a full run. Tiny splits
  belong on one GPU; DDP needs at least one full :code:`batch_size` per
  rank.
- Confirm world size in the log: :code:`N` ranks, one :code:`model_best.pth`
  (or :code:`model{i}_best.pth` per committee member).

Resume and checkpoints
----------------------

Only **rank 0** writes checkpoints, :code:`config.yaml`, TensorBoard, and
the shared log. Other ranks must not rewrite the same files.

:code:`Trainer.resume`:

- :code:`0` — fresh run; may still load :code:`pretrain_path` weights
- :code:`1` — load last checkpoint weights; restart epoch counting
- :code:`2` — full resume: optimizer, scheduler, early-stop state, epoch

Patience always comes from the **current YAML**. Do not edit checkpoint
files on disk to change early-stop state.

Old Lightning :code:`.pth` files (those containing
:code:`pytorch-lightning_version`) are still loadable via
:code:`resume` / :code:`pretrain_path`. Enerzyme converts them without
importing Lightning.

See :doc:`/user_guide/models/checkpoints_resume`.

Committee
---------

Committee members train **serially**. Each member uses the **full DDP
world** (data-parallel over GPUs). Enerzyme does **not** shard committee
members across ranks.

Checkpoint names stay :code:`model{i}_best.pth` /
:code:`model{i}_last.pth`, same as single-GPU.

Active learning
---------------

Internal dataset AL still uses :code:`max_epoch_per_iter` (the loop
stops when :code:`epoch_in_iter` reaches that cap). The AL **picking
strategy** is unchanged.

.. _site-gpu-binding:

Site GPU binding (appendix)
---------------------------

Centers expose GPUs in different ways. The invariant is **one visible GPU
per Enerzyme process**:

+----------------------------------+------------------------------------------+
| Binding style                    | Typical use                              |
+==================================+==========================================+
| :code:`--gpus-per-task=1`        | Preferred when SLURM supports it         |
+----------------------------------+------------------------------------------+
| :code:`--gres=gpu:<N>`           | Older or site-specific GRES; still run   |
|                                  | :code:`N` tasks per node, one GPU each   |
+----------------------------------+------------------------------------------+
| :code:`CUDA_VISIBLE_DEVICES`     | Manual / non-SLURM bind; each process    |
|                                  | should see a **single** index            |
+----------------------------------+------------------------------------------+

Do not let every rank see every GPU and rely on a trainer :code:`devices`
setting to split them. Enerzyme binds one visible GPU per process itself.
