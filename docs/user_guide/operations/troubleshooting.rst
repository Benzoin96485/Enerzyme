Troubleshooting
===============

Common issues grouped by stage.

Installation
------------

**:code:`torch-scatter` import error**
    Install the wheel matching your PyTorch, CUDA, Python, and platform from https://data.pyg.org/whl/

**Optional model fails to import**
    Install NequIP, XequiNet, py-plumed, etc. per :doc:`/getting_started/installation`.

Data
----

**:code:`ModuleNotFoundError: numpy._core`**
    Pickle created with NumPy 2.x loaded under NumPy 1.x. Re-export dataset or align versions.

**Missing field / shape mismatch**
    Check Datahub :code:`features`/:code:`targets` mapping against actual pickle keys and shapes (:code:`N`, :code:`Ra`, :code:`Za`).

**Stale preprocessing cache**
    Delete :code:`processed_dataset_<hash>/` or change a hash-affecting option intentionally, then rerun with :code:`preload: false` once.

Training
--------

**NaN loss**
    Lower learning rate; check :code:`negative_gradient`; verify energy/force units and :code:`atomic_energy` references.

**CUDA OOM**
    Reduce :code:`batch_size`; use :code:`float32`; disable :code:`data_in_memory`; shrink model or cutoff.

**Poor force accuracy, good energy**
    Revisit force loss weight and unit conversion (:doc:`/user_guide/concepts/units_and_fields`).

**Early stopping too aggressive**
    Increase :code:`patience`; align :code:`Metric` weights with :code:`loss`.

Distributed (torch DDP)
-----------------------

**Job hangs under SLURM**
    Almost always a launch-contract error, not the model. Do **not** run
    bare :code:`enerzyme train` inside a **multi-task** or **multi-GPU**
    :code:`sbatch` / :code:`salloc`. Use :code:`srun` (or
    :code:`torchrun`), one process per GPU. Unlaunched multi-task /
    multi-GPU SLURM should **fail fast** with an :code:`srun` example.

**Each process sees multiple GPUs / world size is wrong**
    Bind one GPU per task (:code:`--gpus-per-task=1` or site equivalent).
    Enerzyme binds a single visible GPU per process; it does not spawn.

**Copied a Lightning SLURM tutorial with** :code:`devices` **= GPUs per node**
    Enerzyme does **not** use Lightning. YAML :code:`devices` /
    :code:`strategy` / :code:`num_nodes` are ignored. Launch one process
    per GPU — see :doc:`/user_guide/operations/distributed_training`.

**Stall while writing HDF5 cache, split, or** :code:`config.yaml`
    Only rank 0 may write shared artifacts. Delete a half-written
    :code:`processed_dataset_<hash>/` and retry.

**NCCL timeout / invalid device ordinal**
    Set :code:`NCCL_DEBUG=WARN`; raise :code:`Trainer.ddp_timeout_minutes`;
    confirm GPU bind and that you are not nesting :code:`srun`.
    With :code:`--gpus-per-task=1` (each rank sees only :code:`cuda:0`),
    Enerzyme disables NCCL P2P and SHM automatically. Override with
    :code:`NCCL_P2P_DISABLE=0` / :code:`NCCL_SHM_DISABLE=0` only if your
    site's NCCL can peer through remapped device ids.

**Unused-parameter / DDP deadlock on modular stacks**
    Keep :code:`find_unused_parameters: true` unless every parameter is in
    the loss.

**TensorBoard missing / looking for** :code:`lightning_logs/`
    Logs go to :code:`<model_dir>/tb/`. View with
    :code:`tensorboard --logdir <model_dir>/tb`. Disable with
    :code:`Trainer.tensorboard: false`.

Full launch contract, templates, and hang table:
:doc:`/user_guide/operations/distributed_training`.

Simulation
----------

**PLUMED instability**
    Fix :code:`UNITS` line; verify :code:`Hartree_in_E`, :code:`time_step`, :code:`fs_in_t`.

**Wrong atom in constraint/scan**
    Check :code:`idx_start_from` (0 vs 1).

**NEB fails to interpolate**
    Verify frame count vs :code:`num_images`; relax endpoints if needed.

QM annotation
-------------

**TeraChem not found**
    Module load / PATH / license.

**Unfinished jobs in batch**
    Run :code:`enerzymette terachem_timing`; inspect scratch and :code:`keep_stdout` (legacy :code:`keep_output` is still accepted).

Active learning
---------------

**No uncertainty in extract**
    Enable shallow ensemble or committee; set :code:`non_target_features` in predict path.

**Confusing internal vs Enerzymette AL**
    Internal AL never creates new structures; Enerzymette AL requires template configs and iteration directories (:doc:`/user_guide/workflows/active_learning`).

Documentation build
-------------------

Sphinx autosummary warnings for moved classes (e.g. :code:`ASECalculator` in :code:`calculator` not :code:`simulator`) do not affect runtime CLI behavior.
