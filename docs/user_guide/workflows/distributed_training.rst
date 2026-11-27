Distributed supervised training
===============================

Enerzyme uses PyTorch DistributedDataParallel (DDP) for supervised training
when an external launcher starts several processes. Each process holds one
model replica, processes its own mini-batch and participates in gradient
reduction. A plain invocation runs the native single-process trainer.
Existing configurations can select the Lightning trainer with
``Trainer.lightning: true``.

Launching a job
---------------

Start from a working supervised training configuration. For a global batch of
32 structures on four GPUs, use the following trainer settings:

.. code-block:: yaml

   Trainer:
     cuda: true
     lightning: false
     batch_size: 8
     num_workers: 2

Put the dataset, configuration and output directory on storage visible at the
same path on every node. In a single-node allocation with two visible GPUs,
use one launcher process:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 --module enerzyme.cli train \
     -c train.yaml -o results

On a Slurm cluster, request the site's partition and resources. The following
step uses two nodes and two GPUs on each node:

.. code-block:: bash

   srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1 \
     --kill-on-bad-exit=1 enerzyme train -c train.yaml -o results

Enerzyme reads the rank and device assignments from the launcher. Under Slurm
it derives the rendezvous host from the step node list and the port from the
job ID. Set a common ``MASTER_ADDR`` and ``MASTER_PORT`` to override these values.
Independent simultaneous steps in one allocation should use distinct ports.
For CPU training use ``cuda: false`` in the native trainer, which selects Gloo.

Batch size, data and metrics
----------------------------

``batch_size`` is the number of structures **per process**. For fixed-global-batch
comparisons divide the original batch size by the number of processes and
keep the learning rate and optimizer schedule fixed. Native DDP shuffles the
training data each epoch and drops the trailing incomplete global batch.
Choose a training split with at least one complete global batch.
``num_workers`` sets the DataLoader worker count per process. Zero loads batches
in the training process. The automatic setting uses the CPU allocation per
task and caps the worker count at eight.

Rank zero prepares the shared cache, publishes a saved split, and writes the
configuration and model artifacts. For periodic structures use
``Datahub.neighbor_list: cutoff`` and provide the cell and periodic flags.
Validation and prediction shard the original sample indices without padding.
For energy and force objectives, the reduction weights each term by its own
number of observations. An energy counts once per structure and a force
component counts once per atom and direction. Force RMSE for reporting applies
the square root per component and then averages the component scores.
A custom distributed loss supplies ``components(output, target)``, yielding
:class:`~enerzyme.models.loss.LossComponent` entries with each term's local
mean, observation count, weight and optional RMSE epsilon.

Prediction and recovery
-----------------------

Use the same launcher for prediction. A prediction config specifying a
``Datahub`` selects a new dataset:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 --module enerzyme.cli predict \
     -m results -c predict.yaml -o predictions

Rank zero writes results in the original sample order. To resume a native
run, use the existing output directory, set ``Trainer.resume: 2`` and launch
again. Use the same number of workers and GPUs, dataset, split and training
schedule to reproduce an uninterrupted run. Native last checkpoints capture
optimizer, scheduler, EMA, sampler seed, completed epoch, optimizer step,
early-stopping state and each rank's random number states. Existing
checkpoints can supply the fields recorded by their original trainer.
See :doc:`../models/checkpoints_resume` for checkpoint selection and pretraining.

Lightning supports supervised multi-GPU configurations through the same CLI.
The Slurm example also works with Lightning when the scheduler exposes just
one GPU to each task. On installations that expose all allocated GPUs per
node, use one task per GPU and the local rank selects its device.
Active learning and committee workflows run in a single process.

Comparing training throughput
-----------------------------

The checkout includes :file:`scripts/bench_distributed_training.py`. It times
optimizer updates through the public trainer using a fixed SchNet checkpoint,
energy and force loss, and either a global batch of 32 twenty-atom molecules
or four periodic 256-atom cells. Use the same physical allocation and Python
environment for each checkout. For example, inside a two-GPU Slurm allocation:

.. code-block:: bash

   srun -n2 --gpu-bind=none python scripts/bench_distributed_training.py \
     --source-tree /path/to/baseline --lightning --system crystals --output runs/crystals-A1
   srun -n2 --gpu-bind=none python scripts/bench_distributed_training.py \
     --source-tree . --system crystals --output runs/crystals-B1

Repeat these commands in reverse order with fresh output directories (B2, A2).
Repeat the four runs with ``--system molecules`` and directories
``runs/molecules-A1`` through ``runs/molecules-A2``. Run
``python scripts/bench_distributed_training.py --compare runs`` to verify
all ranks and hardware assignments and print the paired comparisons.
For the old single-GPU baseline omit ``--lightning``. Each process records
its measured wall time per optimizer step and peak allocated CUDA memory
after ten warm-up steps. Use the maximum across ranks for each run, then
compare the two-run medians on the same topology. Fixed-global-batch scaling
efficiency is the single-GPU step time divided by the product of the GPU
count and the corresponding multi-GPU step time.
