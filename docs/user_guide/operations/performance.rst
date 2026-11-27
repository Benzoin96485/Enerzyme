Performance Tuning
==================

Guidelines for large datasets and expensive simulations.

Data loading
------------

:code:`preload: true`
    Reuse HDF5 cache across runs with identical preprocessing hash.

:code:`compressed: true`
    Share static fields across frames of the same stoichiometry.

:code:`max_memory`
    Increase HDF5 cache (GB) for random-access heavy workloads.

:code:`num_workers`
    Dataloader parallelism; increase until CPU saturates; watch memory.

:code:`data_in_memory`
    Load full training set into RAM — fast but memory-intensive.

Training
--------

:code:`batch_size`
    Largest value that fits GPU memory.

:code:`dtype: float32`
    Default for speed; :code:`float64` for numerical experiments.

:code:`neighbor_list: full` (precomputed)
    Faster epochs if memory allows; otherwise on-the-fly lists.

:code:`lightning: true`
    Multi-GPU training when single-process :code:`cuda` is insufficient.

Single-device Muon
~~~~~~~~~~~~~~~~~~

For models with Muon parameter groups, enable batched updates in the training YAML:

.. code-block:: yaml

   Trainer:
     Optimizer:
       name: Muon
       batched: true

The optimizer groups equal-shaped hidden matrices on the same device and uses
foreach updates for the auxiliary Adam parameters. It reads and writes the same
optimizer state as the single-device Muon implementation. Batched bfloat16
matrix products may round differently. The default is ``batched: false``.
If a process group is initialized when Enerzyme builds the optimizer, the trainer
uses distributed Muon.

Simulation
----------

:code:`dtype: float64` vs :code:`float32`
    Optimization and NEB often need :code:`float64`; MD may use :code:`float32` if stable.

:code:`cuda: true`
    GPU inference for ML calculator; external QC still runs on its own resources.

:code:`log_interval`
    Reduce trajectory I/O frequency for long MD.

For repeated ASE evaluations with ``neighbor_list: full``, the calculator keeps
the molecular pair indices and other static inputs on the model device. Each
step uses new coordinates and a fresh force graph. The calculation rebuilds
the inputs when the species, charge, spin, device or precision changes.
This reuse applies to non-periodic full graphs.

Measuring training and prediction
---------------------------------

To compare the shared batch and geometry paths against a baseline checkout,
run the following from the candidate checkout:

.. code-block:: bash

   PYTHONPATH=. python scripts/bench_shared_pipeline.py \
     --reference-tree /path/to/baseline --device cuda --dtype float32 \
     --output model-pipeline.json

The benchmark uses identical weights and a fixed batch of 16 molecules for
SchNet and PaiNN. It alternates baseline and candidate paths on one device,
checks energies, forces and participating parameter gradients, and reports
wall time for batch prediction and force-loss backward. Prediction includes
host result collection. Training includes collation, transfer, forward and
backward. The optimizer step is measured separately when evaluating Muon.
The CPU and CUDA measurements, hardware details and per-preset comparisons
are recorded in ``test/fixtures/validation/shared_performance.json``.

Neighbor list memory
--------------------

Full neighbor lists scale as :math:`O(N^2)`. For large enzyme clusters:

- Use :code:`compressed` in training data
- Avoid storing redundant frames
- Consider smaller QM fragment training sets via extract/annotate rather than full-cluster every step

When to use server mode
-----------------------

:code:`listen` amortizes model load cost over hundreds of short requests (e.g. ORCA optimization loops). For one-off batch predict, use :code:`enerzyme predict`.
