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

Simulation
----------

:code:`dtype: float64` vs :code:`float32`
    Optimization and NEB often need :code:`float64`; MD may use :code:`float32` if stable.

:code:`cuda: true`
    GPU inference for ML calculator; external QC still runs on its own resources.

:code:`log_interval`
    Reduce trajectory I/O frequency for long MD.

Neighbor list memory
--------------------

Full neighbor lists scale as :math:`O(N^2)`. For large enzyme clusters:

- Use :code:`compressed` in training data
- Avoid storing redundant frames
- Consider smaller QM fragment training sets via extract/annotate rather than full-cluster every step

When to use server mode
-----------------------

:code:`listen` amortizes model load cost over hundreds of short requests (e.g. ORCA optimization loops). For one-off batch predict, use :code:`enerzyme predict`.
