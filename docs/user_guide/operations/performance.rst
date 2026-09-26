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

:code:`fast_validation: true` (default)
    For shallow ensembles, validation during fitting computes forces of the ensemble-mean energy in one
    backward pass instead of one per member. It is disabled automatically when a variance-based loss
    (:code:`nll`, :code:`nll_var_only`, :code:`crps`) acts on :code:`Fa` or the :code:`Monitor` collects
    :code:`Fa_*` terms; test-set and active-learning predictions always keep per-member forces.

:code:`Optimizer.batched: true` (default for :code:`name: Muon` on a single device)
    Same update and optimizer state as muon-optimizer's :code:`SingleDeviceMuonWithAuxAdam`, with
    Newton-Schulz batched over equally shaped weights and foreach AdamW updates. Checkpointed optimizer
    states are interchangeable; set :code:`batched: false` to use the original implementation.

:code:`fast_validation: true` (default)
    For shallow ensembles, validation during fitting computes forces of the ensemble-mean energy in one
    backward pass instead of one per member. It is disabled automatically when a variance-based loss
    (:code:`nll`, :code:`nll_var_only`, :code:`crps`) acts on :code:`Fa` or the :code:`Monitor` collects
    :code:`Fa_*` terms; test-set and active-learning predictions always keep per-member forces.

:code:`Optimizer.batched: true` (default for :code:`name: Muon` on a single device)
    Same update and optimizer state as muon-optimizer's :code:`SingleDeviceMuonWithAuxAdam`, with
    Newton-Schulz batched over equally shaped weights and foreach AdamW updates. Checkpointed optimizer
    states are interchangeable; set :code:`batched: false` to use the original implementation.

Simulation
----------

:code:`dtype: float64` vs :code:`float32`
    Optimization and NEB often need :code:`float64`; MD may use :code:`float32` if stable.

:code:`cuda: true`
    GPU inference for ML calculator; external QC still runs on its own resources.

:code:`log_interval`
    Reduce trajectory I/O frequency for long MD.

:code:`neighbor_list: full` for large systems
    Supported SpookyNet models avoid the :math:`O(N^2)` pair list: short-range pairs are found on the GPU and
    the all-pairs Coulomb term is summed densely in blocks. With shallow ensembles and no
    :code:`uncertainty_calculator`, forces of the ensemble-mean energy are obtained in a single backward pass.
    Use :code:`neighbor_list: full_pairs` to reproduce the explicit all-pairs evaluation.

Neighbor list memory
--------------------

Full neighbor lists scale as :math:`O(N^2)`. For large enzyme clusters:

- Use :code:`compressed` in training data
- Avoid storing redundant frames
- Consider smaller QM fragment training sets via extract/annotate rather than full-cluster every step

When to use server mode
-----------------------

:code:`listen` amortizes model load cost over hundreds of short requests (e.g. ORCA optimization loops). For one-off batch predict, use :code:`enerzyme predict`.
