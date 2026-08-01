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
    Run :code:`enerzymette terachem_timing`; inspect scratch and :code:`keep_output`.

Active learning
---------------

**No uncertainty in extract**
    Enable shallow ensemble or committee; set :code:`non_target_features` in predict path.

**Confusing internal vs Enerzymette AL**
    Internal AL never creates new structures; Enerzymette AL requires template configs and iteration directories (:doc:`/user_guide/workflows/active_learning`).

Documentation build
-------------------

Sphinx autosummary warnings for moved classes (e.g. :code:`ASECalculator` in :code:`calculator` not :code:`simulator`) do not affect runtime CLI behavior.
