Reproducibility and Archiving
=============================

Practices for traceable MLFF projects.

Seeds
-----

:code:`Trainer.seed` controls Python, NumPy, and PyTorch RNGs for splitting and initialization.

:code:`Trainer.Splitter.seed` controls partition indices separately.

.. caution::
    CUDA kernels may still introduce nondeterminism even with fixed seeds.

Archive with each model
-----------------------

Minimum artifact set:

- :code:`config.yaml` from training output
- :code:`best/` and optionally :code:`last/` checkpoints
- Training :code:`train.yaml` (or Enerzymette-generated :code:`train.yaml` per iteration)
- :code:`processed_dataset_<hash>/` or instructions to rebuild from raw pickle
- Split index files if :code:`Splitter.save: true`

Document environment
--------------------

Record:

- Python, PyTorch, torch-scatter, Enerzyme commit hash
- Optional: NequIP, PLUMED, TeraChem versions
- Conda export or :code:`requirements.yaml` used

Enerzymette task archives
-------------------------

For external AL, archive the entire task root:

- :code:`config/` templates
- :code:`al.sh` launcher
- :code:`cluster.xyz`, :code:`cluster.mol`
- Each :code:`FFxx_*` iteration directory or symlink policy
- Reference PDB/SDF paths cited in configs

Pretrain chains
---------------

:code:`pretrain_path` links iterations. When publishing results, note the full chain :code:`FF00 → FF01 → ... → FFxx` and which checkpoint (:code:`best` vs :code:`last`) was used for production simulations.

Config drift
------------

Enerzymette rewrites paths each round. The **resolved** :code:`config.yaml` in :code:`FFxx_training/` is the authoritative record for that iteration, not the top-level template alone.

QM data provenance
------------------

Store:

- Annotate :code:`annotate.yaml` per round
- Fragment SDF inputs
- TeraChem template/settings reference
- :code:`terachem_timing` reports for failed-run triage
