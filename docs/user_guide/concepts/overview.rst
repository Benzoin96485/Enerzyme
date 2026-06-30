Overview
========

Enerzyme is a YAML-driven toolkit for training neural network potentials (NNPs), running ASE-based simulations, and closing active-learning loops with QM relabeling. The CLI is the primary user interface; Python APIs exist but are documented separately in :doc:`/api`.

Workflow map
------------

.. code-block:: text

    raw QM / structures
        -> Datahub (preprocess, cache)
        -> Modelhub (build model)
        -> Trainer (fit)
        -> predict / simulate
        -> (optional) extract -> annotate -> merge -> retrain

For enzymatic exploration, the outer loop is often managed by `Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ rather than a single :code:`enerzyme train` job. See :doc:`/user_guide/workflows/active_learning`. A full methyltransferase workflow is documented in :code:`example/NNP4MTase`.

CLI command map
-----------------

- :code:`train` — Fit one or more force fields
- :code:`predict` — Evaluate or infer on a dataset
- :code:`simulate` — ASE workflows: opt, MD, scan, NEB, PLUMED
- :code:`extract` — Pick uncertain fragments from predictions
- :code:`collect` — Preprocess and split only (no training)
- :code:`annotate` — Batch QM labeling from SDF/pickle suppliers
- :code:`bond` — Assign bonds to PDB clusters
- :code:`listen` — Start HTTP prediction server
- :code:`request` — Client request (e.g. ORCA ExtOpt bridge)
- :code:`kill` — Shut down a listening server

Configuration-driven design
---------------------------

Nearly every command takes a YAML file (:code:`-c`) and an output directory (:code:`-o`). Training and inference additionally use a **model directory** (:code:`-m`) containing checkpoints and a resolved :code:`config.yaml`.

Reference configs live in :code:`enerzyme/config/`. After training, treat the saved :code:`config.yaml` in the output directory as the canonical model configuration for :code:`predict` and :code:`simulate`.

Reading path
------------

1. **Getting Started** — install, first dataset, first train, first predict/simulate.
2. **User Guide (this section)** — schema, tuning, advanced workflows.
3. **API Reference** — module and class documentation.
