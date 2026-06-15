Prediction and Evaluation
=========================

:code:`enerzyme predict` runs batch inference and optional metric reporting. Entry: :code:`enerzyme/predict.py` (:code:`FFPredict`).

Command
-------

.. code-block:: bash

    enerzyme predict -c predict.yaml -o results/ -m model_dir/ -mc config.yaml

- :code:`-m` — directory with checkpoints and usually :code:`config.yaml`
- :code:`-mc` — model configuration (defaults to :code:`model_dir/config.yaml`)
- :code:`-c` — prediction overrides for Datahub and Metric
- :code:`-s` — :code:`--simple_predict` (no metric aggregation)

Config overrides
----------------

Prediction YAML typically overrides:

.. code-block:: yaml

    Datahub:
        data_path: external_test.pkl
        data_format: pickle
        features:
            Ra: coord
            Za: atom_type
        preload: true
        targets:
            E: energy
            Fa: grad
    Metric:
        E:
            rmse: 1
        Fa:
            rmse: 52.917721
    Trainer:
        non_target_features:
            - E_var
            - Fa_var

Neighbor list and :code:`transforms` should stay consistent with training unless you intentionally change preprocessing.

Test set sources
----------------

1. **External file** — set :code:`Datahub.data_path` in predict config
2. **Training split** — omit override; uses partitions from training config
3. **Simulation trajectory pickle** — e.g. :code:`plumed.traj.pkl` from MD for downstream extract

Outputs
-------

- Per-active-model prediction pickles under :code:`output_dir`
- Summary CSV when :code:`Metric` is defined
- Uncertainty columns when model supports them and :code:`non_target_features` is set

Simple predict mode
-------------------

Use :code:`-s` for geometry-only inference without ground-truth targets or metric CSV. Useful inside pipelines that only need forces/energies on new structures.

Multi-model predict
-------------------

All :code:`active: true` models in :code:`Modelhub` are evaluated. Disable unused models in the saved training config or maintain separate model directories.
