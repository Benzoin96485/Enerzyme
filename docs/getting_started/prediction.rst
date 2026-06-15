Evaluating a Trained Model
==========================

After training, use :code:`enerzyme predict` to run inference on a test set and report metrics. The reference config is :code:`enerzyme/config/predict.yaml`.

Basic command
-------------

.. code-block:: bash

    enerzyme predict -c predict.yaml -o results/ -m model_dir/ -mc train.yaml

Arguments:

- :code:`-c` / :code:`--config_path` — prediction config (test :code:`Datahub`, optional :code:`Metric`)
- :code:`-m` / :code:`--model_dir` — directory containing trained checkpoints (from :code:`enerzyme train -o`)
- :code:`-mc` / :code:`--model_config_path` — training config used to build the model (defaults to :code:`model_dir/config.yaml` if omitted)
- :code:`-o` / :code:`--output_dir` — where CSV summaries and per-model pickles are written

Configuration
-------------

Prediction YAML overrides the training :code:`Datahub` for an external test set while keeping transforms and neighbor-list settings consistent with training.

.. code-block:: yaml

    Datahub:
        data_path: "test.pkl"
        data_format: pickle
        features:
            Ra: coord
            Za: atom_type
            Q:
            N:
        preload: true
        targets:
            E: energy
            Fa: grad
            Qa: chrg
            M2: dipole
    Metric:
        E:
            rmse: 1
        Fa:
            rmse: 52.917721
    Trainer:
        non_target_features:
            - E_var
            - Qa_var

Key points:

- :code:`data_path` — external test set; if omitted, data comes from the training config
- :code:`Metric` — overrides per-model metrics from training for this evaluation
- :code:`non_target_features` — extra outputs saved to prediction pickles (e.g. committee uncertainty :code:`E_var`, :code:`Qa_var` when using shallow ensembles)

Simple prediction mode
----------------------

For inference only (no metric reporting), pass :code:`--simple_predict`:

.. code-block:: bash

    enerzyme predict -c predict.yaml -o results/ -m model_dir/ -mc train.yaml -s

Outputs
-------

Enerzyme loads all **active** models from the model config, runs prediction, and writes:

- Per-model prediction pickles under each model subfolder in :code:`output_dir`
- A summary CSV in :code:`output_dir` with aggregated metrics when :code:`Metric` is defined

.. note::
    Uncertainty fields require a committee-trained model or another UQ-capable setup. See :doc:`active_learning`.

Next steps
----------

- Run constrained optimization or MD: :doc:`simulation`
- Rank uncertain structures for relabeling: :doc:`fragment_extraction`
