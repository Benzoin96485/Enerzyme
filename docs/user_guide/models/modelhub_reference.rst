Modelhub Reference
==================

:code:`Modelhub` defines one or more force fields (:code:`FF` IDs), their architectures, layer stacks, losses, and per-model metrics.

Structure
---------

.. code-block:: yaml

    Modelhub:
        internal_FFs:
            FF02:
                suffix: '19'
                architecture: SpookyNet
                active: true
                pretrain_path: /path/to/previous/model
                build_params:
                    dim_embedding: 128
                    num_rbf: 16
                    max_Za: 86
                    cutoff_sr: 5.291772108
                    Hartree_in_E: 1
                    Bohr_in_R: 0.5291772108
                layers:
                    - name: RangeSeparation
                    - name: Core
                      params: {num_modules: 6}
                    - name: Force
                loss:
                    rmse:
                        E: 1
                        Fa: 0.91656181
                Metric:
                    E:
                        rmse: 1
                    Fa:
                        rmse: 0.91656181
        external_FFs:
            FF06:
                architecture: NequIP
                active: false

Key fields
----------

:code:`active`
    Only active models are trained, loaded for predict/simulate, or served by :code:`listen`.

:code:`suffix`
    Appended to checkpoint directory names (:code:`FF02_19/`).

:code:`pretrain_path`
    Directory with :code:`best/` or :code:`last/` weights to initialize training.

:code:`build_params`
    Shared hyperparameters passed to all layers unless overridden in :code:`params`.

:code:`loss`
    Weighted sum of registered loss types (:code:`rmse`, :code:`mae`, :code:`crps`, etc.).

:code:`Metric`
    Per-model validation metric for early stopping (:code:`judge_score`).

Output layout
-------------

After training with :code:`-o out/`:

.. code-block:: text

    out/
    ├── config.yaml
    ├── FF02_19/
    │   ├── best/model_best.pth
    │   └── last/model_last.pth
    └── logs/

For committees, expect :code:`model0`, :code:`model1`, ... under :code:`best/` and :code:`last/`.

Multiple models
---------------

You may define many :code:`FF` entries in one YAML but set :code:`active: true` only on those you want. Predict and simulate load all active models from the saved :code:`config.yaml`.
