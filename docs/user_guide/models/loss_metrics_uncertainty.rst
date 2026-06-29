Loss, Metrics, and Uncertainty
==============================

Training objectives and validation scores are configured separately but should use consistent unit weighting.

Loss types
----------

Registered in :code:`enerzyme/models/loss.py`:

- :code:`rmse` / :code:`mse` — root mean square error per target
- :code:`mae` — mean absolute error
- :code:`crps` — continuous ranked probability score (useful with shallow ensembles)
- :code:`nll` — negative log-likelihood where supported
- Architecture-specific terms (e.g. PhysNet :code:`nh_penalty`)

Example:

.. code-block:: yaml

    loss:
        crps:
            E: 1
        rmse:
            Fa: 0.91656181
            M2: 0.32731016
            Q: 0.1

Metrics and early stopping
--------------------------

Per-model :code:`Metric` blocks define the validation :code:`judge_score` — a weighted sum of metric terms. Early stopping uses :code:`Trainer.patience` on this score.

.. code-block:: yaml

    Metric:
        E:
            rmse: 1
        Fa:
            rmse: 0.91656181

Global :code:`Metrics` in older configs may still appear; per-model :code:`Metric` in Modelhub is preferred in current :code:`train.yaml`.

Shallow ensemble uncertainty
----------------------------

:code:`ShallowEnsembleReduce` with :code:`shallow_ensemble_size` > 1 in :code:`Core` trains multiple forward passes. Configure variance outputs:

.. code-block:: yaml

    - name: ShallowEnsembleReduce
      params:
          reduce_mean: [E, Fa, Qa, M2, Q]
          var: [E, Fa]
          eval_only: true

At predict time, request non-target features:

.. code-block:: yaml

    Trainer:
        non_target_features:
            - E_var
            - Fa_var
            - Qa_var

Committee training
------------------

Set :code:`Trainer.committee_size` > 1 to train independent models (:code:`FF_committee`). Uncertainty comes from disagreement across committee members rather than a single shallow ensemble. Used heavily in :code:`active_learning_train.yaml`.

UDD at simulation time
----------------------

Hybrid simulation can apply uncertainty-driven debiasing via :code:`uncertainty_calculator: UDD` in :code:`Simulation` (see :code:`uma.yaml` and :doc:`/user_guide/workflows/enhanced_sampling_plumed`). This is separate from training-time :code:`E_var` / :code:`Fa_var`.

Unit weights
------------

See :doc:`/user_guide/concepts/units_and_fields`. Mismatched force weights are a common cause of poor force accuracy despite good energy loss.
