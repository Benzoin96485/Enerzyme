Dataset Splitting
=================

Partitions are configured under :code:`Trainer.Splitter` and implemented in :code:`enerzyme/tasks/splitter.py` (:code:`RandomSplit`).

Partition semantics
-------------------

Common partition names:

- :code:`training` — used for gradient updates
- :code:`validation` — early stopping and :code:`judge_score`
- :code:`test` — held-out evaluation at end of training
- :code:`withheld` — pool for **internal** active learning (not used in forward pass until picked)

Old-style splitter
------------------

Single dataset; ratios match :code:`parts` order:

.. code-block:: yaml

    Trainer:
        Splitter:
            method: random
            parts:
                - training
                - validation
                - test
            ratios:
                - 0.7
                - 0.1
                - 0.2
            seed: 42
            save: true
            preload: true

Ratios can be floats in :code:`(0, 1)` or positive integers (absolute counts).

Multi-dataset splitter
----------------------

Each part can draw from named datasets with its own ratio:

.. code-block:: yaml

    Trainer:
        Splitter:
            method: random
            parts:
                - name: training
                  dataset: training
                  ratio: 1.0
                - name: validation
                  dataset: validation
                  ratio: 1.0

Or combine multiple sources into one part via :code:`sources`:

.. code-block:: yaml

    parts:
        - name: training
          sources:
              - dataset: dataset_a
                ratio: 0.8
              - dataset: dataset_b
                ratio: 0.2

Withheld pool for internal AL
-----------------------------

.. code-block:: yaml

    Splitter:
        parts:
            - training
            - withheld
        ratios:
            - 0.01
            - 0.99

Pair with :code:`Trainer.active_learning_params.data_source: withheld`. This is **not** the same as Enerzymette's external AL loop; see :doc:`/user_guide/workflows/active_learning`.

Persistence
-----------

- :code:`save: true` — write split indices next to the processed dataset
- :code:`preload: true` — reload indices when the splitter hash matches
- :code:`seed` — reproducible random splits
