Checkpoints and Resume
======================

Training lifecycle is controlled by :code:`Trainer` options and checkpoint files under each model directory.

Checkpoint layout
-----------------

.. code-block:: text

    FF02_suffix/
    ├── best/
    │   └── model_best.pth
    └── last/
        └── model_last.pth

Committee models use :code:`model0.pth`, :code:`model1.pth`, ... under :code:`best/` and :code:`last/`.

Resume modes
------------

:code:`Trainer.resume` (integer):

- :code:`0` — fresh training; may still load weights from :code:`pretrain_path` if set
- :code:`1` — load last checkpoint weights, restart epoch counter behavior per trainer logic
- :code:`2` — full resume: optimizer, scheduler, early-stop state, epoch

Implementation: :code:`enerzyme/tasks/trainer.py`.

Pretraining
-----------

:code:`Modelhub.internal_FFs.FFxx.pretrain_path` points to a previous run directory. Enerzyme resolves :code:`best/` or explicit paths via :code:`get_pretrain_path`.

Typical in iterative AL:

.. code-block:: yaml

    pretrain_path: /task/FF02-SpookyNet-18
    suffix: '19'

EMA
---

- :code:`use_ema: true`
- :code:`ema_decay: 0.999`
- :code:`ema_use_num_updates: true`

Exponential moving average weights can stabilize late training. Check whether your evaluation uses EMA weights in the saved checkpoint.

Lightning multi-GPU
-------------------

:code:`Trainer.lightning: true` enables PyTorch Lightning training (:code:`lightning_utils.py`). Use for multi-GPU scaling; verify batch size and learning rate relative to single-GPU runs.

Logs and config snapshot
------------------------

Each :code:`enerzyme train` invocation appends to :code:`logs/`. The resolved YAML is written as :code:`config.yaml` in the output directory — **archive this file** with checkpoints for reproducible predict/simulate.

Internal AL checkpoint
----------------------

Dataset active learning stores :code:`al_ckp.data` when :code:`active_learning_params.resume: true`. Distinct from model :code:`resume` modes above.
