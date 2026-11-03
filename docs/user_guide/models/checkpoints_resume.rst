Checkpoints and Resume
======================

Training lifecycle is controlled by :code:`Trainer` options and checkpoint files under each model directory.

Checkpoint layout
-----------------

Checkpoints are written **directly** in each model directory (no :code:`best/` or
:code:`last/` subfolders). Filenames encode the preference and optional committee rank:

.. code-block:: text

    FF02-SpookyNet/
    ├── model_best.pth
    ├── model_last.pth
    └── model_epoch=10.pth          # optional, when dump_interval > 0

Committee members use a numeric rank in the filename prefix:

.. code-block:: text

    FF02-SpookyNet/
    ├── model0_best.pth
    ├── model0_last.pth
    ├── model1_best.pth
    └── model1_last.pth

Legacy Lightning runs may also have versioned names such as
:code:`model_best-v1.pth`; :code:`get_pretrain_path` prefers the appropriate
:code:`best` / :code:`last` file in that directory.

Resume modes
------------

:code:`Trainer.resume` (integer):

- :code:`0` — fresh training; may still load weights from :code:`pretrain_path` if set
- :code:`1` — load last checkpoint weights, restart epoch counter behavior per trainer logic
- :code:`2` — full resume: optimizer, scheduler, early-stop state, epoch

Implementation: :code:`enerzyme/tasks/trainer.py`.

Pretraining
-----------

:code:`Modelhub.internal_FFs.FFxx.pretrain_path` points to a previous run directory (or an
explicit :code:`.pth` file). Enerzyme resolves :code:`model*_best.pth` / :code:`model*_last.pth`
in that directory via :code:`get_pretrain_path` (preference :code:`best` or :code:`last`).

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

Distributed DDP
---------------

Launch with :code:`srun` / :code:`torchrun` (one process per GPU). Enerzyme
wraps torch DDP; it does not spawn. Only **rank 0** writes checkpoints,
:code:`config.yaml`, TensorBoard (:code:`<model_dir>/tb/`), and the shared
log.

:code:`resume: 2` still means full resume (optimizer, scheduler,
early-stop state, epoch). Patience comes from the current YAML, not from
editing checkpoint files on disk.

Old Lightning :code:`.pth` files (key :code:`pytorch-lightning_version`)
are converted on load. You do **not** need the Lightning package to resume
or use them as :code:`pretrain_path`.

Committee members train **serially**, each using the full DDP world.
Filenames stay :code:`model{i}_best.pth` / :code:`model{i}_last.pth`.

:code:`batch_size` is per GPU. See :doc:`/user_guide/operations/distributed_training`.

Logs and config snapshot
------------------------

Each :code:`enerzyme train` invocation appends to :code:`logs/`. The resolved YAML is written as :code:`config.yaml` in the output directory — **archive this file** with checkpoints for reproducible predict/simulate.

Internal AL checkpoint
----------------------

Dataset active learning stores :code:`al_ckp.data` when :code:`active_learning_params.resume: true`. Distinct from model :code:`resume` modes above.
