Server Mode
===========

Long-running HTTP service for repeated energy/force requests. Useful with workflow managers and ORCA external optimization.

Commands
--------

Start server (trained Enerzyme model):

.. code-block:: bash

    enerzyme listen -c server.yaml -m model_dir/ -o server_out/ -b 0.0.0.0:5000 -mc config.yaml

Start server as an **external-calculator shell** (no trained Enerzyme model):

.. code-block:: bash

    enerzyme listen -c server_uma.yaml -o server_out/ -b 0.0.0.0:5000 -cp /path/to/uma.py

Client request:

.. code-block:: bash

    enerzyme request -u http://127.0.0.1:5000 -f ORCA -i input.extinp.tmp -k FF02

In shell mode the default :code:`model_key` is :code:`external` (omit :code:`-k` or pass :code:`-k external`).

Shutdown:

.. code-block:: bash

    enerzyme kill -u http://127.0.0.1:5000

Implementation
--------------

- :code:`enerzyme/listen.py` — Flask/Waitress, route :code:`POST /calculate`
- :code:`enerzyme/tasks/server.py` — ASECalculator bridge (internal MLFF and/or external patch)
- :code:`enerzyme/request.py` — ORCA :code:`.extinp.tmp` → :code:`.engrad` bridge

Request format
--------------

JSON body (conceptual):

.. code-block:: json

    {
        "model_key": "FF02",
        "input_file": "/path/to/geometry",
        "features": {
            "Ra": [[...]],
            "Za": [...],
            "N": 100,
            "Q": -1
        }
    }

Exact schema follows what :code:`Server.calculate` expects for your model's active features.

Response
--------

JSON with :code:`outputs` (energy, forces, etc.) and :code:`units` (:code:`Hartree_in_E`, :code:`Bohr_in_R`).

Multi-model serving
-------------------

:code:`listen` loads all :code:`active: true` models from :code:`config.yaml` when an internal calculator is active. Clients select via :code:`model_key` (:code:`-k`).

External calculator and shell mode
----------------------------------

Server YAML mirrors the simulate hybrid keys under :code:`Server:`:

.. code-block:: yaml

    Server:
      cuda: true
      dtype: float64
      neighbor_list: full
      Hartree_in_E: 1.0
      external_calculator:
        name: uma_calculator
        weight: 1.0
        params:
          model: uma-s-1p2
          task: omol
          device: cuda
      internal_calculator_weight: 0.0

- Supply :code:`-cp` pointing to a :code:`.py` patch that exposes :code:`get_<name>` (same contract as :code:`enerzyme simulate`).
- When :code:`internal_calculator_weight` is :code:`0` and :code:`uncertainty_calculator` is omitted, the server runs in **shell mode**: no :code:`-m` / Modelhub / checkpoint is required; only the external patch is loaded.
- If :code:`uncertainty_calculator` (e.g. UDD) is set, an internal model is still required even when its energy weight is zero.
- Bundled example: :code:`enerzyme/config/server_uma.yaml`.

Server config
-------------

For pure internal serving, a minimal config may only need:

.. code-block:: yaml

    Datahub:
        preload: true

Model architecture and transforms come from :code:`-mc` / :code:`model_dir/config.yaml`.

Deployment notes
----------------

- Model (or external calculator) load time is paid once at startup — amortize over many requests
- Bind address :code:`-b` controls network exposure
- Logs go to :code:`out_dir` and waitress logger (wired to Enerzyme logger)
- For batch evaluation on static datasets, prefer :code:`enerzyme predict`
