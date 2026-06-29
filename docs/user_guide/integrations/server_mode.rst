Server Mode
===========

Long-running HTTP service for repeated energy/force requests. Useful with workflow managers and ORCA external optimization.

Commands
--------

Start server:

.. code-block:: bash

    enerzyme listen -c server.yaml -m model_dir/ -o server_out/ -b 0.0.0.0:5000 -mc config.yaml

Client request:

.. code-block:: bash

    enerzyme request -u http://127.0.0.1:5000 -f ORCA -i input.extinp.tmp -k FF02

Shutdown:

.. code-block:: bash

    enerzyme kill -u http://127.0.0.1:5000

Implementation
--------------

- :code:`enerzyme/listen.py` — Flask/Waitress, route :code:`POST /calculate`
- :code:`enerzyme/tasks/server.py` — model forward pass
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

:code:`listen` loads all :code:`active: true` models from :code:`config.yaml`. Clients select via :code:`model_key` (:code:`-k`).

Server config
-------------

No dedicated :code:`listen.yaml` in the repository. A minimal config may only need:

.. code-block:: yaml

    Datahub:
        preload: true

Model architecture and transforms come from :code:`-mc` / :code:`model_dir/config.yaml`.

Deployment notes
----------------

- Model load time is paid once at startup — amortize over many requests
- Bind address :code:`-b` controls network exposure
- Logs go to :code:`out_dir` and waitress logger (wired to Enerzyme logger)
- For batch evaluation on static datasets, prefer :code:`enerzyme predict`
