Prediction Server and Enerzymette Integration
==============================================

Enerzyme can run as a long-lived **prediction server** that responds to HTTP requests. This mode pairs with the `Enerzymette workflow manager <https://github.com/Benzoin96485/Enerzymette>`_ for orchestrated scan, active-learning, and optimization campaigns.

Starting the server
-------------------

.. code-block:: bash

    enerzyme listen -c server.yaml -m model_dir/ -o server_out/ -b 0.0.0.0:5000 -mc train.yaml

Arguments:

- :code:`-c` — server config (minimal YAML; see below)
- :code:`-m` — trained model directory (optional in external-calculator shell mode)
- :code:`-mc` — model config (defaults to :code:`model_dir/config.yaml`)
- :code:`-b` — bind address (host:port)
- :code:`-o` — output directory for server logs and artifacts
- :code:`-cp` — optional external calculator patch (:code:`.py`), same as :code:`simulate`

External-calculator shell (no trained Enerzyme model):

.. code-block:: bash

    enerzyme listen -c enerzyme/config/server_uma.yaml -o server_out/ -b 0.0.0.0:5000 -cp /path/to/uma.py

Set :code:`Server.internal_calculator_weight: 0` and omit :code:`uncertainty_calculator`. The server loads only the :code:`-cp` factory named by :code:`Server.external_calculator`.

Minimal server config
---------------------

.. code-block:: yaml

    Datahub:
        preload: true

With an internal model, the listen process reads :code:`Modelhub` from the model config, loads all :code:`active: true` models, and exposes them via Flask/Waitress. Hybrid / shell keys live under :code:`Server:` — see :doc:`/user_guide/integrations/server_mode`.

Sending a request (client)
--------------------------

.. code-block:: bash

    enerzyme request -u http://127.0.0.1:5000 -f ORCA -i input.extinp.tmp -k FF01

Arguments:

- :code:`-u` — server URL
- :code:`-f` — input format (:code:`ORCA` for external optimizer workflows)
- :code:`-i` — input file path
- :code:`-k` — model key (must match an active :code:`FF` ID in the model config)

The server returns JSON with predicted :code:`outputs` and unit metadata (:code:`Hartree_in_E`, :code:`Bohr_in_R`).

Shutting down the server
------------------------

.. code-block:: bash

    enerzyme kill -u http://127.0.0.1:5000

Sends a shutdown signal to the listening process.

When to use server mode
-----------------------

- External optimizers or workflow tools that request energies/forces repeatedly
- Enerzymette launchers that batch many :code:`simulate` or scan jobs against one loaded model
- Avoiding model reload overhead across hundreds of short calculations

Enerzymette workflow manager
----------------------------

Install Enerzymette separately:

.. code-block:: bash

    git clone https://github.com/Benzoin96485/Enerzymette.git
    cd Enerzymette
    pip install -e .

Relevant launchers:

+-----------------------------------+-----------------------------------------------+
| Command                           | Role                                          |
+===================================+===============================================+
| :code:`enerzymette enerzyme_scan` | Batch flexible scans (bond or PLUMED CV)      |
+-----------------------------------+-----------------------------------------------+
| :code:`enerzymette enerzyme_neb`  | NEB via ORCA ExtOpt + :code:`enerzyme listen` |
+-----------------------------------+-----------------------------------------------+
| :code:`enerzymette enerzyme_active_learning` | PLUMED steered MD AL iterations    |
+-----------------------------------+-----------------------------------------------+

:code:`enerzyme_neb` and :code:`enerzyme_scan` start :code:`enerzyme listen` when needed; :code:`enerzyme_active_learning` invokes :code:`enerzyme simulate` directly. See :doc:`enhanced_sampling` for PLUMED plugin details. End-to-end example: :code:`example/NNP4MTase`.

Architecture sketch
-------------------

.. code-block:: text

    Client / Enerzymette  --POST /calculate-->  enerzyme listen
                                                      |
                                                      v
                                              ASECalculator
                                           (internal and/or -cp)

Related pages
-------------

- Model evaluation without server: :doc:`prediction`
- On-node simulations: :doc:`simulation`
- Utility scripts (IDPP, timing): :doc:`bond_and_utilities`
