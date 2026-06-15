Enerzymette Integration
=======================

`Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ is a separate package of workflow scripts and launchers around Enerzyme. It is **not** installed with core Enerzyme.

Install
-------

.. code-block:: bash

    git clone https://github.com/Benzoin96485/Enerzymette.git
    cd Enerzymette
    pip install -e .

Main launchers
--------------

:code:`enerzymette enerzyme_active_learning`
    Outer AL loop: simulate → predict/extract → annotate → train. See :doc:`/user_guide/workflows/active_learning`.

:code:`enerzymette enerzyme_scan` / :code:`launch_enerzyme_scan`
  Chained reaction-coordinate scans with optional PLUMED CV plugins (:code:`-pp`, :code:`-psc`).

PLUMED CV plugins
-----------------

Plugins live under :code:`enerzymette/plumed_config_generator/`. Register keys (e.g. :code:`sammt`) and pass to Enerzyme:

.. code-block:: bash

    enerzyme simulate ... -pp <plugin_key_or_module>

Contract: three functions :code:`get_<key>_reaction_coordinate`, :code:`get_<key>_config`, :code:`get_<key>_scan_config`. See the `plugin README <https://github.com/Benzoin96485/Enerzymette/blob/main/enerzymette/plumed_config_generator/README.md>`_.

Utility commands
----------------

:code:`enerzymette idpp`
    IDPP path between reactant/product XYZ for NEB seeding.

:code:`enerzymette terachem_timing`
    Wall-time stats from TeraChem output files.

:code:`enerzymette orca_terachem_request`
    ORCA ExtOpt → TeraChem gradient bridge. See :doc:`/user_guide/integrations/orca_terachem`.

Boundary with Enerzyme
----------------------

+------------------+------------------------+---------------------------+
| Responsibility   | Enerzyme               | Enerzymette               |
+==================+========================+===========================+
| Model training   | yes                    | launches :code:`train`    |
+------------------+------------------------+---------------------------+
| MD / PLUMED      | yes (:code:`simulate`) | writes per-round YAML     |
+------------------+------------------------+---------------------------+
| Task scheduling  | no                     | yes                       |
+------------------+------------------------+---------------------------+
| QM engines       | :code:`annotate`       | ORCA bridge utilities     |
+------------------+------------------------+---------------------------+

Recommended task directory
--------------------------

Stable inputs at task root: :code:`al.sh`, :code:`cluster.xyz`, :code:`cluster.mol`, :code:`config/*.yaml`. Generated iteration folders :code:`FFxx_*` under the same root.
