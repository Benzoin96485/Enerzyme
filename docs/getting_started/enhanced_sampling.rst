Enhanced Sampling and Hybrid Potentials
========================================

Beyond standard MD and distance scans, Enerzyme supports **PLUMED-biased dynamics**, **PLUMED flexible scans**, and **hybrid internal/external calculators**. PLUMED workflows integrate with `Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ CV plugins.

PLUMED steered MD (:code:`task: plumed`)
----------------------------------------

Requires :code:`py-plumed` and a PLUMED-enabled library build.

Minimal config (:code:`enerzyme/config/plumed.yaml`):

.. code-block:: yaml

    Simulation:
        task: plumed
        idx_start_from: 1
        neighbor_list: full
        sampling:
            params:
                plumed_setup:
                - "UNITS LENGTH=A TIME=0.010180505671156723 ENERGY=96.48533288249877"
                - "FLUSH STRIDE=20"
        integrate:
            integrator: Langevin
            time_step: 0.5
            temperature_in_K: 300
            friction: 0.01
            n_step: 100000

.. caution::
    The :code:`UNITS` line must match ASE unit conventions for your model's :code:`Hartree_in_E` and time step. Incorrect units are a common source of unstable biased MD.

Enerzyme writes :code:`plumed.dat` and runs Langevin dynamics with the PLUMED wrapper. Trajectory: :code:`plumed.traj.xyz`.

CV plugins via :code:`-pp`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reaction-specific collective variables, pass a PLUMED patch module:

.. code-block:: bash

    enerzyme simulate -c config.yaml -o out/ -m model_dir/ -pp /path/to/sammt.py

When :code:`plumed_config_generator` is set, Enerzyme calls a named generator instead of static :code:`plumed_setup` lines:

.. code-block:: yaml

    Simulation:
        task: plumed
        plumed_config_generator:
            name: SAMMTConfigGenerator
            method: standard_steered_md
        sampling:
            params:
                plumed_config:
                    dump_interval: 20
                    lower_bound: -1.5
                    upper_bound: 1.5
                    reference_pdb_file: ref.pdb

Enerzymette registers built-in plugins (e.g. :code:`sammt`) and resolves them with :code:`get_plumed_patch(key)`. See the `Enerzymette PLUMED plugin README <https://github.com/Benzoin96485/Enerzymette/blob/main/enerzymette/plumed_config_generator/README.md>`_.

PLUMED flexible scan (:code:`task: plumed_scan`)
--------------------------------------------------

Unlike legacy :code:`task: scan` (ASE distance + :code:`FixBondLengths`), :code:`plumed_scan` restrains a CV at each scan point via PLUMED:

.. code-block:: yaml

    Simulation:
        task: plumed_scan
        plumed_config_generator:
            name: SAMMTConfigGenerator
            method: scan
        optimize:
            optimizer: LBFGS
        sampling:
            cv: plumed
            params:
                x0: 0.42
                x1: -1.2
                num: 25
                plumed_config:
                    lower_bound: -1.5
                    upper_bound: 1.5

Output: :code:`scan_optim.xyz` (same as bond-distance scan).

Dual scan paths
---------------

+----------------------+---------------------------+---------------------------+
| Path                 | :code:`task`              | CV mechanism              |
+======================+===========================+===========================+
| Legacy bond scan     | :code:`scan`              | ASE :code:`distance`      |
+----------------------+---------------------------+---------------------------+
| CV plugin scan       | :code:`plumed_scan`       | Enerzymette PLUMED plugin |
+----------------------+---------------------------+---------------------------+

Enerzymette launchers
---------------------

`Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ automates scan and AL workflows:

:code:`enerzymette launch_enerzyme_scan`
    Generates :code:`plumed_scan` configs and passes :code:`-pp` to :code:`enerzyme simulate`. Flags: :code:`-pp` (plugin key), :code:`-psc` (CV parameter YAML).

:code:`enerzymette enerzyme_active_learning`
    Runs the external active-learning loop around Enerzyme simulation, extraction, QM annotation, and retraining. :code:`--initial-scan` runs chained :code:`plumed_scan` jobs before iteration 0 to populate a structure pool. See :doc:`active_learning` for the expected task-folder layout and template input files.

Hybrid and external calculators
-------------------------------

:code:`enerzyme/config/uma.yaml` shows blending an external ASE calculator with an internal MLFF:

.. code-block:: yaml

    Simulation:
        external_calculator:
            name: uma_calculator
            weight: 1.0
        uncertainty_calculator:
            name: UDD
            params:
                A: 4
                B: 1
        internal_calculator_weight: 0.0
        task: md

Provide the external calculator via a patch module:

.. code-block:: bash

    enerzyme simulate -c uma.yaml -o out/ -m model_dir/ -cp my_calculator_patch.py

The patch module must expose a factory (e.g. :code:`get_uma_calculator`) referenced by :code:`external_calculator.name`. :code:`UDD` applies uncertainty-driven debiasing when an internal model remains loaded (:code:`internal_calculator_weight` > 0).

When to use hybrid mode
^^^^^^^^^^^^^^^^^^^^^^^

- Long-range or electronic effects missing from the MLFF
- Uncertainty-aware correction during exploration
- Pure external potential MD with ML uncertainty monitoring (:code:`internal_calculator_weight: 0`)

Next steps
----------

- Iterative retraining from sampled frames: :doc:`active_learning`
- Remote inference for large workflows: :doc:`server_and_enerzymette`
