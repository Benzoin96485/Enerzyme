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

:code:`enerzymette enerzyme_scan`
    Flexible bond or PLUMED CV scans. For each elementary reaction: optimize reactant → scan → optimize product → analyze path. Key flags:

    - :code:`-q` — TeraChem input **or** YAML **scan config** (:code:`reference_pdb`, :code:`freeze_index_types`, scan bond)
    - :code:`-pp` — PLUMED plugin key (e.g. :code:`sammt`); switches to :code:`plumed_scan`
    - :code:`-psc` — YAML with CV parameters (:code:`lower_bound`, :code:`reference_pdb_file`, etc.); **required** when :code:`-pp` is set

    Example scan config for bond-distance mode (:code:`-q scan_config.yaml`):

    .. code-block:: yaml

        reference_pdb: cluster.pdb
        reference_sdf: ligands.sdf   # optional; used with enerzyme bond when charge is omitted
        charge: 0                    # optional; if set, skip PDB charge derivation
        multiplicity: 1
        freeze_index_types: [backbone, Calpha]
        constraint_scan:
            bond:
                plugin: sammt
                substrate: G
                nucleophile: "O2'"

    When :code:`charge` is omitted, Enerzymette derives it via :code:`enerzyme bond` on :code:`reference_pdb` (optionally with :code:`reference_sdf`). Supporting utility: :code:`enerzymette update_terachem_scan` refreshes coordinates in a TeraChem scan input after a structure update.

:code:`enerzymette enerzyme_neb`
    NNP-driven NEB through ORCA ExtOpt and a running :code:`enerzyme listen` server. Key flags:

    - :code:`-r` / :code:`-p` — reactant and product XYZ
    - :code:`-q` — TeraChem reference input **or** YAML **neb config** (:code:`reference_pdb`, :code:`freeze_index_types`; optional :code:`charge`, :code:`reference_sdf`, :code:`multiplicity`)
    - :code:`-c` — server config (for pure external UMA, use :code:`server_uma.yaml` with :code:`internal_calculator_weight: 0`)
    - :code:`-cp` — external calculator patch path or registry key (e.g. :code:`uma`); required in shell mode
    - :code:`-t` — optional initial TS guess XYZ (enables ORCA :code:`%neb TS`)
    - :code:`-m` — trained model directory (optional when :code:`internal_calculator_weight` is 0)

    Example neb config (:code:`-q neb_config.yaml`):

    .. code-block:: yaml

        reference_pdb: cluster.pdb
        freeze_index_types:
            - backbone
        charge: 0          # optional; if omitted, charge is derived via enerzyme bond
        multiplicity: 1
        # reference_sdf: ligands.sdf

    Example launcher call (external-calculator shell):

    .. code-block:: bash

        enerzymette enerzyme_neb \
            -r reactant.xyz -p product.xyz -o neb_out/ \
            -q neb_config.yaml -c server_uma.yaml \
            -cp uma -t ts_guess.xyz -n 8 -b 5001

    See :doc:`server_and_enerzymette` for listen / shell-mode setup.

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
