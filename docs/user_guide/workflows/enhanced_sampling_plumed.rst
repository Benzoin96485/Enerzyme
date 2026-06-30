Enhanced Sampling and PLUMED
==============================

PLUMED-biased dynamics and PLUMED-restrained scans extend standard MD and distance scans with collective variables (CVs). Requires :code:`py-plumed` and a PLUMED-enabled library build.

Static PLUMED input
-------------------

:code:`enerzyme/config/plumed.yaml` uses fixed lines:

.. code-block:: yaml

    sampling:
        params:
            plumed_setup:
            - "UNITS LENGTH=A TIME=0.010180505671156723 ENERGY=96.48533288249877"
            - "FLUSH STRIDE=20"

Enerzyme writes :code:`plumed.dat` in the output directory and wraps the ML calculator with ASE :code:`Plumed`.

PLUMED units
------------

The :code:`UNITS` line must be consistent with:

- Model :code:`Hartree_in_E` and coordinate units
- :code:`integrate.time_step` and :code:`fs_in_t`
- ASE's internal unit conventions

Incorrect units are a primary cause of unstable biased MD.

Dynamic generators (:code:`-pp`)
--------------------------------

For reaction-specific CVs, set:

.. code-block:: yaml

    Simulation:
        task: plumed
        plumed_config_generator:
            name: SAMMTConfigGenerator
            method: standard_steered_md
        sampling:
            params:
                plumed_config:
                    lower_bound: -2
                    upper_bound: 2
                    reference_pdb_file: ref.pdb
                    substrate: KOM
                    nucleophile: O9

Pass the plugin module:

.. code-block:: bash

    enerzyme simulate -c sim.yaml -o out/ -m model_dir/ -pp /path/to/plugin.py

Enerzymette registers keys such as :code:`sammt` and resolves them via :code:`get_plumed_patch(key)`.

Supported generator methods:

- :code:`standard_steered_md` — one round trip across :code:`[lower_bound, upper_bound]`
- :code:`naive_steered_md` — warmup pull to nearest bound, then pull to opposite bound
- :code:`scan` — static PLUMED :code:`RESTRAINT` at :code:`target_value` (used by :code:`task: plumed_scan`)

:code:`plumed` vs :code:`plumed_scan`
-------------------------------------

- :code:`plumed` — Biased Langevin MD
- :code:`plumed_scan` — Restrained optimization per CV point; **does not** insert proton-transfer plugins even if configured

Legacy ASE :code:`task: scan` with :code:`cv: distance` does not use PLUMED.

Hybrid calculators and UDD
--------------------------

:code:`uma.yaml` pattern:

.. code-block:: yaml

    external_calculator:
        name: uma_calculator
        weight: 1.0
    internal_calculator_weight: 0.0
    uncertainty_calculator:
        name: UDD
        params:
            A: 4
            B: 1

Supply :code:`-cp` pointing to a module that implements the external calculator factory. UDD debiases or adjusts forces using ML uncertainty estimates when an internal model remains loaded.

OPES / proton transfer
----------------------

Enerzymette can append OPES biases for proton-transfer enhanced sampling. Enable with a nested mapping under :code:`plumed_config`:

.. code-block:: yaml

    plumed_config:
        lower_bound: -2
        upper_bound: 2
        reference_pdb_file: ref.pdb
        substrate: G
        nucleophile: "O2'"
        proton_transfer:
            enabled: true
            plugin: local_opes
            donor: nucleophile
            flavor: nearest_distance
            scope_file: structure_pool/000.pt_scope.json
            state_file: opes_state.data
            restart: false
            topology_mol_file: cluster.mol
            opes_barrier: 20

Requires a PLUMED build with OPES enabled. If required atoms cannot be resolved, the generator falls back to the main reaction-coordinate bias only.

When :code:`enerzymette enerzyme_active_learning` runs with proton transfer in the simulation template, it injects per-pool-entry :code:`scope_file`, :code:`state_file`, :code:`restart`, and :code:`topology_mol_file` each iteration and persists OPES state under :code:`structure_pool/`.

Field reference and plugin developer interface: `Enerzymette PLUMED plugin README <https://github.com/Benzoin96485/Enerzymette/blob/main/enerzymette/plumed_config_generator/README.md>`_.
