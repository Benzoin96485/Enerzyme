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

    plumed_config_generator:
        name: get_sammt_config
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

:code:`plumed` vs :code:`plumed_scan`
-------------------------------------

- :code:`plumed` — Biased Langevin MD
- :code:`plumed_scan` — Restrained optimization per CV point

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

Enerzymette SAM-MT plugins may append OPES biases when :code:`proton_transfer: true` in :code:`plumed_config`. Requires a PLUMED build with OPES enabled. Documented in the `Enerzymette PLUMED plugin README <https://github.com/Benzoin96485/Enerzymette/blob/main/enerzymette/plumed_config_generator/README.md>`_.
