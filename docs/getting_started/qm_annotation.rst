QM Data Annotation
==================

Enerzyme can drive batch quantum chemistry calculations to label structures with energies, forces, charges, and dipoles. Labeled data is written to pickle format for training. The reference config is :code:`enerzyme/config/annotate.yaml`.

Basic command
-------------

.. code-block:: bash

    enerzyme annotate -c annotate.yaml -o labeled/ -t tmp/ -s 0 -e 100

Arguments:

- :code:`-c` — annotation config (:code:`Supplier` + :code:`QMDriver`)
- :code:`-o` — output directory
- :code:`-t` — scratch directory for QM jobs
- :code:`-s`, :code:`-e` — slice of molecules from the supplier (0-based start, exclusive end; :code:`-1` for all)

Configuration
-------------

.. code-block:: yaml

    Supplier:
        path: molecules.sdf
    QMDriver:
        engine: TeraChem
        bs: 6-31gs
        xc: b3lyp
        pcm: cosmo
        dftd: d3
        pcm_radii_file: /path/to/pcm_radii
        epsilon: 10
        keep_molden: false
        keep_output: false
        clean_tmp: true
        pickle_name: labeled/dataset.pkl

Supplier
^^^^^^^^

:code:`Supplier.path` points to an SDF file. Each record must include **total charge** (RDKit formal charge). Enerzyme iterates structures and submits QM jobs.

QMDriver (TeraChem)
^^^^^^^^^^^^^^^^^^^

The reference driver targets **TeraChem**. Required environment:

- :code:`terachem` executable on :code:`PATH`
- Valid license and scratch space
- PCM radius file when :code:`pcm_radii` mode needs it

Output pickle schema matches Datahub expectations—map fields in your training YAML the same way as in :doc:`preparing_dataset`.

.. caution::
    Prepare TeraChem and RDKit before running :code:`annotate`. These are not part of the core Enerzyme install.

Integrating labeled data into training
--------------------------------------

1. Run :code:`annotate` to produce :code:`dataset.pkl`
2. Append or merge with existing training data
3. Update :code:`train.yaml` :code:`data_path` or multi-dataset :code:`Datahub.datasets`
4. Retrain or resume active learning (:doc:`active_learning`)

ORCA optimizer with TeraChem gradients (Enerzymette)
----------------------------------------------------

`Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ bridges ORCA's external optimizer (ExtOpt) to TeraChem GPU gradients:

.. code-block:: bash

    enerzymette orca_terachem_request -i orca.extinp.tmp -t terachem_template.inp

Setup checklist:

1. ORCA input with :code:`! ExtOpt` and :code:`%method ProgExt "/path/to/wrapper.sh" end`
2. TeraChem template with basis, functional, solvent, SCF settings
3. Wrapper script: :code:`enerzymette orca_terachem_request -i $1 -t template.inp` (executable)
4. :code:`terachem` available in the environment

This complements batch :code:`annotate` when you need QC-level optimization with ORCA's algorithms and TeraChem's energy/gradient engine.

Monitoring TeraChem jobs (Enerzymette)
--------------------------------------

.. code-block:: bash

    enerzymette terachem_timing -f terachem.out

Reports total wall time, per-iteration averages, and flags unfinished calculations—useful for QC campaign quality control before merging data into training sets.

Related workflows
-----------------

- Extract uncertain fragments before expensive QM: :doc:`fragment_extraction`
- Convert existing TeraChem outputs without re-running QC: :doc:`preparing_dataset` (:code:`scripts/picklizer.py`)
