QM Annotation
=============

:code:`enerzyme annotate` drives batch quantum chemistry on structures from a **Supplier** and writes labeled pickles for training. Entry: :code:`enerzyme/annotate.py`.

Command
-------

.. code-block:: bash

    enerzyme annotate -c annotate.yaml -o labeled/ -t tmp/ -s 0 -e -1

- :code:`-t` — scratch directory for QM jobs
- :code:`-s`, :code:`-e` — slice supplier records (0-based start, exclusive end; :code:`-1` = all)

Configuration
-------------

.. code-block:: yaml

    Supplier:
        path: fragments.sdf
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
        pickle_name: fragments.pkl
        dump_single_run: false
        n_processes: 8

Suppliers
---------

Implemented in :code:`enerzyme/data/supplier.py`:

- :code:`SDFSupplier` — SDF with formal charges (RDKit)
- :code:`PickleSupplier` — pre-built datapoint lists

:code:`annotate.py` currently wires **TeraChem** only (:code:`enerzyme/qm/qm_driver.py`). Other engines (ORCA, PySCF, Psi4) may exist as stubs — verify before use.

QMDriver options
----------------

- :code:`n_processes` — parallel QM submissions
- :code:`dump_single_run` — cache per-structure outputs
- :code:`keep_output` / :code:`keep_molden` — retain QC logs
- :code:`clean_tmp` — remove scratch after success

Output schema
-------------

Labeled pickle fields should map to Datahub the same way as raw training data (:code:`coord` → :code:`Ra`, :code:`grad` → :code:`Fa` with :code:`negative_gradient`, etc.).

Merging into training
---------------------

1. Run annotate on extracted fragments
2. Merge :code:`fragments.pkl` into :code:`training_set.pkl` / :code:`validation_set.pkl`
3. Update :code:`Datahub.datasets` paths in the next :code:`train.yaml`

Environment
-----------

- :code:`terachem` on :code:`PATH` with valid license
- PCM radius file when using :code:`pcm: cosmo` or similar
- RDKit for SDF parsing

Related integrations
--------------------

ORCA ExtOpt bridge and batch annotate serve different purposes — see :doc:`/user_guide/integrations/orca_terachem`.
