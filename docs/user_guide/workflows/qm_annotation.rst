QM Annotation
=============

:code:`enerzyme annotate` drives batch quantum chemistry on structures from a **Supplier**. By default it writes an **ASE LMDB** (:code:`.aselmdb`); set :code:`pickle_name` or an :code:`output_file` ending in :code:`.pkl` / :code:`.pickle` for legacy Enerzymette pickle output. Entry: :code:`enerzyme/annotate.py`.

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
        template_input_file: terachem_template.in
        output_file: fragments.aselmdb
        keep_molden: false
        keep_stdout: false
        clean_tmp: true
        n_processes: 8

Suppliers
---------

Implemented in :code:`enerzyme/data/supplier.py`:

- :code:`SDFSupplier` — SDF with formal charges (RDKit)
- :code:`PickleSupplier` — pre-built datapoint lists
- :code:`XYZSupplier` — XYZ trajectories with optional default :code:`Q` / :code:`S`

:code:`annotate.py` currently wires **TeraChem** only (:code:`enerzyme/qm/qm_driver.py`). Other engines (ORCA, PySCF, Psi4) may exist as stubs — verify before use.

QMDriver options
----------------

- :code:`template_input_file` — TeraChem settings (basis, XC, solvent); run/charge/spin/coords are injected per structure
- :code:`output_file` — under the supplier output directory; :code:`.aselmdb` (default) or :code:`.pkl` / :code:`.pickle`
- :code:`pickle_name` — if set, write legacy pickle only (Enerzymette AL); mutually exclusive with ASE DB output
- :code:`n_processes` — parallel QM submissions
- :code:`keep_stdout` / :code:`keep_molden` — retain QC logs
- :code:`clean_tmp` — remove scratch after success

Output schema
-------------

**ASE LMDB (default):** each structure is an ASE :code:`Atoms` row with calculator energy/forces/(charges/dipole) and :code:`data` fields :code:`charge`, :code:`spin`, :code:`index`. Load with :code:`Datahub.data_format: aselmdb` (:doc:`/user_guide/data/dataset_formats`).

**Pickle (compat):** list of dicts with :code:`energy` / :code:`grad` (Hartree), :code:`dipole`, :code:`coord`, :code:`atom_type`, :code:`total_chrg`, :code:`total_spin`, :code:`index` — same contract Enerzymette merges into training pickles.

Merging into training
---------------------

1. Run annotate on extracted fragments
2. Point :code:`Datahub.datasets` at the :code:`.aselmdb` path (or combine multiple DB files via a directory / glob)
3. Update paths in the next :code:`train.yaml`

Environment
-----------

- :code:`terachem` on :code:`PATH` with valid license
- PCM radius file when using :code:`pcm: cosmo` or similar
- RDKit for SDF parsing

Related integrations
--------------------

ORCA ExtOpt bridge and batch annotate serve different purposes — see :doc:`/user_guide/integrations/orca_terachem`.
