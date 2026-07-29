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
- :code:`pickle_name` — if set, write pickle only (Enerzymette AL); mutually exclusive with ASE DB output
- :code:`pickle_fields` — optional map from **standard** names (:code:`E`, :code:`Fa`, :code:`M2`, …) to custom pickle keys; omit for identity. Enerzymette smoke uses :code:`E→energy`, :code:`Fa→grad` (stores :math:`-\mathbf{F}`), :code:`M2→dipole`, …
- :code:`n_processes` — parallel QM submissions
- :code:`keep_stdout` / :code:`keep_molden` — retain QC logs (:code:`keep_output` is a deprecated alias for :code:`keep_stdout`)
- :code:`clean_tmp` — remove scratch after success

For PCM, put :code:`pcm_radii_file <name>` in the template (path relative to the template file is fine). :code:`TeraChemDriver` copies that file into the per-job tmp directory so host absolute paths are not required in committed configs.

Output schema
-------------

**ASE LMDB (default):** each structure is an ASE :code:`Atoms` row with calculator energy/forces/(charges/dipole) and :code:`data` fields :code:`charge`, :code:`spin`, :code:`index`. Load with :code:`Datahub.data_format: aselmdb` and **identity** maps (:code:`E`, :code:`Fa`, :code:`M2`, …); see :doc:`/user_guide/data/dataset_formats`.

**Pickle (default):** list of dicts with standard names :code:`E` / :code:`Fa` (Hartree), :code:`M2`, :code:`Ra`, :code:`Za`, :code:`Q`, :code:`S`, :code:`N`, :code:`index`. With :code:`pickle_fields`, keys are renamed for Enerzymette (:code:`energy` / :code:`grad` / :code:`dipole` / …).

Merging into training
---------------------

1. Run annotate on extracted fragments
2. Point :code:`Datahub.datasets` at the :code:`.aselmdb` path (or combine multiple DB files via a directory / glob)
3. Update paths in the next :code:`train.yaml`

Environment
-----------

- :code:`terachem` on :code:`PATH` with valid license
- PCM radius file when the template uses :code:`pcm_radii read` (prefer a file next to the template; see above)
- RDKit for SDF parsing

Smoke / AL fixtures
-------------------

:code:`example/L3-COMT-aselmdb-smoke/` vendors a tiny COMT topology, PCM radii, annotate/train YAMLs, and an :code:`enerzymette_al/` config set for one minimal Enerzymette iteration (run from the repo root; load modules and caches on the CLI).

Related integrations
--------------------

ORCA ExtOpt bridge and batch annotate serve different purposes — see :doc:`/user_guide/integrations/orca_terachem`.
