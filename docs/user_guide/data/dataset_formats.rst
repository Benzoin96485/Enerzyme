Dataset Formats
===============

Enerzyme accepts :code:`pickle`, :code:`npz`, :code:`hdf5`, and :code:`aselmdb` via :code:`Datahub.data_format` (or by file suffix). Internally, preprocessed training data is stored as HDF5 under :code:`processed_dataset_<hash>/`.

Pickle
------

A dataset is a :code:`list` of dicts. Each dict is one frame or structure.

Typical keys (your names may differ; map in YAML):

- :code:`coord` / :code:`coordinates` — :code:`(N, 3)` float
- :code:`atom_type` / :code:`atomic_numbers` — :code:`(N,)`
- :code:`energy` — scalar
- :code:`grad` / :code:`forces` — :code:`(N, 3)`
- :code:`chrg` / :code:`atomic_charges` — :code:`(N,)`
- :code:`dipole` — :code:`(3,)`
- :code:`total_chrg` / :code:`total_charge` — scalar int

.. danger::
    Pickle is not secure against malicious files. Do not load untrusted pickles.

.. caution::
    Pickle compatibility depends on Python/NumPy versions. Prefer HDF5 cache (:code:`preload: true`) for long-lived projects.

NPZ
---

NumPy archive format for large numeric arrays. Use when you already store frames as stacked arrays. Field mapping in Datahub is the same as for pickle; ensure shapes match Enerzyme expectations (:code:`Nframe`, :code:`Natom`, etc.).

HDF5
----

Used both as an input format and as the **preprocessed cache** written by Datahub. Cached files live in :code:`processed_dataset_<hash>/pre_transformed.hdf5` with standardized internal field names.

ASE LMDB (:code:`aselmdb`)
--------------------------

ASE database / LMDB stores (:code:`ase.db.connect`) for large QM-labeled sets. Set :code:`data_format: aselmdb` or use a path/suffix of :code:`.aselmdb`. :code:`data_path` may be a file, a directory of DB files, or a glob.

Lazy property accessors map ASE calculator results and :code:`atoms.info` / row :code:`data` onto Enerzyme fields:

- :code:`Ra`, :code:`Za`, :code:`N` — geometry
- :code:`E`, :code:`Fa` — energy and forces (converted from ASE eV units toward Hartree / Ha·Å⁻¹ when :code:`new_energy_unit` is not :code:`eV`)
- :code:`M2` — dipole moment via :code:`atoms.get_dipole_moment()` when the calculator has ASE ``dipole`` (e·Å)
- :code:`Qa`, :code:`Sa` — charges / magnetic moments when present on the calculator
- :code:`Q`, :code:`S` — from ASE info ``charge`` / ``spin`` (fairchem-style; :code:`S = spin - 1`)

ASE calculator / info key names (:code:`energy`, :code:`dipole`, :code:`charge`, …) stay on the :code:`Atoms` object; Datahub only exposes the standard Enerzyme names above.

:code:`enerzyme annotate` writes this format by default via :code:`QMDriver.output_file` (see :doc:`/user_guide/workflows/qm_annotation`). Enerzymette's outer AL loop still merges :code:`fragments.pkl` today — keep :code:`pickle_name: fragments.pkl` for those campaigns, or train directly from :code:`.aselmdb` outside Enerzymette.

TeraChem to pickle
------------------

:code:`scripts/picklizer.py` groups TeraChem outputs:

.. code-block:: python

    from scripts.picklizer import picklizer
    picklizer(file_lists, output="dataset.pkl", flavor="terachem", provide_Q=-1)

Each :code:`file_lists` entry maps keys :code:`coord`, :code:`grad`, :code:`chrg`, :code:`dipole` to file paths. Gradients are converted from Ha/Bohr to Ha/Angstrom in the parser.

Merging labeled data
--------------------

Active-learning and QM pipelines often append new pickles. Merge lists in Python, then point :code:`Datahub.datasets.training.data_path` at the combined file or maintain separate dataset keys for old vs new data.
