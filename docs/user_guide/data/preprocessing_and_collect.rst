Preprocessing and Collect
=========================

Before training, Datahub may preprocess raw data into HDF5, build neighbor lists, and apply transforms. You can run this step alone with :code:`enerzyme collect`.

Collect command
---------------

.. code-block:: bash

    enerzyme collect -c train.yaml -o preprocess_out/

Uses the same :code:`Datahub` and :code:`Trainer.Splitter` as training but **does not** fit models. Useful to:

- Validate field mappings on a large dataset
- Pre-build :code:`processed_dataset_<hash>/` before a long GPU job
- Generate and inspect split indices

Preprocessing pipeline
----------------------

1. Load raw data from :code:`data_path` / :code:`datasets`
2. Map features and targets to standard fields
3. Apply transforms (:code:`atomic_energy`, :code:`negative_gradient`, etc.)
4. Optionally build :code:`neighbor_list: full`
5. Write :code:`processed_dataset_<hash>/pre_transformed.hdf5`
6. Optionally split and save partition indices

Hash directory
--------------

The hash string encodes data path, neighbor list mode, and transforms. Non-empty :code:`data_format`, :code:`connect_args`, and :code:`select_args` are included when set (empty values are omitted so older caches remain valid). Reusing :code:`preload: true` skips recomputation when nothing relevant changed.

Neighbor list cost
------------------

:code:`neighbor_list: full` stores all atom pairs — :math:`O(N^2)` per frame. For large clusters:

- Use :code:`compressed: true` when atom order and connectivity are fixed across frames
- Consider on-the-fly neighbor lists (:code:`neighbor_list: ''`) if memory is limiting and models support it

Transform details
-----------------

:code:`atomic_energy`
    Subtracts sum of per-atom reference energies from :code:`E`. CSV columns: :code:`atom_type`, :code:`atomic_energy` (same energy unit as targets).

:code:`negative_gradient`
    Multiplies gradient/force targets by :code:`-1` when QC stored :math:`\nabla E` instead of forces. For :code:`aselmdb`, this transform is disabled because :code:`Fa` already comes from ASE :code:`get_forces()`.

:code:`total_energy_normalization`
    Global mean/variance normalization on total energy (use with care for relative energies).

:code:`uniform_qs_init`
    Splits total :code:`Q` / :code:`S` uniformly onto atoms as :code:`Q_init_a` / :code:`S_init_a` (flow or delta priors).

:code:`xtb_qs_prior`
    Runs GFN2-xTB + xtbml Mulliken populations per frame into :code:`Q_init_a` / :code:`S_init_a`.
    Optional dependency: :code:`tblite` with xtbml (:code:`pip install 'enerzyme[xtb]'`).
    Typical YAML: :code:`enabled: true`, optional :code:`max_scf_iter` (default 1).

:code:`pyscf_nao_qs_prior`
    Runs finite-step DFT + NAO populations into :code:`Q_init_a` / :code:`S_init_a`.
    Requires :code:`xc` and :code:`basis` in YAML. Optional deps: :code:`pyscf` and, when
    :code:`use_gpu: true`, :code:`gpu4pyscf` (install CUDA stack separately; core extra is
    :code:`pip install 'enerzyme[pyscf_nao]'`).

:code:`qs_delta`
    Replaces :code:`Qa` / :code:`Sa` training targets with residuals against the prior.
    Enable together with exactly one prior in the same transform block. At predict/metric
    time, inverse transform restores full charges/spins using :code:`Q_init_a` / :code:`S_init_a`
    copied from batch features.