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

The hash string encodes data path, format, neighbor list mode, compression, and transforms. Reusing :code:`preload: true` skips recomputation when nothing relevant changed.

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
    Multiplies gradient/force targets by :code:`-1` when QC stored :math:`\nabla E` instead of forces.

:code:`total_energy_normalization`
    Global mean/variance normalization on total energy (use with care for relative energies).
