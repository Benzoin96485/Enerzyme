Datahub Reference
=================

:code:`Datahub` connects raw datasets to the training or inference pipeline. Implementation: :code:`enerzyme/data/datahub.py` (:code:`DataHub`, :code:`SingleDataHub`).

Single-dataset layout
---------------------

Legacy / minimal projects use one block:

.. code-block:: yaml

    Datahub:
        data_path: dataset.pkl
        data_format: pickle
        features:
            Ra: coordinates
            Za: atomic_numbers
            Q: total_charge
        targets:
            E: energy
            Fa: forces
        neighbor_list: full
        compressed: true
        preload: true
        transforms:
            atomic_energy: atomic_energy.csv
            negative_gradient: true

Multi-dataset layout
--------------------

Production configs use :code:`datasets:` so training and validation can point to different files:

.. code-block:: yaml

    Datahub:
        data_path: /fallback/or/metadata.pkl
        datasets:
            training:
                data_path: training_set.pkl
                features: {Ra: coord, Za: atom_type, Q: total_chrg}
                targets: {E: energy, Fa: grad, M2: dipole}
                transforms:
                    atomic_energy: /path/to/ref.csv
                    negative_gradient: true
            validation:
                data_path: validation_set.pkl
                # same mappings
        global_transforms:
            atomic_energy: /path/to/ref.csv
            negative_gradient: true

:code:`global_transforms` apply across datasets unless overridden per dataset.

Feature and target mapping
--------------------------

Keys are **standard Enerzyme names**; values are attribute names in your file. Use an empty value when names already match:

.. code-block:: yaml

    features:
        Ra: coord
        Za: atom_type
        N:
        Q: total_chrg

:code:`N` can be omitted and inferred from :code:`Za`. :code:`Q` defaults to 0 if missing.

Custom fields
-------------

Register extra atomic features:

.. code-block:: yaml

    Datahub:
        fields:
            Qa_xTB:
                is_atomic: true
        datasets:
            training:
                features:
                    Qa_xTB: xtb_chrg

Use the custom field in layers (e.g. :code:`ScalarDenseEmbedding` with :code:`embed_field: Qa_xTB`).

Preprocessing flags
-------------------

:code:`neighbor_list`
    :code:`full` precomputes all-pairs edges. Empty string computes on the fly during training (higher cost, flexible).

:code:`compressed`
    Share :code:`Za`, :code:`N`, :code:`Q`, and neighbor lists across frames with identical stoichiometry and atom order.

:code:`preload`
    Load :code:`processed_dataset_<hash>/pre_transformed.hdf5` if the config hash matches.

:code:`max_memory`
    HDF5 read cache limit in GB for large datasets.

Transforms
----------

Defined under :code:`transforms` or :code:`global_transforms`:

- :code:`atomic_energy` — path to CSV (:code:`atom_type`, :code:`atomic_energy`)
- :code:`negative_gradient` — flip gradient sign for force targets
- :code:`total_energy_normalization` — global mean/std on :code:`E`

Cache invalidation
------------------

The preprocessing hash depends on :code:`data_path`, :code:`data_format`, neighbor list settings, and transforms. Changing any of these creates a new :code:`processed_dataset_<hash>/` directory.
