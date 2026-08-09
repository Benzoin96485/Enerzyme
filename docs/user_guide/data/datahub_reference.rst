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

:code:`global_transforms` apply across all datasets. Per-dataset
:code:`transforms:` entries are mapped to dataset-local preprocessing
(:code:`SingleDataHub.preprocessings`). Keys that already appear under
:code:`global_transforms` are omitted from that remapping so overlapping
configs (common in active-learning YAML) are not applied twice; when values
differ, :code:`global_transforms` wins.

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

For :code:`data_format: aselmdb`, ASE LMDB already exposes standard names
(:code:`Ra`, :code:`Za`, :code:`E`, :code:`Fa`, …). Use **identity** maps
(:code:`E: null` / :code:`E: E`), not pickle aliases such as :code:`E: energy`
or :code:`Ra: coord` — non-identity maps for those fixed fields raise
:code:`ValueError`. Declared calculator fields (:code:`E`, :code:`Fa`, :code:`M2`, …)
are registered even when the first DB row lacks them; annotate also stores an
:code:`enerzyme_properties` schema in ASE DB metadata. Declared fields that are
still missing from the source raise at load time instead of being skipped silently.
Custom keys stored in ASE row :code:`data` may still use non-identity maps.

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

Defined under :code:`transforms`, :code:`global_transforms`, or (advanced)
:code:`preprocessings`:

- :code:`atomic_energy` — path to CSV (:code:`atom_type`, :code:`atomic_energy`)
- :code:`negative_gradient` — flip gradient sign for force targets (pickle / QC ∇E). Disabled for :code:`aselmdb`, where :code:`Fa` is already ASE physical forces
- :code:`total_energy_normalization` — global mean/std on :code:`E`
- :code:`energy_unit_conversion` — convert energy/force units into the training convention
- :code:`uniform_qs_init` — write per-atom :code:`Q_init_a` / :code:`S_init_a` as :code:`Q/N` and :code:`S/N` for flow-matching init (registers those fields as features automatically). Place under :code:`global_transforms` (single-dataset :code:`transforms:` is remapped there) or per-dataset :code:`transforms:` / :code:`preprocessings`
- :code:`xtb_qs_prior` — GFN2-xTB + xtbml Mulliken-style prior into the same :code:`Q_init_a` / :code:`S_init_a` keys (optional; requires :code:`tblite` with xtbml, e.g. :code:`pip install 'enerzyme[xtb]'`). Mutually exclusive with :code:`uniform_qs_init` and :code:`pyscf_nao_qs_prior`
- :code:`pyscf_nao_qs_prior` — GPU4PySCF/PySCF NAO prior into :code:`Q_init_a` / :code:`S_init_a`. YAML must set :code:`xc` and :code:`basis` (no defaults). Optional :code:`use_gpu` (default true) needs :code:`gpu4pyscf`; CPU path uses :code:`pyscf` only (:code:`pip install 'enerzyme[pyscf_nao]'`). Mutually exclusive with the other Q/S priors
- :code:`qs_delta` — overwrite :code:`Qa` / :code:`Sa` targets with residuals vs :code:`Q_init_a` / :code:`S_init_a`. Must share the same :code:`preprocessings` or :code:`global_transforms` block as exactly one prior hook so the prior exists before deltas; prediction inverse adds the prior back

Enable **only one** of :code:`uniform_qs_init`, :code:`xtb_qs_prior`, or :code:`pyscf_nao_qs_prior` across preprocessings and global transforms — they all write the same prior fields.

Cache invalidation
------------------

The preprocessing hash depends on :code:`data_path`, neighbor list settings, and transforms. Non-empty :code:`data_format`, :code:`connect_args`, and :code:`select_args` are included as well (empty / unset values are omitted for cache compatibility). Changing any of these creates a new :code:`processed_dataset_<hash>/` directory.
