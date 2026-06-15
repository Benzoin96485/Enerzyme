Fragment Extraction
===================

:code:`enerzyme extract` ranks structures by local force uncertainty and extracts capped substructures for QM relabeling. Algorithm: :code:`enerzyme/tasks/extractor.py` (RDKit).

Command
-------

.. code-block:: bash

    enerzyme extract -c extract.yaml -o out/ -m model_dir/ -mc config.yaml [-s]

:code:`-s` skips prediction if pickles already exist in :code:`output_dir`.

Configuration
-------------

.. code-block:: yaml

    Datahub:
        data_path: candidates.pkl
        features:
            N:
            Q: Q
            Ra: Ra
            Za: Za
        neighbor_list: full
        preload: true
    Extractor:
        reference_mol_path: cluster.mol
        fragment_per_frame: 6
        local_uncertainty_radius: 5.0
        fragment_radius: 4.0
        n_centers: 2
    Trainer:
        inference_batch_size: 4

Parameters
----------

:code:`reference_mol_path`
    MOL/SDF defining connectivity. One molecule applies to all frames; multiple entries align by index.

:code:`local_uncertainty_radius`
    Neighborhood (Angstrom) for aggregating atomic force variance (`J. Chem. Inf. Model. 2024, 64, 6377-6387`).

:code:`fragment_radius`
    All heavy atoms within this radius of the uncertain center are included.

:code:`fragment_per_frame`
    Top-N fragments per trajectory frame.

:code:`n_centers`
    Number of uncertainty centers per frame.

Prerequisites
-------------

- Uncertainty-capable model (shallow ensemble or committee)
- Prediction step on the same :code:`Datahub` features as training
- Consistent charge and topology with reference molecule

Pipeline position
-----------------

Typical Enerzymette round: MD trajectory → predict → **extract** → annotate → train.

Outputs
-------

Fragment SDF (e.g. :code:`FF02-SpookyNet-19_fragments.sdf`) fed to :code:`annotate` via :code:`Supplier.path`.
