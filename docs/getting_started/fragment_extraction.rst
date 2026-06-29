Extracting Fragments by Local Uncertainty
=========================================

:code:`enerzyme extract` identifies substructures around **locally uncertain** regions in predicted structures, typically for targeted QM relabeling in an active-learning loop. The reference config is :code:`enerzyme/config/extract.yaml`.

Basic command
-------------

.. code-block:: bash

    enerzyme extract -c extract.yaml -o extract_out/ -m model_dir/ -mc train.yaml

Arguments match :code:`predict` plus:

- :code:`-s` / :code:`--skip_prediction` — skip inference if prediction pickles already exist in :code:`output_dir`

Workflow
--------

1. **Predict** on a candidate structure set (committee model recommended)
2. **Rank** atoms by local force uncertainty (committee std)
3. **Extract** fragments centered on uncertain atoms, using a reference molecule for connectivity
4. **Annotate** extracted fragments with QM (:doc:`qm_annotation`)
5. **Merge** new labels into the training pool (:doc:`active_learning`)

Configuration
-------------

.. code-block:: yaml

    Datahub:
        data_format: pickle
        data_path: "candidates.pkl"
        features:
            N:
            Q: Q
            Ra: Ra
            Za: Za
        neighbor_list: full
        preload: true
    Extractor:
        reference_mol_path: "reference.sdf"
        fragment_per_frame: 1
        local_uncertainty_radius: 5
        fragment_radius: 5

Extractor parameters
--------------------

:code:`reference_mol_path`
    Reference structure in SDF format. One molecule applies to all frames; multiple entries align by index with frames.

:code:`fragment_per_frame`
    Number of top-uncertainty fragments to extract per structure.

:code:`local_uncertainty_radius`
    Neighborhood radius (Å) for aggregating atomic uncertainty (`J. Chem. Inf. Model. 2024, 64, 6377–6387`).

:code:`fragment_radius`
    All heavy atoms within this radius (Å) of the uncertain center are included in the fragment.

Prerequisites
-------------

- A **committee-trained** model (:code:`committee_size` > 1) for meaningful force variance
- Prediction step (unless :code:`--skip_prediction`) using the same :code:`Datahub` features as training
- RDKit (installed with Enerzyme) for fragment manipulation

Example with skip prediction
----------------------------

After a prior :code:`predict` or :code:`extract` run saved pickles:

.. code-block:: bash

    enerzyme extract -c extract.yaml -o extract_out/ -m model_dir/ -mc train.yaml -s

Outputs
-------

Extracted fragments are written under :code:`output_dir` for downstream :code:`annotate` or manual QC. Use consistent charge and connectivity with your reference SDF.

Closing the loop
----------------

.. code-block:: text

    simulate / AL sampling  →  predict  →  extract  →  annotate  →  merge dataset  →  train

See :doc:`active_learning` for the iterative training side and :doc:`qm_annotation` for labeling.
