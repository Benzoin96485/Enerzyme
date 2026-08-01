Training a Neural Network Potential
===================================

To train a neural network potential (NNP) with :code:`dataset.pkl`, write a configuration file :code:`train.yaml` that specifies data preprocessing, model architecture, training protocol, and metrics. This page walks through a **minimal SchNet setup**. The full multi-architecture reference is :code:`enerzyme/config/train.yaml`.

Datahub
-------

.. code-block:: yaml

    Datahub:

The :code:`Datahub` section connects your dataset to the model. At minimum specify:

- :code:`data_path` — path to the dataset (relative paths are allowed)
- :code:`data_format` — :code:`pickle`, :code:`npz`, or :code:`hdf5`

Then define :code:`features` (model inputs) and :code:`targets` (quantities to fit). Each key is a **standard Enerzyme field name**; the value is the attribute name in your dataset (leave empty if they match).

Standard Data Field Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Physical quantities use capitalized standard names. Atomic quantities end with :code:`a`:

- :code:`N` — number of atoms
- :code:`Ra` — coordinates
- :code:`Za` — atomic numbers
- :code:`E` — total energy
- :code:`Fa` — forces
- :code:`Qa` — atomic charges
- :code:`Q` — total charge
- :code:`M2` — dipole moment

You may rename dataset attributes freely as long as the mapping is consistent, e.g. :code:`Ra: xyz` instead of :code:`Ra: coordinates`.

Preprocessing the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^

NNPs in Enerzyme are graph neural networks. Optional preprocessing improves efficiency and accuracy:

- :code:`neighbor_list: full` — precompute all-pairs neighbor lists before training
- :code:`compressed: true` — share :code:`Za`, :code:`N`, :code:`Q`, and neighbor lists across frames of the same stoichiometry (same atom order)
- :code:`preload: true` — reuse cached HDF5 under :code:`processed_dataset_<hash>/` when config hash matches
- :code:`transforms.negative_gradient: true` — flip gradient sign when :code:`Fa` stores QC gradients, not forces
- :code:`transforms.atomic_energy` — subtract per-atom reference energies from a CSV (:code:`atom_type`, :code:`atomic_energy`)

.. note::
    The canonical config uses :code:`compressed`, not :code:`compression`. Older single-dataset YAML without :code:`datasets:` still accepts the legacy layout; new projects should follow :code:`train.yaml`.

Final Datahub Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    Datahub:
        data_path: "dataset.pkl"
        data_format: "pickle"
        features:
            N: "number_of_atoms"
            Ra: "coordinates"
            Za: "atomic_numbers"
            Q: "total_charge"
        targets:
            E: "energy"
            Fa: "forces"
            Qa: "atomic_charges"
            M2: "dipole"
        neighbor_list: full
        compressed: true
        preload: true
        transforms:
            atomic_energy: "atomic_energy.csv"
            negative_gradient: true

Modelhub
--------

Models live under :code:`internal_FFs` (built into Enerzyme) or :code:`external_FFs` (NequIP, XPaiNN, etc.). Each model has a unique ID (e.g. :code:`FF01`).

Choosing an architecture
^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+------------------------------------------+---------------------------+
| Architecture   | Good for                                 | Notes                     |
+================+==========================================+===========================+
| SchNet         | First tutorial / baseline                | Charge + dipole capable   |
+----------------+------------------------------------------+---------------------------+
| PhysNet        | Production charge-aware PES              | Electrostatics, D3 layers |
+----------------+------------------------------------------+---------------------------+
| SpookyNet      | Large organic / mixed systems            | Similar feature set       |
+----------------+------------------------------------------+---------------------------+
| MACE           | Equivariant accuracy                     | Higher compute cost       |
+----------------+------------------------------------------+---------------------------+
| NequIP         | External equivariant model               | Requires :code:`nequip`   |
+----------------+------------------------------------------+---------------------------+
| XPaiNN         | External XPaiNN via XequiNet             | Extra pip packages        |
+----------------+------------------------------------------+---------------------------+

Enable exactly one model (:code:`active: true`) when starting out.

SchNet configuration
^^^^^^^^^^^^^^^^^^^^

SchNet is fully internal—no extra pip packages. Key entries:

- :code:`architecture: SchNet`
- :code:`build_params` — :code:`dim_embedding`, :code:`num_rbf`, :code:`max_Za`, :code:`cutoff_sr`, :code:`Hartree_in_E`, :code:`Bohr_in_R`
- :code:`layers` — modular stack ending with :code:`Force` for analytic forces
- :code:`loss` — weighted sum over targets (convert force weights to your energy/length units)
- :code:`Metric` — per-model validation metric for early stopping (same weights as loss is common)

.. note::
    :code:`Hartree_in_E` and :code:`Bohr_in_R` convert internal atomic units to your dataset units. For Ha and Å, use :code:`1` and :code:`0.5291772108`. For eV and Å, use :code:`~27.2` and :code:`~0.529`.

Final Modelhub configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    Modelhub:
        internal_FFs:
            FF01:
                suffix:
                architecture: SchNet
                active: true
                build_params:
                    dim_embedding: 128
                    num_rbf: 128
                    max_Za: 94
                    cutoff_sr: 5.0
                    Hartree_in_E: 1
                    Bohr_in_R: 0.5291772108
                layers:
                  - name: RangeSeparation
                  - name: GaussianSmearing
                  - name: RandomAtomEmbedding
                  - name: Core
                    params:
                        num_interactions: 4
                        hidden_channels: 128
                  - name: AtomicAffine
                  - name: ChargeConservation
                  - name: AtomicCharge2Dipole
                  - name: ElectrostaticEnergy
                  - name: EnergyReduce
                  - name: Force
                loss:
                    rmse:
                        Fa: 52.917721
                        Qa: 1
                        E: 1
                        M2: 1.8897261
                        Q: 1
                Metric:
                    E:
                        rmse: 1
                    Fa:
                        rmse: 52.917721
                    Qa:
                        rmse: 1
                    M2:
                        rmse: 1.8897261
                    Q:
                        rmse: 1

Trainer
-------

Splitter
^^^^^^^^

Random splitting is configured under :code:`Trainer.Splitter`:

- :code:`parts` — partition names (at least :code:`training` and :code:`validation`)
- :code:`ratios` — fractions or absolute counts, in the same order as :code:`parts`
- :code:`seed`, :code:`save`, :code:`preload` — reproducibility and index caching

Training loop
^^^^^^^^^^^^^

Each epoch loads mini-batches of :code:`batch_size` using :code:`num_workers` processes. Training stops at :code:`max_epochs` or when validation :code:`judge_score` fails to improve for :code:`patience` epochs.

Optimizer and scheduler
^^^^^^^^^^^^^^^^^^^^^^^

Enerzyme uses Adam with a linear warmup scheduler (:code:`schedule: linear`, :code:`warmup_ratio`).

System settings
^^^^^^^^^^^^^^^

Set :code:`cuda: true` for GPU training and :code:`dtype: float32` (or :code:`float64`). Set :code:`seed` for reproducibility.

.. caution::
    CUDA nondeterminism may still cause run-to-run differences even with a fixed seed.

Final Trainer configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    Trainer:
        Splitter:
            method: random
            parts:
            - training
            - validation
            - test
            preload: true
            ratios:
            - 0.7
            - 0.1
            - 0.2
            save: true
            seed: 42
        batch_size: 64
        cuda: true
        dtype: float32
        schedule: linear
        learning_rate: 0.001
        max_epochs: 10000
        num_workers: 10
        patience: 50
        seed: 42
        warmup_ratio: 0.001

Running the training job
------------------------

.. code-block:: bash

    enerzyme train -c train.yaml -o .

Output artifacts
----------------

After training, the output directory typically contains:

- :code:`config.yaml` — resolved configuration (**keep this for predict/simulate**)
- :code:`processed_dataset_<hash>/` — preprocessed HDF5 cache
- :code:`logs/` — training logs, metrics, early-stopping traces
- :code:`FF01/` (or your model ID) — :code:`best/` and :code:`last/` checkpoints

Use :code:`enerzyme/config/train.yaml` when you need multi-dataset Datahub, external models, EMA, Lightning multi-GPU, or pretraining paths.
