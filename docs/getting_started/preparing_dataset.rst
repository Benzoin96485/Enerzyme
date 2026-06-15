Preparing a Neural Network Potential Dataset
============================================

To train a neural network potential (NNP), you need a dataset of atomic systems and their labels. Enerzyme supports :code:`pickle`, :code:`npz`, and :code:`hdf5` formats.

+----------------+----------------------------------------------------------+
| Format         | When to use                                              |
+================+==========================================================+
| :code:`pickle` | Quick start; list of Python dicts (most intuitive)       |
+----------------+----------------------------------------------------------+
| :code:`npz`    | NumPy-native storage; good for large numeric arrays      |
+----------------+----------------------------------------------------------+
| :code:`hdf5`   | Efficient random access; used internally after preprocess|
+----------------+----------------------------------------------------------+

This page focuses on :code:`pickle` as the entry point. The same field naming and Datahub mappings apply to all formats.

Datapoint schema
----------------

A pickle dataset is a list of datapoints. Each datapoint is a dictionary of attribute name–value pairs. A typical QM-labeled entry includes:

- :code:`number_of_atoms` — integer :math:`N`
- :code:`coordinates` — float array of shape :code:`(N, 3)` in Å (or your chosen length unit)
- :code:`atomic_numbers` — integer array of shape :code:`(N,)`
- :code:`energy` — scalar total energy
- :code:`forces` — float array of shape :code:`(N, 3)` (or raw QC gradients; see below)
- :code:`atomic_charges` — float array of shape :code:`(N,)`
- :code:`total_charge` — integer (defaults to 0 if omitted)
- :code:`dipole` — float array of shape :code:`(3,)`

.. note::
    :code:`coordinates` and :code:`atomic_numbers` define the system. :code:`number_of_atoms` can be inferred from :code:`atomic_numbers`. For PES learning, :code:`energy` and/or :code:`forces` are required targets. Additional fields depend on your task and model architecture.

Standard field mapping
----------------------

Enerzyme maps your attribute names to internal standard names in the training YAML (see :doc:`training`). Common mappings:

+---------------------+---------------+
| Your attribute      | Standard name |
+=====================+===============+
| coordinates         | :code:`Ra`    |
+---------------------+---------------+
| atomic_numbers      | :code:`Za`    |
+---------------------+---------------+
| number_of_atoms     | :code:`N`     |
+---------------------+---------------+
| energy              | :code:`E`     |
+---------------------+---------------+
| forces              | :code:`Fa`    |
+---------------------+---------------+
| atomic_charges      | :code:`Qa`    |
+---------------------+---------------+
| total_charge        | :code:`Q`     |
+---------------------+---------------+
| dipole              | :code:`M2`    |
+---------------------+---------------+

Units and gradients
-------------------

.. caution::
    **Forces vs. gradients.** Quantum chemistry packages often output energy *gradients* :math:`\nabla E`, not forces. Forces are :math:`F = -\nabla E`. Set :code:`negative_gradient: true` in Datahub transforms when your :code:`Fa` targets are raw gradients.

.. note::
    **TeraChem gradients.** The helper script :code:`scripts/picklizer.py` converts TeraChem gradient files from Ha/Bohr to Ha/Å by dividing by 0.5291772108. Keep units consistent with :code:`Hartree_in_E` and :code:`Bohr_in_R` in your model config.

Building a dataset from TeraChem outputs
----------------------------------------

The repository includes :code:`scripts/picklizer.py` for grouping TeraChem output files into a pickle. Each entry in :code:`file_lists` is a dict pointing to per-structure files:

.. code-block:: python

    from scripts.picklizer import picklizer

    file_lists = [
        {
            "coord": "run001/structure.xyz",
            "grad": "run001/grad.xyz",
            "chrg": "run001/mulliken.chrg",
            "dipole": "run001/dipole.txt",
        },
        # ...
    ]
    picklizer(file_lists, output="dataset.pkl", flavor="terachem", provide_Q=-1)

The resulting datapoints use keys :code:`coord`, :code:`grad`, :code:`chrg`, :code:`dipole`, :code:`total_chrg`. Map them in your YAML, for example:

.. code-block:: yaml

    features:
        Ra: coord
        Za: atom_type
        Q: total_chrg
    targets:
        E: energy
        Fa: grad
        Qa: chrg
        M2: dipole
    transforms:
        negative_gradient: true

Generic pickle builder
----------------------

If you already have a parser for your QM package:

.. code-block:: python

    import pickle
    from my_script import parse_qm_output, find_qm_outputs

    datapoints = []
    for qm_output in find_qm_outputs():
        parsed_data = parse_qm_output(qm_output)
        datapoints.append({
            'number_of_atoms': parsed_data['number_of_atoms'],
            'coordinates': parsed_data['coordinates'],
            'atomic_numbers': parsed_data['atomic_numbers'],
            'energy': parsed_data['energy'],
            'forces': parsed_data['forces'],
            'atomic_charges': parsed_data['atomic_charges'],
            'total_charge': parsed_data['total_charge'],
            'dipole': parsed_data['dipole'],
        })

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(datapoints, f)

Preprocess without training
---------------------------

To only preprocess and split a dataset (write HDF5 cache and partition indices) without starting training:

.. code-block:: bash

    enerzyme collect -c train.yaml -o .

Use the same :code:`Datahub` and :code:`Trainer.Splitter` sections as in a training config. This is useful to validate mappings and inspect :code:`processed_dataset_<hash>/` before a long run.

Security and compatibility
--------------------------

.. danger::
    Pickle files are not secure. Do not load pickles from untrusted sources.

.. caution::
    Pickle compatibility depends on Python and library versions. Loading a file created with NumPy 2.x under NumPy 1.x may raise :code:`ModuleNotFoundError: No module named numpy._core`.
