Preparing a Neural Network Potential Dataset
===========================================

To train a neural network potential (NNP), you need to prepare a dataset of atomic systems and their corresponding attributes. Enerzyme now supports the :code:`pickle`, :code:`npz`, and :code:`hdf5` formats. As a quick start, we introduce the :code:`pickle` format here as it is the most intuitive one.

Pickle file is a binary file that can store Python objects. In Enerzyme, the `dataset`, as a Python object stored in a pickle file, is a list of `datapoints`, each of which is a dictionary of attribute name-value pairs.

In the following example, we consider the following attributes of a datapoint:

- `number_of_atoms`: the number of atoms in the system, an integer. Let :code:`N` be its value in the following.
- `coordinates`: the atomic Cartesian coordinates of the system, a float array of shape :code:`(N, 3)`.
- `atomic_numbers`: the atomic numbers of the atoms, a integer array of shape :code:`(N,)`.
- `energy`: the total energy of the system, a float.
- `forces`: the forces on the atoms, a float array of shape :code:`(N, 3)`.
- `atomic_charges`: the partial charges of the atoms, a float array of shape :code:`(N,)`.
- `total_charge`: the total charge of the system, an integer.
- `dipole`: the dipole moment of the system, a float array of shape :code:`(3,)`.

.. note::
    `number_of_atoms`, `coordinates`, `atomic_numbers`, `total_charge` can be easily obtained from the quantum chemistry input files, while `energy`, `forces`, `atomic_charges`, and `dipole` should be parsed from the quantum chemistry output files.

    For NNP training, the `coordinates`, `atomic_numbers` are necessary to define the system, while `number_of_atoms` can be counted from those, and `total_charge` can be assumed to be zero. For a potential energy surface learning task, `energy` and/or `forces` are necessary for the training objective.

    The attributes are not limited to the ones listed above and also depend on the task you are training for, the available data from quantum chemistry,  and the architecture of the neural network potential. You can add any attributes you want to the datapoint.

Assume that we wrote a script :code:`my_script.py` in the current working directory, which has a function :code:`parse_qm_output` that parses the output of a quantum chemistry calculation and returns a dictionary of such information, and a function :code:`find_qm_outputs` that finds all the relevant quantum chemistry output files. Then the dataset :code:`dataset.pkl` can be prepared as follows:

.. code-block:: python

    import pickle
    from my_script import parse_qm_output, find_qm_outputs
    
    qm_outputs = find_qm_outputs() # all qm output files
    
    datapoints = []
    for qm_output in qm_outputs:
        parsed_data = parse_qm_output(qm_output)
        datapoint = {
            'number_of_atoms': parsed_data['number_of_atoms'],
            'coordinates': parsed_data['coordinates'],
            'atomic_numbers': parsed_data['atomic_numbers'],
            'energy': parsed_data['energy'],
            'forces': parsed_data['forces'],
            'atomic_charges': parsed_data['atomic_charges'],
            'total_charge': parsed_data['total_charge'],
            'dipole': parsed_data['dipole']
        }

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(datapoints, f)

.. danger::
    Pickle files are not secure. You should not trust the data in a pickle file from an untrusted source.

.. caution::
    Pickle files don't guarantee the version compatibility. When loading the pickle file, you should use the same version of Python and relevant libraries as the one used to dump the pickle file. For example, if you used Numpy 2.x to store the arrays in the dataset, but now are using Numpy 1.x to load it, it will raise an error like :code:`ModuleNotFoundError: No module named numpy._core`.