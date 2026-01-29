Training a Neural Network Potential
==================================

To train a neural network potential (NNP) with our prepared dataset :code:`dataset.pkl`, you need to write a configuration file :code:`train.yaml` to specify how Enerzyme processes the dataset, builds the NNP, trains it and saves it. Examples for the configuration file can be found in the :code:`Enerzyme/enerzyme/config/train.yaml` directory. Here we introduce the most basic configuration.

Datahub
-------

.. code-block:: yaml
    
    Datahub:
      
We have a :code:`Datahub` section in the configuration file. Datahub connects the dataset to the NNP. First, it is always necessary to specify the following things:

- `data_path`: The path to the dataset. It can be a relative path to the working directory.
- `data_format`: The format of the dataset. For our example picklized dataset :code:`dataset.pkl`, we use :code:`pickle`.

Then, we need to specify the set of attributes for the input of the NNP (`features`), and the attributes that the output of the NNP should fit (`targets`). 

You are almost free to use any strings for the attributes in your picklized dataset. However, Enerzyme needs to **understand** the attributes of each datapoint. Enerzyme internally takes a standard data field name for each attribute to consistently process the data type and the array shape, as well as the data flow in the NNP. Therefore, features and targets require a mapping from our attribute names to the standard data field names.

Standard Data Field Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Enerzyme, all datapoint attributes with a clear physical meaning has a name starting with a capital letter. Atomic attributes, namely the attributes that can be attached to each atom in the system, have a suffix "a". Based on those conventions, the standard data field names adopt common letters of physical quantities for the attributes in our example dataset are as follows:

- :code:`N` for `number_of_atoms`
- :code:`Ra` for `coordinates`
- :code:`Za` for `atomic_numbers`
- :code:`E` for `energy`
- :code:`Fa` for `forces`
- :code:`Qa` for `atomic_charges`
- :code:`Q` for `total_charge`
- :code:`M2` for `dipole`

This allows for the flexibility of using your favorite attribute names. For example, you can replace `coordinates` with `xyz`, and at the same time, change the mapping :code:`Ra: coordinates` to :code:`Ra: xyz`.

Preprocessing the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^

Before sending the dataset to the NNP, some preprocessing are helpful for efficiency and accuracy of the NNP.

Because all NNPs in Enerzyme are formulated as a graph neural network (GNN), the atomic system will be represented as a graph. Obviously, each atom in the system is a node, but the edges depend on how we want atom pairs to interact in the NNP. The data structure we use to store the interaction relationship is called `neighbor_list`. If the `neighbor_list` purely depends on the input features, it can be computed and stored before training, rather than on the fly during training. We currently only support `full` neighbor list, which means all atom pairs are interacting.

The full neighbor list of a system grows as :math:`N^2` with the number of atoms :math:`N`. For large systems, it will be a significant memory overhead for loading the dataset and a huge disk space for storing the `neighbor_list`. However, for a dataset that contains different configurations of the same molecule (with the same atom order), the storage for the `neighbor_list`, as well as the :code:`Za`, :code:`Q`, and :code:`N` fields, can be shared among all datapoints. This trick will be turned on if you set the `compression` to `true`.

It is a common practice to do normalization of the data in deep learning, and it works for NNPs as well. The following transformations are particularly helpful:

- `negative_gradient`: When set to `true`, it flips the sign of the gradient in the training data. Use it when the :code:`Fa` targets are raw gradients instead of forces.
- `atomic_energy`: It takes in a :code:`csv` file with columns :code:`atom_type` (atomic symbol) and :code:`atomic_energy` (in the same unit as the :code:`E` targets). For example

  .. code-block:: csv

        atom_type,atomic_energy
        H,-0.5

  Those `atomic_energy` are typically obtained from the quantum chemistry calculations for the ground state of an isolated atom of the corresponding atom type.
  The transformation will deduct the reference atom energy from the total energy. 

For dataloader's parallelism, the dataset will be stored in a :code:`hdf5` file in a folder :code:`processed_dataset_<hash_string>` in the working directory. The :code:`<hash_string>` is hashed from the configuration of the `data_path`, `data_format` and all preprocessing configurations. When the `preload` option is set to `true`, the dataset with the same hash string will be loaded instead of doing preprocessing again.

Final Datahub Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we want the NNP to fit the energy, forces, atomic charges, and dipole moment from the number of atoms, atomic numbers, total charge, and coordinates, where the forces are from raw gradients of the quantum chemistry, and the atomic energy is compiled in :code:`atomic_energy.csv`, the `Datahub` configuration should look like this:

The final `Datahub` configuration should look like this:

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
        compression: true
        preload: true
        transforms:
            atomic_energy: "atomic_energy.csv"
            negative_gradient: true


Modelhub
---------
The `Modelhub` section is where you specify the NNP architecture and its associated loss functions. We will show how to build a SchNet NNP here.

SchNet is an internal architecture in Enerzyme, which means the whole model is already written in the package with no additional external dependencies. Because you can builds multiple NNPs in the same configuration file, each internal NNP architecture should be specified under `internal_FFs` with its unique `model name` (for example, `FF01`) and the following entries:

- `architecture`: The architecture name of the NNP. SchNet is our choice here.
- `active`: Whether the NNP is active in the training. It will be trained only if it is set to `true`.
- `suffix`: A suffix for the model, which can be used to distinguish different versions of the same model. Leave it empty if not needed.
- `build_params`: The shared architectural hyperparameters for the whole NNP. For SchNet, we need
  
  - `dim_embedding`: The dimension of the embedding of the atom types.
  - `num_rbf`: The number of radial basis functions.
  - `max_Za`: The maximum atomic number in the dataset.
  - `cutoff_sr`: The cutoff radius for the radial basis functions.
  - `Hartree_in_E`: The numerical value of one Hartree in the unit of the :code:`E` targets.
  - `Bohr_in_R`: The numerical value of one Bohr in the unit of the :code:`Ra` targets.
  
  .. note::
    The `Hartree_in_E` and `Bohr_in_R` are used to convert the energy and distance to the unit of the :code:`E` and :code:`Ra` targets. For example, if the :code:`E` targets are in eV, and the :code:`Ra` targets are in Angstrom, you should set `Hartree_in_E` to ~27.2 and `Bohr_in_R` to ~0.53.
  
- `layers`: Enerzyme build an NNP in a modular way to make every component interchangeable. The `architecture` only specifies the message passing core of the GNN-based NNP, which typically takes in atom types, atom embeddings, geometric features, and the neighbor list, and outputs atomic energies and charges. To produce the atom embeddings and geometric features, some pre-core layers are needed. And to postprocess the atomic properties from the core to the final predictions, some post-core layers are needed. All pre-core layers, the `Core`, and the post-core layers are defined by their standard `name` and architectural hyperparameters in `params` if necessary. Shared architectural hyperparameters in `build_params` will be passed to all layers if not specified in their own `params`.

  SchNet's `Core` will prepend a `DistanceLayer` to the layers. So, here we need 

  - `RangeSeparation` to filter the neighbor list based on their distances to only include the atom pairs with in the `cutoff_sr` radius.
  - `GaussianSmearing` to convert atom pair distances to the radial basis function representations. 
  - `RandomAtomEmbedding` to embed the atom types
  
  Then the `Core` can do the job. Here we choose to set `num_interactions` to 4 (4 message passing steps) and `hidden_channels` to 128 (in the MLP). After that, we prefer
  
  - `AtomicAffine` to apply a atom-type-dependent normalization to atomic properties.
  - `ChargeConservation` to ensure total predicted atomic charges is equal to given total charge.
  - `AtomicCharge2Dipole` to compute predicted dipole moment from atomic charges and coordinates.
  - `ElectrostaticEnergy` to compute atomic electrostatic energy from atomic charges and coordinates.
  - `EnergyReduce` to reduce atomic energy terms to total energy.
  - `Force` to compute the forces from the gradients of predicted total energy.

  Then those layers make the NNP ready to do energy, forces, atomic charges, and dipole moment prediction.
  
- `loss`: The loss can be defined as a weighted sum of some common loss functions like `rmse` over training targets. Under the `rmse` key, you can specify the weight attached to each target with their standard data field names. Here we use the energy: forces: atomic charges: dipole moment = 1: 100: 1: 1 in atomic units. Don't forget to convert the numerical values to the unit of the targets.
  
The final `Modelhub` configuration should look like this:

.. code-block:: yaml

    Modelhub:
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

Trainer
-------

The `Trainer` section is where you specify the training protocol. First of all, you need to split the dataset into different partitions.

Splitter
^^^^^^^^

Enerzyme currently only supports random splitting of the dataset into the necessary training partitions and other partitions including validation, test, and any other you want. So you'll need to define under the `Splitter` key:

- `parts`: A list of partition names.
- `ratio`: The ratio (between 0 and 1) of the dataset or the absolute number of datapoints in the partition

For reproducibility, you can set the splitter's own random `seed`, and also `save` the partitioned indices of the dataset in the associated dataset directory once produced, and `preload` them if the hash string of the partition is the same, as the same as the data preprocessing mechanism.

Traning Loop
^^^^^^^^^^^^^

In each training epoch, the dataloader uses `num_workers` processes to load the whole training data into mini-batches of `batch_size`.
The training loop ends when the training epochs reaches the specified `max_epochs`. But before that, early stopping is a common technique to prevent overfitting. It's able to use it when the validation set presents. If the score defined as a weighted sum of the metric on the validation set at an epoch hasn't been improved for `patience` epochs, the training will be stopped.

Optimizer and Learning Rate Scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enerzyme currently only adopts :code:`torch.optim.Adam` as the optimizer. Here we control `learning_rate` and keep others default.
The learning rate scheduler is a `linear` scheduler with warmup. It will increase the learning rate from 0 to the specified `learning_rate` in the `warmup_ratio` of the `max_epochs`, and then decrease it to 0 linearly in the remaining of the `max_epochs`.

System Configuration
^^^^^^^^^^^^^^^^^^^^^
If the model is trained on a single GPU, you can set `cuda` to `true` to benefit from GPU acceleration. The data and model precision are controlled by `dtype`. A unified `seed` can be set for reproducibility of the training process.

.. caution::
    The `seed` here controls :code:`python`, :code:`numpy`, and :code:`pytorch`'s random number generators. However, due to the intrinsic randomness of some CUDA operations, the training process may still be different even with the same `seed`.

Final Trainer Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We assume the dataset of 1000 datapoints is splitted into 7:1:2 for training, validation, and test. It is trained for 10,000 epochs at maximum, at a learning rate of 0.001, with a patience of 50 epochs, a warmup ratio of 0.001 (10 epochs) for the linear scheduler. The early stopping is enabled with a patience of 50 epochs. The dataloader uses 10 processes to load the data into mini-batches of 64. The training computation is done on a `cuda` device with `float32` precision. The final `Trainer` configuration should look like this:

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

Metrics
-------

The `Metrics` section is where you specify the metrics to show for the test set (if present) and compute the weighted sum score on the validation set for early stopping. Here we use the `rmse` metric for the energy, forces, atomic charges, and dipole moment with the same weight as in the loss function.

.. code-block:: yaml

    Metrics:
        rmse:
            E: 1
            Fa: 52.917721
            Qa: 1
            M2: 1.8897261
            Q: 1

Running the Training Job
------------------------

Once combining all the configuration into a single file :code:`train.yaml` in the working directory, you can run the training job in the same directory:

.. code-block:: bash

    enerzyme train -c train.yaml -o .

:code:`-o .` means the output artifacts, including the preprocessed data (and its partitions), and the trained model, will be saved in the current working directory. You will find a newly created log directory `logs` which stores the logs every time :code:`enerzyme train` command line is executed, and the time of each important event during the data preprocessing, NNP building, a backup output of the full configuration, as well as the training loss, validation loss, each term of the metric score on the validation set, the early stopping information, and the final metric terms on the test set when the training is finished. A formatted version of the configuration file will be saved in the current working directory as well. **It is extremely useful to keep this file for future use.**