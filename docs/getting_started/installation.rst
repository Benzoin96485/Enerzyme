Installation
============

We recommend first creating a conda environment with a yaml file :code:`requirements.yaml` by

.. code-block:: bash

    conda env create -f requirements.yaml

which includes the following dependencies:

.. code-block:: yaml

    name: enerzyme
    channels:
    - conda-forge
    - defaults
    dependencies:
        # Base depends
    - python
    - pip
        # Pip-only installs
    - pip:
        - numpy             # for numerical computing
        - h5py              # for HDF5 file support
        - tqdm              # for progress bars
        - ase               # for simmulation environment
        - joblib            # for checkpointing
        - addict            # for passing parameters to submodules
        - pandas            # for saving prediction results
        - torch             # for deep neural networks
        - scikit-learn      # for data splitting
        - transformers      # for training schedulers
        - torch-ema         # for EMA training
        - pyyaml            # for parsing configuration files
        - torch_geometric   # for graph neural networks
        - rdkit             # for chemoinformatics
        - e3nn              # for equivariant neural networks
        - lightning         # for multi-GPU training

Then activate the environment:

.. code-block:: bash

    conda activate enerzyme

and go to https://data.pyg.org/whl/ and find the latest wheel file for :code:`torch-scatter` that matches your PyTorch version, CUDA version, Python version, and platform. For example, if you are using PyTorch 2.5.1, CUDA 12.4, Python 3.12, and Linux x86_64 platform, you can click on the `torch-2.5.1+cu124 <https://data.pyg.org/whl/torch-2.5.1%2Bcu124.html>`_ link and find the link to the wheel file `torch_scatter-2.1.2+pt25cu124-cp312-cp312-linux_x86_64.whl <https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl>`_.

Then install the wheel file by

.. code-block:: bash

    pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl

Finally, install the package in the repository root directory by

.. code-block:: bash

    pip install -e .

Check the library installation by

.. code-block:: python

    import enerzyme

and the command line interface by

.. code-block:: bash

    enerzyme -h




