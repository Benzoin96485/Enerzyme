Installation
============

This guide covers installing Enerzyme from source on the :code:`devel` branch. For a pinned release-style environment (Python 3.13, PyTorch 2.8, and related packages), see the version table in the repository :code:`README.md`. The conda workflow below is the recommended development setup.

Clone the repository
--------------------

.. code-block:: bash

    git clone https://github.com/Benzoin96485/Enerzyme.git
    cd Enerzyme
    git checkout devel

Create a conda environment
--------------------------

We recommend creating a conda environment from :code:`requirements.yaml`:

.. code-block:: bash

    conda env create -f requirements.yaml
    conda activate enerzyme

The file installs core dependencies including NumPy, PyTorch, ASE, RDKit, :code:`torch_geometric`, and Lightning.

Install torch-scatter
---------------------

:code:`torch-scatter` must match your PyTorch, CUDA, Python, and platform. Go to https://data.pyg.org/whl/ and pick the wheel that matches your stack. For example, with PyTorch 2.5.1, CUDA 12.4, Python 3.12, and Linux x86_64:

.. code-block:: bash

    pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl

Install Enerzyme
----------------

From the repository root:

.. code-block:: bash

    pip install -e .

Optional dependencies
---------------------

Some workflows need extra packages or external programs that are **not** installed by default:

- **NequIP models** — :code:`nequip`
- **XPaiNN models** — :code:`XequiNet`, SciPy, PySCF, pydantic
- **PLUMED enhanced sampling** — :code:`py-plumed` and a PLUMED-enabled build
- **Prediction server** — Flask and Waitress (included in README version pins)
- **QM annotation** — TeraChem (licensed), RDKit
- **Bond assignment** — RDKit, QuantumPDB cluster PDB and template SDF
- **Enerzymette launchers** — install from the `Enerzymette repository <https://github.com/Benzoin96485/Enerzymette>`_ with :code:`pip install -e .`

Verify the installation
-----------------------

.. code-block:: bash

    python -c "import enerzyme"
    enerzyme -h
    enerzyme predict -h
    enerzyme simulate -h

If all commands print help text without import errors, the core install is ready.
