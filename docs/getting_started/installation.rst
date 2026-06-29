Installation
============

This guide covers installing Enerzyme from source on the :code:`main` branch.

Clone the repository
--------------------

.. code-block:: bash

    git clone https://github.com/Benzoin96485/Enerzyme.git
    cd Enerzyme

Create a conda environment
--------------------------

We recommend creating a conda environment from :code:`requirements.yaml`:

.. code-block:: bash

    conda env create -f requirements.yaml
    conda activate enerzyme

The file installs core dependencies. For a pinned release for reproducibility of paper results, install from the :code:`requirements.yaml` in the corresponding subdirectory of :code:`examples/`. The environment may have a different name from :code:`enerzyme`.

Install torch-scatter
---------------------

:code:`torch-scatter` must match your PyTorch, CUDA, Python, and platform. Go to https://data.pyg.org/whl/ and pick the wheel that matches your stack. For example, with PyTorch 2.5.1, CUDA 12.4, Python 3.12, and Linux x86_64:

.. code-block:: bash

    pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp312-cp312-linux_x86_64.whl

If the wheel for your stack is not found, you can build it from source.

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
- **QM annotation with TeraChem** — TeraChem (licensed)
- **Bond assignment** — QuantumPDB https://github.com/hjkgrp/quantumPDB (optional but recommended)
- **Enerzymette launchers** — install from the Enerzymette repository https://github.com/Benzoin96485/Enerzymette with :code:`pip install -e .`
- **fairchem** — install from the fairchem repository https://github.com/Benzoin96485/fairchem 

Verify the installation
-----------------------

.. code-block:: bash

    python -c "import enerzyme"
    enerzyme -h
    enerzyme predict -h
    enerzyme simulate -h

If all commands print help text without import errors, the core install is ready.
