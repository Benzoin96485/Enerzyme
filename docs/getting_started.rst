Getting Started
===============

Enerzyme trains neural network potentials (NNPs) for enzymatic and molecular systems, then uses them for prediction, simulation, and active learning. The tutorials below follow a typical workflow:

.. code-block:: text

    install → prepare data → train → predict / simulate
                                    ↓
                         active learning ← QM annotate ← extract

Optional `Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ tools assist with PLUMED scans, NEB path building, ORCA/TeraChem bridges, and workflow launchers.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started/installation
   getting_started/preparing_dataset
   getting_started/training
   getting_started/prediction
   getting_started/simulation
   getting_started/enhanced_sampling
   getting_started/active_learning
   getting_started/qm_annotation
   getting_started/fragment_extraction
   getting_started/server_and_enerzymette
   getting_started/bond_and_utilities
