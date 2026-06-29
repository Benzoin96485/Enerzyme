Bond Assignment
===============

:code:`enerzyme bond` assigns chemical bonds to PDB cluster structures, typically from `QuantumPDB <https://github.com/davidkastner/quantumPDB>`_ workflows.

Command
-------

.. code-block:: bash

    enerzyme bond -p cluster.pdb -m cluster.mol -i cluster.png -t ligands.sdf

Arguments
---------

- :code:`-p` — input PDB (cluster build output)
- :code:`-m` — output MOL with bond orders
- :code:`-i` — optional 2D depiction image
- :code:`-t` — template SDF (e.g. QuantumPDB :code:`ligands.sdf`)

Implementation
--------------

:code:`enerzyme/bond/bond.py`:

- :code:`pdb2mol()` — RDKit PDB import
- :code:`bond_with_template()` — match ligand connectivity from template SDF
- Residue-specific fixes (e.g. carboxylate protonation on ASP/GLU)
- Metal/ligand handling heuristics

Use cases
---------

- Prepare :code:`cluster.mol` as :code:`Extractor.reference_mol_path`
- Generate chemically consistent inputs for QM suppliers
- Visualize cluster connectivity (:code:`-i`)

Requirements
------------

RDKit (included in standard Enerzyme conda env). Template quality strongly affects ligand bond orders in heterogeneous clusters.
