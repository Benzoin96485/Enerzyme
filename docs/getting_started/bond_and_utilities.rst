Bond Assignment and Utility Tools
==================================

This page covers Enerzyme CLI utilities and `Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ helper scripts that support Enerzyme workflows but are not part of core training.

Bond order assignment (:code:`enerzyme bond`)
---------------------------------------------

Guess bond orders for enzymatic cluster structures from a PDB file. Compatible with `QuantumPDB <https://github.com/davidkastner/quantumPDB>`_ cluster outputs.

.. code-block:: bash

    enerzyme bond -p cluster.pdb -m cluster.mol -i cluster.png -t ligands.sdf

Arguments:

- :code:`-p` — input PDB (QuantumPDB cluster build output)
- :code:`-m` — output MOL file with assigned bonds
- :code:`-i` — optional 2D structure image
- :code:`-t` — template SDF (e.g. QuantumPDB :code:`ligands.sdf`)

Requires RDKit. Use the resulting MOL/SDF as reference connectivity for :doc:`fragment_extraction` or QM input preparation.

Dataset preprocessing only (:code:`enerzyme collect`)
-----------------------------------------------------

Run Datahub preprocessing and splitting without training:

.. code-block:: bash

    enerzyme collect -c train.yaml -o .

See :doc:`preparing_dataset`.

Enerzymette: IDPP path interpolation
------------------------------------

Build an initial minimum-energy path between reactant and product for NEB:

.. code-block:: bash

    enerzymette idpp -r reactant.xyz -p product.xyz -o path.xyz -n 25 -c terachem.inp

- Uses ASE IDPP interpolation (`J. Chem. Phys. 2014, 140, 214106 <https://doi.org/10.1063/1.4878664>`_)
- :code:`-c` — TeraChem input containing fixed-atom constraints applied during interpolation
- If reactant/product XYZ contain trajectories, the **last frame** is used

Use the output multi-frame XYZ as :code:`System.structure_file` for :code:`task: neb`, or let Enerzyme interpolate from two endpoints (:doc:`simulation`). IDPP complements built-in :code:`interpolation.method: idpp` in NEB configs.

Enerzymette: TeraChem timing
----------------------------

.. code-block:: bash

    enerzymette terachem_timing -f terachem.out

Summarizes wall time across SCF iterations and flags incomplete jobs. Useful before merging QC data into training sets (:doc:`qm_annotation`).

Enerzymette: ORCA ↔ TeraChem bridge
-----------------------------------

.. code-block:: bash

    enerzymette orca_terachem_request -i orca.extinp.tmp -t terachem_template.inp

Combines ORCA ExtOpt with TeraChem gradients. Full setup is documented in :doc:`qm_annotation`.

Enerzymette: flexible scan launcher
-----------------------------------

Automated opt → scan → opt loops for reaction-coordinate exploration:

.. code-block:: bash

    enerzymette enerzyme_scan \
        -r reactant.xyz \
        -o scan_out/ \
        -m model_dir/ \
        -q scan_config.yaml \
        -pp sammt \
        -psc cv_params.yaml \
        -n 25

- :code:`-q` — TeraChem input with :code:`constraint_freeze` / :code:`constraint_scan`, **or** a YAML scan config (see :doc:`enhanced_sampling`)
- :code:`-pp` / :code:`-psc` — PLUMED CV scan; both flags required together
- :code:`enerzymette update_terachem_scan` — refresh bond-scan coordinates in a TeraChem input after geometry update

Enerzymette: NEB launcher
-------------------------

NNP-driven NEB via ORCA and :code:`enerzyme listen`:

.. code-block:: bash

    enerzymette enerzyme_neb \
        -r reactant.xyz \
        -p product.xyz \
        -o neb_out/ \
        -m model_dir/ \
        -q reference.in \
        -c server.yaml \
        -n 25 -b 5000

See :doc:`server_and_enerzymette` for server setup.

Quick reference
---------------

- :code:`enerzyme bond` (Enerzyme) — PDB to MOL connectivity
- :code:`enerzyme collect` (Enerzyme) — preprocess and split only
- :code:`enerzymette idpp` (Enerzymette) — NEB initial path
- :code:`enerzymette terachem_timing` (Enerzymette) — QC job monitoring
- :code:`enerzymette orca_terachem_request` (Enerzymette) — ORCA optimizer with TeraChem QC
- :code:`enerzymette enerzyme_scan` (Enerzymette) — batch PLUMED or bond-distance scans
- :code:`enerzymette enerzyme_neb` (Enerzymette) — NNP-driven NEB via ORCA ExtOpt
- :code:`enerzymette update_terachem_scan` (Enerzymette) — refresh TeraChem scan coordinates
- :code:`enerzymette enerzyme_active_learning` (Enerzymette) — active-learning campaign launcher

Install Enerzymette with :code:`pip install -e .` in its repository; it is not bundled with Enerzyme.
