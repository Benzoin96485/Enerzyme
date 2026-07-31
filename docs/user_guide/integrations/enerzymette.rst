Enerzymette Integration
=======================

`Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_ is a separate package of workflow scripts and launchers around Enerzyme. It is **not** installed with core Enerzyme.

Application: :code:`example/NNP4MTase` in the Enerzyme repository (methyltransferase active learning, scan, and NEB).

For a **tiny** ASELMDB / annotate / Enerzymette AL smoke (vendored fixtures, repo-relative paths), see :code:`example/L3-COMT-aselmdb-smoke/`.

Install
-------

.. code-block:: bash

    git clone https://github.com/Benzoin96485/Enerzymette.git
    cd Enerzymette
    pip install -e .

Main launchers
--------------

:code:`enerzymette enerzyme_active_learning`
    Outer AL loop: simulate → predict/extract → annotate → train. Manages a per-task **structure pool** and rewrites YAML each round. See :doc:`/user_guide/workflows/active_learning`.

:code:`enerzymette enerzyme_scan`
    Chained reaction-coordinate scans (bond-distance or PLUMED CV). Reads system settings from a TeraChem input or a YAML **scan config** (:code:`-q`: :code:`reference_pdb`, :code:`freeze_index_types`, scan bond; optional :code:`charge` skips :code:`enerzyme bond` charge derivation). PLUMED mode requires :code:`-pp` and :code:`-psc`.

:code:`enerzymette enerzyme_neb`
    NNP-driven NEB via ORCA ExtOpt and :code:`enerzyme listen`. :code:`-q` accepts a TeraChem reference **or** YAML **neb config** (same freeze/charge pattern as scan). Pass :code:`-cp` for external-calculator shell mode and optional :code:`-t` for an initial TS guess. See :doc:`/getting_started/enhanced_sampling` and :doc:`/getting_started/server_and_enerzymette`.

PLUMED CV plugins
-----------------

Plugins live under :code:`enerzymette/plumed_config_generator/`. Enerzymette registers built-in keys (e.g. :code:`sammt`) and resolves module paths with :code:`get_plumed_patch(key)`. Pass the resolved module to Enerzyme:

.. code-block:: bash

    enerzyme simulate ... -pp <plugin_key_or_module.py>

**Contract (current).** A plugin module exposes a :code:`PlumedConfigGenerator` subclass (for example :code:`SAMMTConfigGenerator`). The simulation YAML selects the class and method:

.. code-block:: yaml

    plumed_config_generator:
        name: SAMMTConfigGenerator
        method: standard_steered_md   # or naive_steered_md, scan

Enerzyme instantiates the class with the current :code:`ase.Atoms` object and YAML parameters under :code:`sampling.params.plumed_config`, then calls the named method to produce :code:`plumed.dat`.

To register a new built-in key, use :code:`register_plumed_cv_plugin()` in the plugin package. Optional **proton-transfer** strategies are separate plugins under :code:`proton_transfer.py` (for example :code:`local_opes`).

Full developer and user reference: `Enerzymette PLUMED plugin README <https://github.com/Benzoin96485/Enerzymette/blob/main/enerzymette/plumed_config_generator/README.md>`_.

Utility commands
----------------

:code:`enerzymette idpp`
    IDPP path between reactant/product XYZ for NEB seeding.

:code:`enerzymette terachem_timing`
    Wall-time stats from TeraChem output files.

:code:`enerzymette orca_terachem_request`
    ORCA ExtOpt → TeraChem gradient bridge. See :doc:`/user_guide/integrations/orca_terachem`.

:code:`enerzymette update_terachem_scan`
    Refresh bond-scan coordinates in a TeraChem input after a structure update (:code:`-i`, :code:`-s`, :code:`-o`).

Active-learning launcher flags (summary)
----------------------------------------

In addition to template paths (:code:`-sc`, :code:`-ec`, :code:`-ac`, :code:`-tc`) and plugin keys (:code:`-cp`, :code:`-pp`):

- :code:`--initial-scan` / :code:`-nis` — run :code:`plumed_scan` jobs before iteration 0 and seed :code:`structure_pool/` from local minima
- :code:`--initial-structures-config` — multi-system manifest YAML (mutually exclusive with :code:`-rp`, :code:`-ts`, :code:`-ix`, and :code:`--initial-scan`)
- :code:`-ix` — single-system custom initial XYZ (overrides :code:`System.structure_file` in the simulation template for pool initialization)
- :code:`-cl` / :code:`--continual_learning` — resume in-round training with :code:`Trainer.resume: 2` on later iterations
- :code:`--reset_parameters` — reinitialize weights at iteration 0; switch extraction to random picking for that round
- :code:`-mc` — override model config path passed to :code:`enerzyme simulate` / :code:`train`

See :doc:`/getting_started/active_learning` for structure-pool layout and manifest format.

Boundary with Enerzyme
----------------------

+------------------+------------------------+---------------------------+
| Responsibility   | Enerzyme               | Enerzymette               |
+==================+========================+===========================+
| Model training   | yes                    | launches :code:`train`    |
+------------------+------------------------+---------------------------+
| MD / PLUMED      | yes (:code:`simulate`) | writes per-round YAML     |
+------------------+------------------------+---------------------------+
| Task scheduling  | no                     | yes                       |
+------------------+------------------------+---------------------------+
| QM engines       | :code:`annotate`       | ORCA bridge utilities     |
+------------------+------------------------+---------------------------+

Architecture selection (e.g. :code:`Equiformer`, :code:`equiformer_v2`, :code:`equiformer_v3`, :code:`escn`) is entirely in the Enerzyme training /
model YAML (:code:`Modelhub.*.architecture`). Enerzymette only rewrites dataset paths,
suffixes, and pretrain paths — no Enerzymette code changes are required when adding a
new Enerzyme Core. Future shallow-ensemble or larger-degree EquiformerV2 configs remain
YAML-only as long as checkpoint paths stay stable.

Recommended task directory
--------------------------

Stable inputs at task root: :code:`al.sh`, :code:`cluster.xyz`, :code:`cluster.mol`, :code:`config/*.yaml`. Generated iteration folders :code:`FFxx_*`, plus :code:`structure_pool/` and :code:`structure_pool.json` when using the AL launcher.
