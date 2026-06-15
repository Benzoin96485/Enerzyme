Architecture Catalog
====================

Enerzyme supports several internal architectures and external wrappers. Choose based on targets (energy/forces only vs charge/dipole), system size, equivariance needs, and optional dependencies.

Internal architectures
----------------------

+----------------+----------+--------+--------+-------------+------------------+
| Architecture   | Charge   | Dipole | Modular| Shallow ens.| Notes            |
+================+==========+========+========+=============+==================+
| SchNet         | yes      | yes    | partial| yes         | Good baseline    |
+----------------+----------+--------+--------+-------------+------------------+
| PhysNet        | yes      | yes    | yes    | yes         | Electrostatics,  |
|                |          |        |        |             | D3/D4 optional   |
+----------------+----------+--------+--------+-------------+------------------+
| SpookyNet      | yes      | yes    | yes    | yes         | Enzyme-scale     |
|                |          |        |        |             | clusters         |
+----------------+----------+--------+--------+-------------+------------------+
| MACE           | yes      | yes    | partial| yes         | Equivariant,     |
|                |          |        |        |             | higher cost      |
+----------------+----------+--------+--------+-------------+------------------+
| LEFTNet        | varies   | varies | partial| varies      | See config TODOs |
+----------------+----------+--------+--------+-------------+------------------+

External wrappers
-----------------

+----------------+------------------------------------------+
| Architecture   | Extra install                            |
+================+==========================================+
| NequIP         | :code:`nequip`                           |
+----------------+------------------------------------------+
| XPaiNN         | :code:`XequiNet` and dependencies        |
+----------------+------------------------------------------+

External models are declared under :code:`Modelhub.external_FFs` with the same :code:`active` / :code:`layers` pattern where supported.

Selection guidelines
--------------------

**Baseline / tutorial**
    SchNet — minimal dependencies, charge-aware stacks available.

**Production QM-labeled clusters with charge and solvent**
    PhysNet or SpookyNet — long-range electrostatics, optional dispersion layers.

**Maximum accuracy on diverse geometries**
    MACE or NequIP — equivariant message passing; tune cutoff and depth.

**Active learning with force variance**
    Any architecture with :code:`ShallowEnsembleReduce` or :code:`committee_size` > 1.

Spin and charge
---------------

Charge-aware stacks need :code:`Q` (and often :code:`ChargeConservation`). SpookyNet-style models may use :code:`ElectronicEmbedding` for :code:`charge` and :code:`spin` (:code:`S` / multiplicity). Match simulation :code:`System.charge` and :code:`multiplicity` to training data conventions.

Reference configs
-----------------

Full multi-architecture examples: :code:`enerzyme/config/train.yaml`. Enable one :code:`FF` entry at a time when starting (:code:`active: true`).
