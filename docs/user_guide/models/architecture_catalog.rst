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
| AlphaNet       | varies   | varies | partial| varies      | See config TODOs |
+----------------+----------+--------+--------+-------------+------------------+
| AllScAIP       | yes      | yes    | yes    | yes         | **Experimental** |
|                |          |        |        |             | (魔改 / not      |
|                |          |        |        |             | recommended)     |
+----------------+----------+--------+--------+-------------+------------------+
| eSCN           | via      | via    | yes    | via         | Native paper     |
|                | readout  | readout|        | readout     | SO(2) GNN; no    |
|                |          |        |        |             | fairchem needed  |
+----------------+----------+--------+--------+-------------+------------------+
| Equiformer     | yes      | yes    | yes    | no          | Equivariant      |
|                |          |        |        |             | graph attention; |
|                |          |        |        |             | higher cost;     |
|                |          |        |        |             | see parity tests |
+----------------+----------+--------+--------+-------------+------------------+
| EquiformerV2   | via      | via    | yes    | no          | SO(2) attention  |
|                | readout  | readout|        |             | + S2/gate FFN;   |
|                |          |        |        |             | higher ``lmax``  |
|                |          |        |        |             | with ``mmax``    |
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

**eSCN** (:code:`architecture: escn`) is a native port of Passaro & Zitnick (2023) SO(3)→SO(2) convolutions under :code:`enerzyme/models/escn/`, backed by shared primitives in :code:`enerzyme/models/so3/`. The Core emits scalar :code:`atom_feature` (spherical :code:`l=0`, with :code:`feature_irreps: "Cx0e"` and :code:`dim_feature_out = C`) and :code:`atom_sphere_feature` (full :code:`(lmax+1)^2` SH coefficients after message :code:`rotate_inv`; :code:`mmax` only reduces edge-frame SO(2)). Default stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` (energy-conserving :code:`Fa=-∇E`, so edge frames stay in the autograd graph). Opt-in :code:`SphereSampleReadout` applies the paper's S² sampling head to any atomic property fields in :code:`output_fields`; :code:`vector_output_fields: [Fa]` is the paper-style direct-force alternative. Examples: :code:`enerzyme/config/escn_layers_example.yaml`, :code:`escn_sphere_readout_example.yaml`. No fairchem dependency. Ops/block numerical parity vs vendored fairchem v1 lives in :code:`test/test_escn_parity_*.py` (forward only; force conservation is covered by unit tests).

**EquiformerV2** (:code:`architecture: equiformer_v2`) is a native port of Liao et al. (ICLR 2024) under :code:`enerzyme/models/equiformer_v2/`. It reuses shared :code:`enerzyme/models/so3/` (with EquiformerV2 ``mmax`` rescale / component grids / :code:`SO3_LinearV2`) and implements SO(2) equivariant graph attention + S²/gate feed-forward blocks. The Core emits the same latent contract as eSCN (:code:`atom_feature` / :code:`atom_sphere_feature`). Default production stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. Opt-in :code:`EquiformerV2FeedForwardReadout` wraps the paper energy FFN on :code:`atom_sphere_feature`. Distinct from Equiformer V1 (e3nn TP attention) and from paper eSCN (message SO(2) without transformer attention). Example: :code:`enerzyme/config/equiformer_v2_layers_example.yaml`. Parity vs vendored upstream nets: :code:`test/test_equiformer_v2_parity_*.py`.

**UMA** (:code:`architecture: uma_qs`) requires the :code:`fairchem` package. The Core wraps Meta's UMA / eSCN-MD backbone as an atom descriptor under :code:`enerzyme/models/esen/` (name is historical; this is **not** the 2023 paper eSCN). Shared layers such as :code:`SimpleReadout`, :code:`HierachicalReadout`, and :code:`SpinConservation` predict atomic or molecular charge/spin outside the Core. Pair with :code:`aselmdb` datasets that provide :code:`Q` / :code:`S` (and optionally :code:`Qa` / :code:`Sa`).

.. warning::

   **AllScAIP** (:code:`architecture: AllScAIP`) in Enerzyme is an **experimental, heavily adapted (魔改)** port of fairchem-style attention IP, not a drop-in of the upstream model.
   It is **not recommended as a primary production architecture** yet. Prefer PhysNet, SpookyNet, MACE, or UMA for real campaigns; keep AllScAIP for research / ablation only (:code:`train.yaml` FF08 stays :code:`active: false` by default). The Core emits :code:`atom_feature` only — always attach :code:`SimpleReadout` (or NSE heads) before energy/charge physics layers.

Selection guidelines
--------------------

**Baseline / tutorial**
    SchNet — minimal dependencies, charge-aware stacks available.

**Production QM-labeled clusters with charge and solvent**
    PhysNet or SpookyNet — long-range electrostatics, optional dispersion layers.

**Maximum accuracy on diverse geometries**
    MACE, NequIP, Equiformer, or EquiformerV2 — equivariant message passing; tune cutoff and depth.
    Equiformer uses SO(3) graph attention (default MD17-style stack with ``ExpNormalSmearing``
    and :code:`output_mode: feature` emitting full irreps plus :code:`feature_irreps`;
    production default is :code:`SimpleReadout` with :code:`head_type: two_layer` after 0e
    extract, optional :code:`EquiformerGraphAttentionReadout`); prefer smaller irreps /
    fewer layers for enzyme-scale clusters. EquiformerV2 uses SO(2)-reduced attention
    (default :code:`GaussianSmearing` + :code:`atom_feature` / :code:`atom_sphere_feature`)
    and scales more easily to higher :code:`lmax` with truncated :code:`mmax`.
    Charge/dipole use shared readouts outside the Core. Numerical fidelity against the
    official Equiformer MD17 path is covered by :code:`test/test_equiformer_parity_*.py`
    (operator / latent / direct E·F / gradient checks — not the production SimpleReadout stack);
    EquiformerV2 ops/blocks by :code:`test/test_equiformer_v2_parity_*.py`.

**Active learning with force variance**
    Any architecture with :code:`ShallowEnsembleReduce` or :code:`committee_size` > 1.

**Not for production yet**
    AllScAIP — experimental Enerzyme adaptation; see warning above.

Spin and charge
---------------

Charge-aware stacks need :code:`Q` (and often :code:`ChargeConservation`). SpookyNet-style models may use :code:`ElectronicEmbedding` for :code:`charge` and :code:`spin` (:code:`S` / multiplicity). Match simulation :code:`System.charge` and :code:`multiplicity` to training data conventions.

Reference configs
-----------------

Full multi-architecture examples: :code:`enerzyme/config/train.yaml`. Enable one :code:`FF` entry at a time when starting (:code:`active: true`).

NSE and flow matching
---------------------

Neural Spin Equilibration (:code:`NSEReadout` / :code:`NeuralSpinChargeEquilibration`) lives in shared layers outside Core. Existing architectures expose :code:`output_mode: feature` so Core emits :code:`atom_feature` for modular Q/S heads.

:code:`architecture: uma_flow_qs` uses continuous-flow charge/spin generation on top of the UMA Core. Install :code:`pip install -e ".[flow]"` for :code:`torchdiffeq`, plus fairchem. Example layer stacks: :code:`enerzyme/config/uma_qs_layers_example.yaml` and :code:`enerzyme/config/uma_flow_qs_layers_example.yaml`.

