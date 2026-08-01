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
| Equiformer     | yes      | yes    | yes    | via         | Equivariant      |
|                |          |        |        | readout     | graph attention; |
|                |          |        |        |             | higher cost;     |
|                |          |        |        |             | see parity tests |
+----------------+----------+--------+--------+-------------+------------------+
| EquiformerV2   | via      | via    | yes    | via         | SO(2) attention  |
|                | readout  | readout|        | readout     | + S2/gate FFN;   |
|                |          |        |        |             | higher ``lmax``  |
|                |          |        |        |             | with ``mmax``    |
+----------------+----------+--------+--------+-------------+------------------+
| EquiformerV3   | via      | via    | yes    | via         | Merged LN +      |
|                | readout  | readout|        | readout     | SwiGLU-S² +      |
|                |          |        |        |             | smooth-cutoff    |
|                |          |        |        |             | attention        |
+----------------+----------+--------+--------+-------------+------------------+
| So3krates      | via      | via    | yes    | via         | Dual-stream      |
|                | readout  | readout|        | readout     | Euclidean        |
|                |          |        |        |             | transformer;     |
|                |          |        |        |             | SPHC ``χ``       |
+----------------+----------+--------+--------+-------------+------------------+
| EFA            | via      | via    | yes    | via         | So3krates +      |
|                | readout  | readout|        | readout     | Euclidean Fast   |
|                |          |        |        |             | Attention        |
+----------------+----------+--------+--------+-------------+------------------+
| SO3LR+EFA      | via      | via    | yes    | via         | SO3LR stack +    |
|                | readout  | readout|        | readout     | EFA nonlocal     |
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

**EquiformerV2** (:code:`architecture: equiformer_v2`) is a native port of Liao et al. (ICLR 2024) under :code:`enerzyme/models/equiformer_v2/`. It reuses shared :code:`enerzyme/models/so3/` (with EquiformerV2 ``mmax`` rescale / component grids / :code:`SO3_LinearV2`) and implements SO(2) equivariant graph attention + S²/gate feed-forward blocks. The Core emits the same latent contract as eSCN (:code:`atom_feature` / :code:`atom_sphere_feature`). Default production stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. Opt-in :code:`EquiformerV2FeedForwardReadout` wraps the paper energy FFN on :code:`atom_sphere_feature`. All external EquiformerV2 readouts accept :code:`shallow_ensemble_size` (widen last linear → :code:`ShallowEnsembleReduce`). Distinct from Equiformer V1 (e3nn TP attention) and from paper eSCN (message SO(2) without transformer attention). Examples: :code:`enerzyme/config/equiformer_v2_layers_example.yaml`, :code:`equiformer_v2_ffn_readout_example.yaml`, :code:`equiformer_v2_shallow_ensemble_example.yaml`. Parity vs vendored upstream nets: :code:`test/test_equiformer_v2_parity_*.py`.

**EquiformerV3** (:code:`architecture: equiformer_v3`) is a native port of Liao et al. (2026, arXiv:2604.09130) under :code:`enerzyme/models/equiformer_v3/`. It extends shared :code:`so3/` with merged layer norm, SwiGLU-S² activations, fused SO(2) linears (:code:`SO2Linear`), polynomial attention envelopes, and asymmetric S² grids. The Core keeps the same :code:`atom_feature` / :code:`atom_sphere_feature` contract as eSCN/V2. Default stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` (DeNS / stress / direct force heads stay outside the Core). Example: :code:`enerzyme/config/equiformer_v3_layers_example.yaml`. Parity: :code:`test/test_equiformer_v3_parity_*.py`. Enerzymette only needs :code:`architecture: equiformer_v3` plus a resolved :code:`config.yaml`.

**DPA4** (:code:`architecture: dpa4`) implements the EMFA SO(2) descriptor of Li et al. (2026, arXiv:2606.02419) under :code:`enerzyme/models/dpa4/` (:code:`core.py`, :code:`interaction.py`, :code:`so2.py`). Shared SO(3) helpers (C³ envelope, Apache e3x Lebedev tables / :code:`S2LebedevProjector`, packed indexing, :code:`FocusSO2Linear`, focus gated activation, degree RMSNorm, Bessel×C³ RBF, quaternion edge frames + shared e3nn/:code:`Jd` Wigner-D via :code:`A R Aᵀ` into DPA4's Cartesian basis; flat :code:`SO3Grid` / Lebedev behind :code:`S2GridProjector`) live in :code:`enerzyme/models/so3/`. The Core owns the radial cache, geometric/environment seed, m-truncated interactions, and Lebedev-grid equivariant FFN. It emits :code:`atom_feature` / :code:`atom_sphere_feature`, so the standard :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` stack applies. Example: :code:`enerzyme/config/dpa4_layers_example.yaml`. Tests: :code:`test/test_dpa4_core.py`, :code:`test/test_dpa4_parity_ops.py`. Enerzymette only needs :code:`architecture: dpa4` plus a resolved :code:`config.yaml`.

**So3krates** (:code:`architecture: so3krates`) is a native port of Frank et al. (NeurIPS 2022) under :code:`enerzyme/models/so3krates/`, following the So3krates-torch EuclideanTransformer (fused FeatureBlock + GeometricBlock + InteractionBlock). Shared :code:`RealSphericalHarmonics` and :code:`L0Contraction` live in :code:`enerzyme/models/so3/`. The Core emits :code:`atom_feature` (invariant stream ``x``, :code:`feature_irreps: "Fx0e"`) and :code:`atom_sphere_feature` (SPHC ``χ`` with shape ``[N, m_tot]``, **not** eSCN/EquiformerV2's ``[N, (lmax+1)^2, C]`` — :code:`SphereSampleReadout` does not apply). Default stacks use :code:`BernsteinRBF` + :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. Long-range physics (ZBL / electrostatics / dispersion) belong in post-core layers, not inside the Core. Example: :code:`enerzyme/config/so3krates_layers_example.yaml`. Parity: :code:`test/test_so3krates_parity_ops.py`.

**SO3LR** (:code:`architecture: so3lr`) is a So3krates **variant** (Kabylda et al., JACS 2025): the same :code:`So3kratesCore` plus charge/spin embeds and shared physics layers configured for SO3LR — :code:`ZBLRepulsionEnergy` with :code:`switch_off: 1.5`, :code:`ElectrostaticEnergy` with :code:`flavor: SO3LR` (``erf(r/σ)/r``, pretrained ``σ=4``), and :code:`TSQDODispersionEnergy` (Hirshfeld-scaled TS + vdW-QDO; **not** Grimme D3/D4). Post-core heads: :code:`SimpleReadout(Qa)` + :code:`AtomicAffine` (unit scale; element shift), :code:`HirshfeldReadout` (``ha``). Defaults match the public pretrained hyperparams (``r_max=4.5``, ``L≤4``, ``H=128``, ``T=3``, phys cutoff). Example: :code:`enerzyme/config/so3lr_layers_example.yaml`. Tests: :code:`test/test_so3lr.py`. Enerzymette only needs :code:`architecture: so3lr` plus a resolved :code:`config.yaml`.

**Euclidean Fast Attention (EFA)** is a linear-scaling geometry-aware nonlocal plug-in (Frank et al., arXiv:2412.08541) implemented under :code:`enerzyme/models/efa/` (PyTorch; no JAX). Three wiring modes:

1. **SpookyNet** — set Core :code:`use_efa: true` to replace geometry-free :code:`NonlocalInteraction` with :code:`EFABlock` (needs absolute :code:`Ra`). Default remains :code:`false` for checkpoint compatibility.
2. **So3krates lineage** — :code:`architecture: efa` (So3krates + EFA) and :code:`architecture: so3lr_efa` (SO3LR stack + EFA). Both reuse :code:`So3kratesCore` with :code:`era_use_in_iterations` (e.g. ``[0, 1]``). EFA updates the **invariant** stream only; SPHC stays local.
3. **Other / future Cores** — call :code:`EFABlock` or :code:`apply_efa_if_configured` on invariant ``[N, F]`` features with :code:`Ra` and :code:`batch_seg` (see developer guide). Do not feed eSCN/EquiformerV2 ``atom_sphere_feature`` layouts into L=0 EFA.

Examples: :code:`enerzyme/config/efa_layers_example.yaml`, :code:`so3lr_efa_layers_example.yaml`. Tests: :code:`test/test_efa.py`.

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
    MACE, NequIP, Equiformer, EquiformerV2, EquiformerV3, or So3krates — equivariant message passing; tune cutoff and depth.
    Equiformer uses SO(3) graph attention (default MD17-style stack with ``ExpNormalSmearing``
    and :code:`output_mode: feature` emitting full irreps plus :code:`feature_irreps`;
    production default is :code:`SimpleReadout` with :code:`head_type: two_layer` after 0e
    extract, optional :code:`EquiformerGraphAttentionReadout`); prefer smaller irreps /
    fewer layers for enzyme-scale clusters. EquiformerV2 uses SO(2)-reduced attention
    (default :code:`GaussianSmearing` + :code:`atom_feature` / :code:`atom_sphere_feature`)
    and scales more easily to higher :code:`lmax` with truncated :code:`mmax`.
    EquiformerV3 adds merged LN, SwiGLU-S², and smooth-cutoff attention on the same
    latent contract (default :code:`norm_type: merge_layer_norm`,
    :code:`attn/ffn_activation: sep-merge_gates2_swiglu`, :code:`use_envelope: true`).
    So3krates uses dual-stream geometric attention
    (invariants + SPHC; default :code:`BernsteinRBF` + :code:`architecture: so3krates`).
    Charge/dipole use shared readouts outside the Core. Numerical fidelity against the
    official Equiformer MD17 path is covered by :code:`test/test_equiformer_parity_*.py`
    (operator / latent / direct E·F / gradient checks — not the production SimpleReadout stack);
    EquiformerV2 ops/blocks by :code:`test/test_equiformer_v2_parity_*.py`;
    EquiformerV3 ops/blocks by :code:`test/test_equiformer_v3_parity_*.py`;
    So3krates by :code:`test/test_so3krates_parity_ops.py`.

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
