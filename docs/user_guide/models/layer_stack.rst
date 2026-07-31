Layer Stack
===========

Enerzyme builds models as ordered **layers** rather than monolithic classes. The :code:`architecture` name selects a :code:`Core`; pre- and post-core layers handle geometry, embeddings, physics, and output reductions.

Registered layers
-----------------

From :code:`enerzyme/models/layers/`:

**Geometry**
    :code:`DistanceLayer`, :code:`RangeSeparationLayer`, :code:`RadiusGraphLayer`

**Radial basis**
    :code:`GaussianSmearing`, :code:`ExponentialGaussianRBFLayer`, :code:`ExponentialBernsteinRBFLayer`, :code:`BesselRBFLayer`, :code:`BernsteinRBFLayer`, :code:`SincRBFLayer`

**Embeddings**
    :code:`RandomAtomEmbedding`, :code:`NuclearEmbedding`, :code:`ElectronicEmbedding`, :code:`ChargeSpinEmbedding` (SO3LR-style), :code:`ScalarDenseEmbedding`, :code:`GatherAtomEmbedding` (optional :code:`scale_by_sqrt_count` for SO3LR)

**Core**
    Architecture-specific message passing (:code:`Core` with :code:`architecture` in Modelhub)

**Physics / post-processing**
    :code:`AtomicAffine`, :code:`ChargeConservation`, :code:`ElectrostaticEnergy` (flavors: SpookyNet / PhysNet / SO3LR), :code:`AtomicCharge2Dipole`, :code:`GrimmeD3Energy`, :code:`GrimmeD4Energy`, :code:`TSQDODispersionEnergy` (SO3LR; not Grimme), :code:`ZBLRepulsionEnergy` (optional :code:`switch_off` for SO3LR)

**Output**
    :code:`EnergyReduce`, :code:`Force`, :code:`ShallowEnsembleReduce`

**Readouts**
    :code:`SimpleReadout` — per-atom MLP over scalar features. With equivariant Cores
    that set :code:`feature_irreps`, it extracts even-scalar (:code:`0e`) channels first,
    then applies :code:`dense` / :code:`residual_*` / :code:`two_layer` heads.
    Use :code:`head_type: equiformer_linear_rs` for the official Equiformer MD17 scalar
    energy MLP (:code:`LinearRS` → :code:`normalize2mom(SiLU)` → :code:`LinearRS`).
    :code:`EquiformerGraphAttentionReadout` — separate GraphAttention head over full
    irreps + graph edges (not mixed into SimpleReadout); use when you want an
    attention-style multi-field atomic scalar head.
    :code:`PartialChargeReadout` / :code:`HirshfeldReadout` — SO3LR charge and Hirshfeld-ratio heads.

Typical charge-aware stack
--------------------------

.. code-block:: yaml

    layers:
      - name: RangeSeparation
      - name: ExponentialBernsteinRBF
      - name: NuclearEmbedding
      - name: ElectronicEmbedding
        params:
            attribute: charge
      - name: Core
        params:
            num_modules: 6
            shallow_ensemble_size: 10
      - name: AtomicAffine
      - name: ChargeConservation
      - name: ElectrostaticEnergy
        params:
            flavor: SpookyNet
            dielectric_constant: 10.0
      - name: AtomicCharge2Dipole
      - name: EnergyReduce
      - name: ShallowEnsembleReduce
        params:
            var: [E]
            train_only: true
      - name: Force
      - name: ShallowEnsembleReduce
        params:
            var: [E, Fa]
            eval_only: true

Layer ordering matters
----------------------

1. Build geometric features (range separation, RBFs)
2. Embed atoms and optional scalar features
3. Message passing (:code:`Core`)
4. Normalize atomic outputs (:code:`AtomicAffine`)
5. Enforce physics (charge conservation, electrostatics, dispersion)
6. Reduce to molecular properties (:code:`EnergyReduce`)
7. Optional ensemble statistics
8. Analytic forces via autograd (:code:`Force`)

Shared :code:`build_params`
---------------------------

Common keys in :code:`build_params`:

- :code:`cutoff_sr`, :code:`cutoff_lr`, :code:`cutoff_fn`
- :code:`dim_embedding`, :code:`num_rbf`, :code:`max_Za`
- :code:`Hartree_in_E`, :code:`Bohr_in_R`

Layers inherit these unless :code:`params` overrides them.

Monitoring energy terms
-----------------------

Optional :code:`Trainer.Monitor` lists terms such as :code:`E_ele` (electrostatic), :code:`E_disp` (D3/D4), :code:`E_zbl` for debugging layer contributions during training.

UMA and modular readouts
------------------------

With :code:`architecture: uma_qs`, the Core returns atom-level embeddings; attach :code:`SimpleReadout` / :code:`HierachicalReadout` and optional :code:`SpinConservation` in the Modelhub :code:`layers` list rather than embedding prediction heads inside the Core.

Equivariant feature contract and Equiformer readouts
----------------------------------------------------

Equivariant Cores may emit a flat irreps tensor as :code:`atom_feature` and advertise
layout via :code:`feature_irreps` (e.g. :code:`"64x0e+32x1e"`). :code:`dim_feature_out` is the
**0e channel count** used by scalar MLP readouts. :code:`SimpleReadout` extracts those
0e channels (identity when :code:`feature_irreps` is absent). For a GraphAttention
energy/charge head, swap in :code:`EquiformerGraphAttentionReadout` (see FF09 comments
in :code:`train.yaml`).

eSCN and modular readouts
-------------------------

With :code:`architecture: escn`, the native paper eSCN Core returns :code:`atom_feature`
as spherical :code:`l=0` scalars (advertised as :code:`feature_irreps: "Cx0e"`) and
:code:`atom_sphere_feature` (full SH coefficients, shape :code:`(N, (lmax+1)^2, C)` —
not e3nn-flat; reduced :code:`mmax` applies only inside edge SO(2) messages). Default
stacks use :code:`SimpleReadout` → :code:`EnergyReduce` → :code:`Force` for
energy-conserving forces, which needs differentiable edge frames / Wigner-D in
:code:`enerzyme.models.so3.rotation` (unlike fairchem v1, which detached frames for a
direct force head). Opt-in :code:`SphereSampleReadout` integrates
:code:`atom_sphere_feature` over fixed S² samples (Passaro & Zitnick energy-head
pattern) into any named atomic scalar fields (:code:`Ea`, :code:`Qa`, …); use
:code:`vector_output_fields: [Fa]` (and omit :code:`Force`) for the paper-style
direct vector path. See :code:`enerzyme/config/escn_sphere_readout_example.yaml`.
Do not confuse with :code:`uma_qs` (Meta UMA under :code:`esen/`).

So3krates and modular readouts
------------------------------

With :code:`architecture: so3krates`, the Core returns :code:`atom_feature` (invariant
stream ``x``, :code:`feature_irreps: "Fx0e"`) and :code:`atom_sphere_feature` (SPHC
``χ`` with shape :code:`[N, m_tot]`). This SPHC layout is **not** the eSCN
:code:`[N, (lmax+1)^2, C]` tensor — do not attach :code:`SphereSampleReadout`.
Default stacks use :code:`BernsteinRBF` + :code:`SimpleReadout` → :code:`EnergyReduce`
→ :code:`Force`. Optional ZBL / electrostatics / dispersion are post-core layers
(same as PhysNet / SpookyNet), not part of the Core. Example:
:code:`enerzyme/config/so3krates_layers_example.yaml`.

SO3LR (So3krates + universal pairwise FF)
-----------------------------------------

:code:`architecture: so3lr` reuses :code:`So3kratesCore` and composes SO3LR-specific
layers: :code:`ChargeSpinEmbedding` + :code:`GatherAtomEmbedding(scale_by_sqrt_count=true)`,
:code:`PartialChargeReadout` / :code:`ChargeConservation` / :code:`HirshfeldReadout`,
then :code:`ZBLRepulsionEnergy(switch_off=1.5)`, :code:`ElectrostaticEnergy(flavor=SO3LR)`,
and :code:`TSQDODispersionEnergy`. Do **not** substitute Grimme D3/D4 when targeting
SO3LR dispersion. Example: :code:`enerzyme/config/so3lr_layers_example.yaml`. For
Enerzymette, set :code:`architecture: so3lr` in the resolved :code:`config.yaml`.

NSE readout layers
------------------

:code:`NSEReadout` / :code:`HierachicalNSEReadout` and :code:`NeuralSpinChargeEquilibration` refine atomic charge and spin after the Core. Prefer :code:`output_mode: feature` on legacy Cores (PhysNet, SpookyNet, SchNet, MACE, AlphaNet) when stacking these heads. The experimental AllScAIP Core always emits :code:`atom_feature` and likewise needs an external readout (see the AllScAIP warning in :doc:`architecture_catalog`).
