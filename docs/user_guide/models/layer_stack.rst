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
    :code:`HirshfeldReadout` — SO3LR Hirshfeld-ratio head (``ha`` for TS–QDO).
    Partial charges use :code:`SimpleReadout(Qa)` + :code:`AtomicAffine` with
    fixed unit scale (element shift ≡ SO3LR ``Emb(Za)`` bias).
    Equiformer-series external readouts (including LinearRS / GraphAttention /
    :code:`EquiformerV2FeedForwardReadout`) accept :code:`shallow_ensemble_size`
    by widening the last linear layer.

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

EquiformerV2 / EquiformerV3 and modular readouts
-----------------------------------------------

With :code:`architecture: equiformer_v2`, :code:`equiformer_v3`, :code:`dpa4`,
:code:`e2former`, or :code:`e2former_v2`, the Core returns the same latent pair as eSCN
(:code:`atom_feature` as :code:`l=0` scalars with :code:`feature_irreps: "Cx0e"`, plus
:code:`atom_sphere_feature`). Default stacks use :code:`SimpleReadout` →
:code:`EnergyReduce` → :code:`Force`. Opt-in :code:`EquiformerV2FeedForwardReadout`
applies the paper sphere FFN energy head to :code:`atom_sphere_feature` after an
**EquiformerV2** Core (it looks up :code:`SO3_grid` on that Core and is **not**
wired for :code:`EquiformerV3Core` / :code:`E2FormerCore`; use :code:`SimpleReadout`
with those for now).
Shared :code:`so3` primitives provide component-normalized
grids, :code:`mmax < lmax` rotate-back rescale, and EquiformerV3 additions
(merged LN, SwiGLU-S², :code:`PolynomialEnvelope` / :code:`GraphSoftmax`).
E2Former additionally ports Wigner-6j tensor products under
:code:`enerzyme/models/e2former/` and reuses EquiformerV2's S² FFN inside its
transformer blocks. E2Former-V2 keeps the same package and Core class, switching
defaults to SO2/EAAS attention (:code:`attn_type: so2-first-order`) with optional
Triton sparse kernels (:code:`tp_type: QK_alpha+triton`).
All external Equiformer / EquiformerV2 readouts
(:code:`SimpleReadout` including :code:`equiformer_linear_rs`,
:code:`EquiformerGraphAttentionReadout`, :code:`EquiformerV2FeedForwardReadout`)
accept :code:`shallow_ensemble_size` on the last linear head; pair with
:code:`ShallowEnsembleReduce`. Examples:
:code:`enerzyme/config/equiformer_v2_layers_example.yaml`,
:code:`equiformer_v2_ffn_readout_example.yaml`,
:code:`equiformer_v2_shallow_ensemble_example.yaml`,
:code:`equiformer_v3_layers_example.yaml`,
:code:`dpa4_layers_example.yaml`,
:code:`e2former_layers_example.yaml`,
:code:`e2former_v2_layers_example.yaml`,
:code:`equiformer_shallow_ensemble_example.yaml`.

So3krates and modular readouts
------------------------------

With :code:`architecture: so3krates`, the Core returns :code:`atom_feature` (invariant
stream ``x``, :code:`feature_irreps: "Fx0e"`) and :code:`atom_sphere_feature` (SPHC
``χ`` with shape :code:`[N, m_tot]`). This SPHC layout is **not** the eSCN /
EquiformerV2 :code:`[N, (lmax+1)^2, C]` tensor — do not attach :code:`SphereSampleReadout`.
Default stacks use :code:`BernsteinRBF` + :code:`SimpleReadout` → :code:`EnergyReduce`
→ :code:`Force`. Optional ZBL / electrostatics / dispersion are post-core layers
(same as PhysNet / SpookyNet), not part of the Core. Example:
:code:`enerzyme/config/so3krates_layers_example.yaml`.

SO3LR (So3krates + universal pairwise FF)
-----------------------------------------

:code:`architecture: so3lr` reuses :code:`So3kratesCore` and composes SO3LR-specific
layers: :code:`ChargeSpinEmbedding` + :code:`GatherAtomEmbedding(scale_by_sqrt_count=true)`,
:code:`SimpleReadout(Qa)` / :code:`AtomicAffine` / :code:`ChargeConservation` / :code:`HirshfeldReadout`,
then :code:`ZBLRepulsionEnergy(switch_off=1.5)`, :code:`ElectrostaticEnergy(flavor=SO3LR)`,
and :code:`TSQDODispersionEnergy`. Do **not** substitute Grimme D3/D4 when targeting
SO3LR dispersion. Example: :code:`enerzyme/config/so3lr_layers_example.yaml`. For
Enerzymette, set :code:`architecture: so3lr` in the resolved :code:`config.yaml`.

Euclidean Fast Attention (EFA)
------------------------------

EFA is a **Core-internal** nonlocal block (not a YAML physics layer). Shared
implementation: :code:`enerzyme/models/efa/` (:code:`EFABlock`,
:code:`apply_efa_if_configured`).

* :code:`architecture: efa` — So3krates stack with :code:`era_use_in_iterations`
  set (needs :code:`Ra` + :code:`batch_seg` in the Core). Example:
  :code:`enerzyme/config/efa_layers_example.yaml`.
* :code:`architecture: so3lr_efa` — SO3LR post-core physics with EFA on the Core.
  Example: :code:`enerzyme/config/so3lr_efa_layers_example.yaml`.
* SpookyNet — Core param :code:`use_efa: true` replaces Performer nonlocal;
  default :code:`false`.
* New Cores — declare :code:`Ra` / :code:`batch_seg` inputs, construct
  :code:`EFABlock`, add its delta to invariant :code:`atom_feature` on selected
  layers. Feed only ``[N, F]`` scalars (not eSCN/EquiformerV2 sphere layouts).

NSE readout layers
------------------

:code:`NSEReadout` / :code:`HierachicalNSEReadout` and :code:`NeuralSpinChargeEquilibration` refine atomic charge and spin after the Core. Prefer :code:`output_mode: feature` on legacy Cores (PhysNet, SpookyNet, SchNet, MACE, AlphaNet) when stacking these heads. The experimental AllScAIP Core always emits :code:`atom_feature` and likewise needs an external readout (see the AllScAIP warning in :doc:`architecture_catalog`).
