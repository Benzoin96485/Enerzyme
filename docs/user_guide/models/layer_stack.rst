Layer Stack
===========

Enerzyme builds models as ordered **layers** rather than monolithic classes. The :code:`architecture` name selects a :code:`Core`; pre- and post-core layers handle geometry, embeddings, physics, and output reductions.

Registered layers
-----------------

From :code:`enerzyme/models/layers/`:

**Geometry**
    :code:`DistanceLayer`, :code:`RangeSeparationLayer`, :code:`RadiusGraphLayer`

**Cutoff / envelopes**
    Transition functions and :code:`PolynomialEnvelope` / :code:`DimeNetEnvelope` /
    :code:`C3CutoffEnvelope` in :code:`enerzyme/models/cutoff.py` (also re-exported
    from :code:`enerzyme.models.so3` for equivariant cores).

**Radial basis**
    :code:`GaussianRBFLayer` (``flavor``: PhysNet / SchNet / esen / alignn / spinconv / gemnet-oc; optional ``apply_cutoff_fn``; ``basis_width_scalar`` for spinconv; ``envelope_exponent`` for gemnet-oc), :code:`ExponentialGaussianRBFLayer`, :code:`ExponentialBernsteinRBFLayer`, :code:`BesselRBFLayer` (``flavor``: default / dimenet / painn / equflash / nequip / esen / gemnet), :code:`SphericalBesselRBFLayer` (M3GNet ``j_l`` stack, width ``max_l * num_rbf``, no envelope), :code:`BernsteinRBFLayer`, :code:`SincRBFLayer`. Cutoff key :code:`nequip` is the polynomial envelope (``p = 6``) that Nequix and EQNorm multiply onto the equflash Bessel basis. Bessel flavor :code:`nequip` is the separate mir-group basis, with that envelope already inside the layer.

**Embeddings**
    :code:`RandomAtomEmbedding`, :code:`OneHotAtomEmbedding` (Nequix and EQNorm species vocabulary), :code:`NuclearEmbedding`, :code:`ElectronicEmbedding`, :code:`ChargeSpinEmbedding` (SO3LR-style), :code:`ScalarDenseEmbedding`, :code:`GatherAtomEmbedding` (optional :code:`scale_by_sqrt_count` for SO3LR)

**Core**
    Architecture-specific message passing (:code:`Core` with :code:`architecture` in Modelhub)

**Physics / post-processing**
    :code:`AtomicAffine`, :code:`ChargeConservation`, :code:`ElectrostaticEnergy` (flavors: SpookyNet / PhysNet / SO3LR), :code:`AtomicCharge2Dipole`, :code:`GrimmeD3Energy`, :code:`GrimmeD4Energy`, :code:`TSQDODispersionEnergy` (SO3LR; not Grimme), :code:`ZBLRepulsionEnergy` (optional :code:`switch_off` for SO3LR)

**Output**
    :code:`EnergyReduce`, :code:`Force`, :code:`ShallowEnsembleReduce`

**Readouts**
    :code:`SimpleReadout` — per-atom MLP over scalar features. :code:`head_type: two_layer` accepts :code:`pyramidal: true` (hidden width ``max(n_out, F // 2)``) or :code:`dim_hidden`. With equivariant Cores
    that set :code:`feature_irreps`, it extracts even-scalar (:code:`0e`) channels first,
    then applies :code:`dense` / :code:`residual_*` / :code:`two_layer` / :code:`mlp` / :code:`gated` heads.
    :code:`head_type: gated` is the M3GNet atomic head (parallel SiLU MLP times a sigmoid gate).
    :code:`head_type: mlp` is a stack of dense hidden layers plus a linear property
    head (DimeNet output blocks; :code:`num_hidden_layers`, optional :code:`use_bias_out: false`).
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
Do not confuse with :code:`uma_qs` (Meta UMA under :code:`esen/`) or with :code:`esen` (native Fu et al. 2025).

eSEN
----

With :code:`architecture: esen`, pre-core layers are :code:`RangeSeparation` (no neighbor cap), :code:`GaussianRBF` (``flavor: esen``, ``apply_cutoff_fn: false``) or :code:`BesselRBF` (``flavor: esen``), and :code:`RandomAtomEmbedding`. The Core returns :code:`atom_feature` (``l=0``) and :code:`atom_sphere_feature`. Energy uses :code:`SimpleReadout` (``head_type: mlp``, ``num_hidden_layers: 2``, ``use_bias_out: true``, SiLU / ``swish``) → :code:`EnergyReduce` → :code:`Force`. The message envelope and the SO(2) blocks stay inside the Core. Example: :code:`enerzyme/config/esen_layers_example.yaml`. :code:`uma_qs` is unchanged and still needs fairchem.

EquiformerV2 / EquiformerV3 and modular readouts
-----------------------------------------------

With :code:`architecture: equiformer_v2`, :code:`equiformer_v3`, :code:`dpa4`,
:code:`e2former`, :code:`e2former_v2`, or :code:`e2former_lsr`, the Core returns the same latent pair as eSCN
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
:code:`e2former_lsr` extends that package with fragment bipartite long-range
attention (see :doc:`architecture_catalog`).
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
:code:`e2former_lsr_layers_example.yaml`,
:code:`equiformer_shallow_ensemble_example.yaml`.

BOTNet
------

With :code:`architecture: botnet` (also :code:`body_ordered` and :code:`body-ordered`), pre-core layers are :code:`RangeSeparation`, :code:`BesselRBF` (``flavor: mace``, ``polynomial_p: 6``), and :code:`RandomAtomEmbedding`. The Bessel layer already applies the polynomial envelope, so :code:`RangeSeparation` does not need a cutoff function. The Core returns stacked equivariant :code:`atom_feature` ``[N, irreps, T]`` when ``output_mode`` is ``layers``. Energy uses :code:`BOTNetEnergyReadout` (linear on each layer, scalar MLP on the last when there are at least two interactions) → :code:`AtomicAffine` → :code:`EnergyReduce` → :code:`Force`. Set :code:`avg_num_neighbors` to the dataset mean. ``variant`` is ``agnostic``, ``element``, or ``residual_element``. Example: :code:`enerzyme/config/botnet_layers_example.yaml`. Published CLI widths are :code:`PUBLISHED_CORE_PARAMS`.

NequIP
------

With :code:`architecture: nequip`, pre-core layers are :code:`RangeSeparation`, :code:`BesselRBF` (``flavor: nequip``, polynomial order ``polynomial_p``, default 6), and :code:`RandomAtomEmbedding`. The Bessel layer already applies the polynomial envelope, so :code:`RangeSeparation` does not need a cutoff function. The Core returns scalar :code:`atom_feature`. Energy uses :code:`SimpleReadout` (bias-free linear matches the NequIP readout) → :code:`AtomicAffine` (per-element scale and shift) → :code:`EnergyReduce` → :code:`Force`. Set Core :code:`avg_num_neighbors` to the dataset mean. Optional Core :code:`preset` is ``S``, ``M``, ``L``, ``XL``, or ``paper``. Example: :code:`enerzyme/config/nequip_layers_example.yaml`.

Allegro
-------

With :code:`architecture: allegro`, pre-core layers are :code:`RangeSeparation`, :code:`BesselRBF` (``flavor: nequip``), and :code:`PairChemicalRadial`. The pair layer embeds the center species and the neighbor species separately and multiplies them by a linear map of the Bessel basis. The Core returns per-edge :code:`edge_feature`. Energy uses :code:`EdgeReadout` (``head_type: variance_preserving``, output ``Eij``) → :code:`EdgeEnergyReduce` (center-atom sum, ``sqrt(avg_num_neighbors)`` and optional ``sqrt(2)``) → :code:`AtomicAffine` → :code:`EnergyReduce` → :code:`Force`. Set :code:`avg_num_neighbors` on :code:`build_params` so the Core and the edge sum share it. Optional Core :code:`preset` is ``tutorial``. Example: :code:`enerzyme/config/allegro_layers_example.yaml`.

EquFlashV2
----------

With :code:`architecture: equflashv2`, pre-core layers are :code:`BesselRBF` (``flavor: equflash``, prefactor ``2/cutoff``) and :code:`RandomAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: xplor` and :code:`cuton` supplies the same radial window as EquFlash. The Core returns flat :code:`atom_feature`. Energy uses :code:`SimpleReadout` (two bias-free linears when ``activate_hidden: false`` and ``use_bias: false``) → :code:`AtomicAffine` (``order: scale_then_shift``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/equflashv2_layers_example.yaml`.

Nequix
------

With :code:`architecture: nequix`, pre-core layers are :code:`BesselRBF` (``flavor: equflash``) and :code:`OneHotAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: nequip` multiplies the basis by the NequIP polynomial envelope (``p = 6``). The Core returns even-scalar :code:`atom_feature`. Energy uses :code:`SimpleReadout` (``head_type: dense``, ``use_bias: false``) → :code:`AtomicAffine` (``order: scale_then_shift``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/nequix_layers_example.yaml`. Published MPtrj widths are :code:`PUBLISHED_MP_CORE_PARAMS`.

EQNorm
------

With :code:`architecture: eqnorm`, pre-core layers are :code:`BesselRBF` (``flavor: equflash``, trainable frequencies) and :code:`OneHotAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: nequip` multiplies the basis by the polynomial envelope. That cutoff key is fixed at ``p = 6``. The species linear stays in the Core and starts both branches. The Core returns the scalar accumulator as even-scalar :code:`atom_feature`. Energy uses :code:`SimpleReadout` (``head_type: variance_preserving``, hidden width ``F / 2``, bias-free SiLU) → :code:`AtomicAffine` (``order: scale_then_shift``) → :code:`EnergyReduce` → :code:`Force`. ``normalization: constant`` divides each message by ``sqrt(avg_num_neighbors)`` (MPtrj uses 61.84). ``normalization: degree`` divides the aggregated message by the square root of the receiver degree. ``message_envelope: true`` multiplies the polynomial window onto the tensor-product output a second time. Core ``polynomial_p`` is that second window and is read only in this mode, so keep it at 6 to match ``cutoff_fn: nequip``. The checked-in stack is small. Published widths are :code:`PUBLISHED_MPTRJ_CORE_PARAMS`, :code:`PUBLISHED_MAX_CORE_PARAMS` (the second envelope), and :code:`PUBLISHED_OMAT_CORE_PARAMS` (degree normalization). Example: :code:`enerzyme/config/eqnorm_layers_example.yaml`.

PaiNN and Cartesian tensor heads
--------------------------------

With :code:`architecture: painn`, pre-core layers are :code:`BesselRBF` (``flavor: painn``, ``apply_cutoff_fn: false``) and :code:`RandomAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: cosine` supplies the filter window. The Core returns :code:`atom_feature` ``[N, F]`` and :code:`atom_vector_feature` ``[N, 3, F]``. Energy uses :code:`SimpleReadout` → :code:`EnergyReduce` → :code:`Force`. Dipole and polarizability are post-core layers :code:`EquivariantDipole` and :code:`EquivariantPolarizability`; they read the same latents and :code:`Ra`. Charge-only dipoles remain :code:`SimpleReadout(Qa)` + :code:`AtomicCharge2Dipole`. Center coordinates in the dataset when the tensor reference frame should be the center of mass. Examples: :code:`enerzyme/config/painn_layers_example.yaml`, :code:`painn_spectra_layers_example.yaml`.

With :code:`architecture: et` (aliases :code:`torchmd-et` and :code:`equivariant-transformer`), pre-core layers are :code:`ExpNormalSmearing` (``trainable: false``) and :code:`RandomAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: cosine` supplies the second window. The basis already contains a cosine window. ``vector_cutoff: true`` multiplies that window onto the attention values; ``false`` multiplies it onto the attention score, which is the 2022 paper. The Core returns the same Cartesian pair as PaiNN, so :code:`EquivariantDipole` and :code:`EquivariantPolarizability` apply. Energy uses :code:`SimpleReadout` (``head_type: two_layer``, ``pyramidal: true``) → :code:`EnergyReduce` → :code:`Force`. The MD17 widths are 6 layers, 8 heads, 128 channels, 32 radial functions, and 5 Å. QM9 is 8 layers, 256 channels, and 64 radial functions. Example: :code:`enerzyme/config/et_layers_example.yaml`.

With :code:`architecture: tensornet2`, the pre-core layers are the same :code:`ExpNormalSmearing` and :code:`RandomAtomEmbedding` used by TensorNet. The Core returns the TensorNet latents plus ``charge_channels`` ``[N, (L+1) q]``. Every channel sums to the molecular total charge ``Q``. Energy is :code:`SimpleReadout` plus :code:`WeightedCoulombEnergy` (AIMNet-damped, weighted over charge channels), then :code:`EnergyReduce` and :code:`Force`. ``layer_weights: [1, 2, 3]`` is the default for two interaction layers. Its length must divide ``(L+1) q``; a mismatch fails when the model is built. A channel whose weights are all zero is still shifted so that it sums to ``Q``. A length-``N`` ``Q`` has to repeat the same molecular total on every atom of that molecule. ``pairs: all_to_all`` ignores periodic images. A periodic reaction field uses ``pairs: directed``, ``cutoff``, and ``reaction_field: true``. AceFF-2-OMOL adds :code:`StandardZBLRepulsionEnergy` on the short-range list. Set ``Hartree_in_E: 27.211386245988`` on those pair layers when energies are in eV. Example: :code:`enerzyme/config/tensornet2_layers_example.yaml`.

With :code:`architecture: tensornet`, pre-core layers are :code:`ExpNormalSmearing` (``trainable: false``) and :code:`RandomAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: cosine` supplies the second window. The basis already contains a cosine window, so the Core's extra multiply matches TorchMD-Net. The Core inserts self pairs at ``r = 0``. It returns ``atom_feature`` ``[N, F]``, ``atom_vector_feature`` ``[N, 3, F]`` (the skew part), and ``atom_tensor_feature`` ``[N, 3, 3, F]``. Energy uses :code:`SimpleReadout` (``head_type: two_layer``, ``pyramidal: true``) → :code:`EnergyReduce` → :code:`Force`. ``Q`` is an optional molecular total charge; the default factor is ``1 + 0.1 Q``. The rMD17 widths are 2 interactions, 128 channels, 32 radial functions, and 4.5 Å. Example: :code:`enerzyme/config/tensornet_layers_example.yaml`.

REANN
-----

With :code:`architecture: reann`, the only pre-core layers are :code:`RangeSeparation` and :code:`RandomAtomEmbedding`. The Core owns the Gaussian orbitals, the squared cosine window, and the coefficient iterations. It returns invariant :code:`atom_feature` ``[N, n_orbit]``. Energy uses :code:`SimpleReadout` (``head_type: mlp``, ``hidden_dims: [128, 128]``, ``relu_like``) → :code:`EnergyReduce` → :code:`Force`. ``num_iterations: 0`` drops the coefficient updates. ``elemental: true`` with ``atomic_numbers`` gives one update network per species. Dipoles are :code:`SimpleReadout(Qa)` and :code:`AtomicCharge2Dipole`. Polarizabilities are :code:`DerivativePolarizability`; ``variant: paper`` is Eqs. 9–11 and ``variant: repo`` is the released three-head product. ``Ra`` has to require grad. Examples: :code:`enerzyme/config/reann_layers_example.yaml`, :code:`reann_response_layers_example.yaml`.

NewtonNet
---------

With :code:`architecture: newtonnet`, pre-core layers are :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``) and :code:`RandomAtomEmbedding`. The Core multiplies the polynomial envelope, so :code:`RangeSeparation` only supplies the hard cutoff. It returns :code:`atom_feature` ``[N, F]``, :code:`atom_vector_feature` ``[N, 3, F]``, and :code:`latent_force` ``[N, 3]``. ``variant: paper`` keeps a separate displacement. ``variant: repo`` is the current THGLab interaction, and ``latent_force`` is then the channel sum of the force features. Energy uses :code:`SimpleReadout` (``head_type: mlp``, ``hidden_dims: [128, 64]``) → :code:`EnergyReduce` → :code:`Force`. The polynomial order is 9. ``layer_norm`` normalizes invariant features only. The direction loss is removed when another architecture is loaded. Dipoles and polarizabilities reuse :code:`EquivariantDipole` and :code:`EquivariantPolarizability`. The optional loss ``latent_force_cosine`` compares ``latent_force`` with ``Fa``. Example: :code:`enerzyme/config/newtonnet_layers_example.yaml`.

EquFlash
--------

With :code:`architecture: equflash`, pre-core layers are :code:`BesselRBF` (``flavor: equflash``, ``(2/c) sin(nπr/c)/r``) and :code:`RandomAtomEmbedding`. :code:`RangeSeparation` with :code:`cutoff_fn: xplor` and :code:`cuton` supplies the radial window, which :code:`BesselRBF` multiplies in when ``apply_cutoff_fn: true``. The Core returns flat irreps :code:`atom_feature` and sets :code:`feature_irreps`. The default schedule ends on even scalars, so :code:`SimpleReadout` → :code:`EnergyReduce` → :code:`Force` is the energy stack. :code:`conv_denominator` on the Core rescales each aggregated message. Example: :code:`enerzyme/config/equflash_layers_example.yaml`.

FACTO
-----

With :code:`architecture: facto`, the Core owns the elemental descriptor, the radial filters, and the interaction blocks. Pre-core :code:`RangeSeparation` should use :code:`cutoff_fn: facto` (6 Å in the paper) so the shared ZBL term sees the same smooth outer window. The Core returns :code:`atom_feature` ``(N, D, K)`` and :code:`isolated_atom_feature`. The paper energy head is :code:`FACTOEnergyReadout`; :code:`SimpleReadout` still works and reads the last block. :code:`ZBLRepulsionEnergy` is summed into :code:`Ea` by :code:`EnergyReduce` because its field name is :code:`E_zbl_a`. Set Core :code:`preset` to ``S``, ``M``, ``L``, or ``XL`` for the published widths. ``L`` and ``XL`` enable tensor transport. Example: :code:`enerzyme/config/facto_layers_example.yaml`.

GRACE
-----

With :code:`architecture: cace`, pre-core layers are :code:`RangeSeparation` (no ``cutoff_fn``), :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``, trainable frequencies in the published fits), and :code:`RandomAtomEmbedding`. The Core applies the polynomial envelope of order :code:`envelope_p` and returns invariant ``atom_feature`` ``[N, F]``. :code:`CACEEnergyReadout` is a linear map plus a SiLU MLP. :code:`SimpleReadout` reads the same vector. :code:`EnergyReduce` and :code:`Force` are unchanged. Set Core :code:`preset` to ``water``, ``water-local``, ``ethanol``, ``bpa``, ``hea``, or ``custom``. ``water`` is the published liquid-water fit (embedding 3, 6 Bessel functions, 12 mixed radial channels, :math:`\ell_{\max}=3`, :math:`\nu_{\max}=3`, one ``b`` message, cutoff 5.5 Å, envelope order 6, ``avg_num_neighbors: 1``). An isolated atom has a zero descriptor; put elemental references in :code:`AtomicAffine` or the datahub. Example: :code:`enerzyme/config/cace_layers_example.yaml`.

With :code:`architecture: grace`, pre-core layers are :code:`RangeSeparation` and :code:`RandomAtomEmbedding`. The Core does not own the embedding table. It returns invariant densities :code:`atom_feature` ``[N, n_densities]``. The foundation energy head is :code:`GRACEEnergyReadout` (linear density plus an MLP, then a per-element shift). :code:`SimpleReadout` reads the same vector. :code:`EnergyReduce` and :code:`Force` are unchanged, so forces stay :code:`Fa = -∇E`. Optional :code:`ZBLRepulsionEnergy` is added by name, as for other stacks. Set Core :code:`preset` to ``1L``, ``1L-M``, ``1L-L``, ``2L``, ``2L-M``, ``2L-L``, ``1L-latest``, ``2L-latest``, ``FS``, or ``custom``. ``FS`` is the Finnis–Sinclair readout; set :code:`dim_embedding: 64` and :code:`cutoff_sr: 7`. Example: :code:`enerzyme/config/grace_layers_example.yaml`.

Geodite
-------

With :code:`architecture: geodite`, pre-core layers are :code:`RangeSeparation` (no ``cutoff_fn``), :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``), :code:`RadialWindow` (polynomial envelope of order 6 and covalent attenuation), and :code:`RandomAtomEmbedding`. The Core returns hierarchical :code:`atom_feature` ``[N, C, K]``, :code:`isolated_atom_feature`, and :code:`atom_steerable_feature` in the Geodite polynomial basis. Energy uses :code:`IsolatedAtomResidual` → bias-free :code:`HierachicalReadout` → :code:`ScreenedZBLRepulsionEnergy` → :code:`EnergyReduce` → :code:`Force`. Place the residual immediately after the Core so the readout sees :code:`dim_feature_out` and :code:`num_output_blocks`. The neighbor list stores both directions. Default build parameters are Geodite-MP in eV (192 channels, 12 Bessel functions, 6 Å, ``Hartree_in_E: 27.211386245988``). An empty Core parameter block selects 4 layers, :code:`lmax` 2, 8 heads, dropout 0.2, and edge update ``linw_mlpa_ln``. Example: :code:`enerzyme/config/geodite_layers_example.yaml`. The published second-stage Huber loss is not implemented; use MAE and set the force weight to 1 or 5.

DimeNet and modular readouts
----------------------------

With :code:`architecture: dimenet`, the Core is original DimeNet directional
message passing (bilinear SBF — not DimeNet++). Pre-core :code:`BesselRBF`
must use :code:`flavor: dimenet`. The Core emits hierarchical
:code:`atom_feature` of shape :code:`(N, F, num_blocks+1)` (embedding block plus
each interaction). Default stacks use :code:`HierachicalReadout` with
:code:`head_type: mlp` (inferred :code:`num_blocks` from Core
:code:`num_output_blocks`). :code:`SimpleReadout` still works: it uses the last
block. Charge/dipole use the same readout fields plus post-core physics layers.
Example: :code:`enerzyme/config/dimenet_layers_example.yaml`.

With :code:`architecture: dimenet++` (alias :code:`dimenetpp`), the same pre-core
Bessel flavor and post-core :code:`HierachicalReadout` are used. The Core is
:class:`~enerzyme.models.dimenet.core_pp.DimeNetPPCore`: Hadamard interaction
and a bias-free map to :code:`out_emb_size`, which becomes
:code:`dim_feature_out`. Example:
:code:`enerzyme/config/dimenetpp_layers_example.yaml`.

With :code:`architecture: gemnet` (aliases :code:`gemnet-t` and :code:`gemnett`), pre-core layers are :code:`RangeSeparation`, :code:`BesselRBF` (``flavor: gemnet``), and :code:`RandomAtomEmbedding`. The Core is GemNet-T. Leave ``basis_cutoff`` unset when ``cutoff_sr`` equals the embedding cutoff (default 5 Å). The envelope lives inside that Bessel flavor (order ``envelope_exponent``, default 5) and is not the DimeNet envelope. The Core emits hierarchical :code:`atom_feature` ``(N, F, num_blocks+1)``. Energy uses bias-free :code:`HierachicalReadout` (``head_type: dense``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/gemnet_layers_example.yaml`.

With :code:`architecture: gemnet-q` (alias :code:`gemnetq`), the same stack uses two cutoffs. Set ``cutoff_sr`` and Core ``int_cutoff`` to 10 Å, and set both ``BesselRBF.basis_cutoff`` and Core ``emb_cutoff`` to 5 Å. ``triplets_only`` is false. Quadruplet central bonds use the longer neighbor list; messages stay on the embedding edges. Example: :code:`enerzyme/config/gemnetq_layers_example.yaml`.

With :code:`architecture: gemnet-oc` (aliases :code:`gemnetoc` and :code:`gemnet_oc`), pre-core layers are :code:`RangeSeparation`, :code:`GaussianRBF` (``flavor: gemnet-oc``, ``apply_cutoff_fn: false``), and :code:`RandomAtomEmbedding`. The Gaussian is ``u_p(r/c) exp(-½ ((r/c - μ) / Δ)²)`` with centers on ``[0, 1]`` and polynomial envelope order ``envelope_exponent`` (default 5). One basis at ``cutoff_sr`` is enough: the Core calls :meth:`GaussianRBFLayer.expansion` at each shorter interaction cutoff. ``cutoff_aint`` is the longest of those cutoffs and cannot exceed ``cutoff_sr``. ``cutoff``, ``cutoff_qint``, and ``cutoff_aeaint`` cannot exceed ``cutoff_aint``. Each graph also keeps its closest ``max_neighbors``. The embedding edges are symmetrized after that cap. Quadruplet central bonds, atom–edge edges, and atom–atom edges are not. The Core emits hierarchical :code:`atom_feature` ``(N, F, num_blocks+1)``. An isolated atom keeps a species-dependent latent. Energy uses bias-free :code:`HierachicalReadout` (``head_type: dense``) → :code:`EnergyReduce` → :code:`Force`. The OC20 base widths are the defaults (cutoff 12 Å, 4 blocks, edge width 512). Example: :code:`enerzyme/config/gemnet_oc_layers_example.yaml`.

With :code:`architecture: m3gnet`, pre-core layers are :code:`RangeSeparation`, :code:`SphericalBesselRBF`, and :code:`RandomAtomEmbedding`. The Core owns the three-body update and the gated convolution. :code:`cutoff_sr` is the neighbor list (default 5 Å). :code:`threebody_cutoff` (default 4 Å) is the polynomial envelope inside the Core and cannot exceed :code:`cutoff_sr`. Energy uses :code:`SimpleReadout` (``head_type: gated``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/m3gnet_layers_example.yaml`.

With :code:`architecture: mattersim` (aliases :code:`m3gnet-graphormer` and :code:`m3gnet_graphormer`), the same stack adds :code:`GaussianRBF` (``flavor: SchNet``, ``apply_cutoff_fn: false``, its own ``num_rbf``). The Core renames that field to ``graphormer_rbf`` and runs centrality plus cutoff-masked attention around the M3GNet blocks. ``attention_cutoff`` cannot exceed ``cutoff_sr``. The checked-in stack is small (2 attention layers, 4 heads, 16 Gaussians). The paper's 24-layer Graphormer is the same Core with those widths. Example: :code:`enerzyme/config/mattersim_layers_example.yaml`.

With :code:`architecture: dpa3`, pre-core layers are :code:`RangeSeparation` (no ``cutoff_fn``) and :code:`RandomAtomEmbedding`. ``cutoff_sr`` must be at least the Core ``e_rcut`` (default 6 Å). The Core owns the polynomial or exponential gate, the edge and angle seeds, and the repflow stack. It returns invariant :code:`atom_feature`. Energy uses :code:`SimpleReadout` → :code:`EnergyReduce` → :code:`Force`. :code:`AtomicAffine` is the elemental energy bias. Published widths are :code:`PUBLISHED_CORE_PARAMS`. Example: :code:`enerzyme/config/dpa3_layers_example.yaml`.

With :code:`architecture: dpa4c`, pre-core layers are :code:`RangeSeparation` (``build`` rejects ``cutoff_fn``) and :code:`RandomAtomEmbedding`. Self-loops are rejected. ``dim_embedding`` is the descriptor channel width and must be one of 8, 16, 32, 64, 128. The Core owns the raw Bessel or Gaussian basis, the C³ envelope, the ordered-pair radial map, and the polynomial invariants. It returns one invariant :code:`atom_feature`. Energy uses :code:`SimpleReadout` (``head_type: mlp``, ``identity_residual: true``, SiLU) → :code:`EnergyReduce` → :code:`Force`. :code:`AtomicAffine` is the elemental reference energy. Published grades are :code:`PUBLISHED_GRADES`. The checked-in stack is Nano. Example: :code:`enerzyme/config/dpa4c_layers_example.yaml`.

With :code:`architecture: chgnet`, pre-core layers are :code:`RangeSeparation`, two :code:`BesselRBF` layers (``flavor: gemnet``), and :code:`RandomAtomEmbedding`. The first Bessel basis uses :code:`cutoff_sr` (default 5 Å). The second sets ``cutoff_sr`` to ``bond_cutoff`` (default 3 Å); the Core renames that field to ``rbf_bg``. The neighbor list must contain both directions of every bond. The Core owns the undirected bond index, the angle Fourier expansion, and the atom, bond, and angle updates. It emits ``atom_feature`` and ``atom_feature_charge``. :code:`AbsoluteLinearReadout` maps the charge latent to :code:`Ma`. Energy uses :code:`SimpleReadout` (``head_type: mlp``, two hidden SiLU layers, ``use_bias_out: true``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/chgnet_layers_example.yaml`.

With :code:`architecture: alignn`, pre-core layers are :code:`RangeSeparation`, :code:`GaussianRBF` (``flavor: alignn``, ``apply_cutoff_fn: false``), and :code:`RandomAtomEmbedding`. ``cutoff_sr`` is both the neighbor list and the Gaussian support (default 8 Å). The Core owns the directed line graph, the cosine Gaussian on ``[-1, 1]``, and the edge-gated updates. ``update_order: code`` is the released layer order. Energy uses :code:`SimpleReadout` (``head_type: dense``) → :code:`EnergyReduce` → :code:`Force`. Example: :code:`enerzyme/config/alignn_layers_example.yaml`.

With :code:`architecture: spinconv`, pre-core layers are :code:`RangeSeparation`, :code:`GaussianRBF` (``flavor: spinconv``, ``apply_cutoff_fn: false``, ``basis_width_scalar`` default 3), and :code:`RandomAtomEmbedding`. ``cutoff_sr`` is the neighbor list and the Gaussian support (default 6 Å). The Core owns the source-centered latitude–longitude grid, the circular spin convolution, and the species mixture. ``atom_feature`` is ``(N, mid_hidden_channels)``. Energy uses :code:`SimpleReadout` (``head_type: dense``) → :code:`EnergyReduce` → :code:`Force`. The bilinear splat weights are held fixed in the backward pass, which is how the released model evaluates gradient forces. Example: :code:`enerzyme/config/spinconv_layers_example.yaml`.

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

:code:`NSEReadout` / :code:`HierachicalNSEReadout` and :code:`NeuralSpinChargeEquilibration` refine atomic charge and spin after the Core. Prefer :code:`output_mode: feature` on legacy Cores (PhysNet, SpookyNet, SchNet, MACE) when stacking these heads. LEFTNet and AlphaNet always emit :code:`atom_feature`. The experimental AllScAIP Core always emits :code:`atom_feature` and likewise needs an external readout (see the AllScAIP warning in :doc:`architecture_catalog`).
