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
| PaiNN          | via      | tensor | yes    | via         | Cartesian        |
|                | readout  | head   |        | readout     | scalar + vector  |
+----------------+----------+--------+--------+-------------+------------------+
| EquFlash       | via      | via    | yes    | via         | uvu tensor       |
|                | readout  | readout|        | readout     | product + gate   |
+----------------+----------+--------+--------+-------------+------------------+
| NequIP         | via      | via    | yes    | via         | E(3) tensor      |
|                | readout  | readout|        | readout     | product conv     |
+----------------+----------+--------+--------+-------------+------------------+
| FACTO          | via      | via    | yes    | via         | Atom-centered    |
|                | readout  | readout|        | readout     | tensor operator  |
+----------------+----------+--------+--------+-------------+------------------+
| GRACE          | via      | via    | yes    | via         | Recursive ACE    |
|                | readout  | readout|        | readout     | densities; 1L/2L |
+----------------+----------+--------+--------+-------------+------------------+
| DimeNet        | via      | via    | yes    | via         | Directional      |
|                | readout  | readout|        | readout     | messages + SBF;  |
|                |          |        |        |             | bilinear, not ++ |
+----------------+----------+--------+--------+-------------+------------------+
| DimeNet++      | via      | via    | yes    | via         | Hadamard         |
|                | readout  | readout|        | readout     | down-projection  |
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
| eSEN           | via      | via    | yes    | via         | Smooth SO(2)     |
|                | readout  | readout|        | readout     | energy network;  |
|                |          |        |        |             | not ``uma_qs``   |
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
| EquFlashV2     | via      | via    | yes    | via         | Channelwise CG   |
|                | readout  | readout|        | readout     | conv + bilinear  |
|                |          |        |        |             | gate; scalar out |
+----------------+----------+--------+--------+-------------+------------------+
| Nequix         | via      | via    | yes    | via         | Separable NequIP |
|                | readout  | readout|        | readout     | + RMSNorm;       |
|                |          |        |        |             | scalar final     |
|                |          |        |        |             | layer            |
+----------------+----------+--------+--------+-------------+------------------+
| EquiformerV3   | via      | via    | yes    | via         | Merged LN +      |
|                | readout  | readout|        | readout     | SwiGLU-S² +      |
|                |          |        |        |             | smooth-cutoff    |
|                |          |        |        |             | attention        |
+----------------+----------+--------+--------+-------------+------------------+
| E2Former       | via      | via    | yes    | via         | Wigner-6j        |
|                | readout  | readout|        | readout     | equivariant      |
|                |          |        |        |             | attention (TP)   |
+----------------+----------+--------+--------+-------------+------------------+
| E2Former-V2    | via      | via    | yes    | via         | SO2/EAAS +       |
|                | readout  | readout|        | readout     | sparse Triton    |
|                |          |        |        |             | attention        |
+----------------+----------+--------+--------+-------------+------------------+
| E2Former-LSR   | via      | via    | yes    | via         | Fragment         |
|                | readout  | readout|        | readout     | bipartite        |
|                |          |        |        |             | long-range MP    |
+----------------+----------+--------+--------+-------------+------------------+
| So3krates      | via      | via    | yes    | via         | Dual-stream      |
|                | readout  | readout|        | readout     | Euclidean        |
|                |          |        |        |             | transformer;     |
|                |          |        |        |             | SPHC ``χ``       |
+----------------+----------+--------+--------+-------------+------------------+
| TACE           | via      | via    | yes    | via         | ACE via CGTP /   |
|                | readout  | readout|        | readout     | Cartesian-3j     |
+----------------+----------+--------+--------+-------------+------------------+
| TECE           | via      | via    | yes    | via         | Edge Cluster     |
|                | readout  | readout|        | readout     | Expansion + RRA  |
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
| XPaiNN         | :code:`XequiNet` and dependencies        |
+----------------+------------------------------------------+

External models are declared under :code:`Modelhub.external_FFs` with the same :code:`active` / :code:`layers` pattern where supported.

**DimeNet** (:code:`architecture: dimenet`) is a native port of Gasteiger et al. (ICLR 2020, arXiv:2003.03123) under :code:`enerzyme/models/dimenet/`, following `gasteigerjo/dimenet <https://github.com/gasteigerjo/dimenet>`_ (MIT) **original DimeNet** (bilinear spherical Fourier–Bessel interaction — **not** DimeNet++). Shared pieces: :code:`DimeNetEnvelope` in :code:`enerzyme/models/cutoff.py`, :code:`BesselRBF` with :code:`flavor: dimenet` (trainable frequencies × ``u(x)/x``), and :code:`HierachicalReadout` with :code:`head_type: mlp` for the paper output MLPs. The Core owns embedding-block messages, directional MP, SBF, and RBF-gated atom scatter; it emits hierarchical :code:`atom_feature` ``(N, F, num_blocks+1)``. Default stacks use :code:`HierachicalReadout` + :code:`EnergyReduce` + :code:`Force`. Angles use the paper / DimeNet++ convention at atom ``j`` (not the official TF pretrained bug at atom ``i``). Example: :code:`enerzyme/config/dimenet_layers_example.yaml`. Tests: :code:`test/test_dimenet_ops.py`, :code:`test/test_dimenet_core.py`, :code:`test/test_dimenet_parity_ops.py` (PyG original DimeNet). Enerzymette only needs :code:`architecture: dimenet` plus a resolved :code:`config.yaml`.

**DimeNet++** (:code:`architecture: dimenet++` or :code:`dimenetpp`) is the fast variant from the same repository (arXiv:2011.14115). It reuses the DimeNet triplets, spherical basis, embedding block, residual layers, Bessel flavor, and :code:`HierachicalReadout`. The Core replaces the bilinear tensor with Hadamard products plus a down/up bottleneck (:code:`int_emb_size`, :code:`basis_emb_size`) and a bias-free projection to :code:`out_emb_size`, so :code:`atom_feature` is ``(N, out_emb_size, num_blocks+1)`` and the readout MLP runs at that width (:code:`num_hidden_layers: 3` matches the paper output block after the projection). Defaults follow ``config_pp.yaml`` (4 blocks, emb 128, out 256, int 64, basis 8, 7 spherical, 6 radial, cutoff 5). The radial filter and Swish down-projection run on every edge (so the gathered source message still sees ``d_kj``); the spherical basis is applied after the triplet gather. Example: :code:`enerzyme/config/dimenetpp_layers_example.yaml`. Tests: :code:`test/test_dimenetpp_core.py`, :code:`test/test_dimenetpp_parity_ops.py`. Enerzymette only needs :code:`architecture: dimenet++` in the resolved :code:`config.yaml`.

**PaiNN** (:code:`architecture: painn`) is a native port of Schütt, Unke, and Gastegger (ICML 2021, arXiv:2102.03150) under :code:`enerzyme/models/painn/`, adapted from `atomistic-machine-learning/schnetpack <https://github.com/atomistic-machine-learning/schnetpack>`_ 2.0.4 (MIT). The Core is the equivariant message/update stack only. It reads :code:`RandomAtomEmbedding` and :code:`BesselRBF` with :code:`flavor: painn` (``sin(nπr/c) / r``, cutoff left on the filter) and emits :code:`atom_feature` ``[N, F]`` plus channel-last Cartesian :code:`atom_vector_feature` ``[N, 3, F]``. This vector layout is separate from :code:`atom_sphere_feature`. The scalar energy head is the shared :code:`SimpleReadout` (:code:`head_type: two_layer`, :code:`pyramidal: true`, exact SiLU). Dipoles and polarizabilities use the shared Cartesian heads :code:`EquivariantDipole` and :code:`EquivariantPolarizability` in :code:`enerzyme/models/layers/tensorial.py` (gated blocks in :code:`layers/gated_equivariant.py`). Those heads are reusable by any later Core with the same latent pair. The external :code:`xpainn` wrapper is a different model (XequiNet). Examples: :code:`enerzyme/config/painn_layers_example.yaml`, :code:`painn_spectra_layers_example.yaml`. Tests: :code:`test/test_painn_core.py`, :code:`test/test_painn_parity_ops.py`. Enerzymette only needs :code:`architecture: painn` plus a resolved :code:`config.yaml`.

**EquFlash** (:code:`architecture: equflash`) is a native reimplementation of the EquFlash MLFF from `SamsungDS/GGNN <https://github.com/SamsungDS/GGNN>`_ (CC BY-NC-SA 4.0; the Enerzyme module does not vendor that source). The Core is the convolution stack only: a radial MLP weights a uvu Clebsch–Gordan product of node features with spherical harmonics, messages are summed on the receiver, and an :code:`e3nn` gate mixes the residual. It reads :code:`RandomAtomEmbedding` and :code:`BesselRBF` with :code:`flavor: equflash` (``(2/c) sin(nπr/c) / r``). The published radial window is :code:`cutoff_fn: xplor` with :code:`cuton` below :code:`cutoff_sr` (6 Å / 5.5 Å). The generated irrep schedule matches the released config: scalar input, natural-parity hidden irreps up to :code:`lmax`, scalar output. The Core emits flat :code:`atom_feature` plus :code:`feature_irreps`, so :code:`SimpleReadout` extracts the even scalars. Energy and forces use :code:`EnergyReduce` and :code:`Force`. :code:`conv_denominator` scales aggregated messages; the GGNN OMat config uses a dataset neighbor count, and the Enerzyme default is 1. Cuequivariance and FlashTP are not required. The uvu instruction list is the shared :func:`enerzyme.models.e3nn_nn.tp_out_irreps_with_instructions`, also used by NequIP. Example: :code:`enerzyme/config/equflash_layers_example.yaml`. Tests: :code:`test/test_equflash_core.py`. Enerzymette only needs :code:`architecture: equflash` plus a resolved :code:`config.yaml`.

**NequIP** (:code:`architecture: nequip`) is a native port of Batzner et al. (Nat. Commun. 2022) under :code:`enerzyme/models/nequip/`, following the convolutions in `mir-group/nequip <https://github.com/mir-group/nequip>`_ (MIT). The Core reads :code:`RandomAtomEmbedding` and :code:`BesselRBF` with :code:`flavor: nequip` (normalized sinc Bessel × polynomial envelope of order :code:`polynomial_p`, times :math:`2\pi/c^2`). It does not own the embedding table, the energy head, or pair priors. Each layer is a Clebsch–Gordan tensor product of neighbor features with real spherical harmonics, weighted by a radial MLP, plus an optional self-connection against the fixed species embedding and a gate (even scalars SiLU, odd scalars tanh). The first layer skips the self-connection unless :code:`sc_first: true`, so an isolated atom stays at zero and its reference energy belongs in :code:`AtomicAffine`. The latent is scalar :code:`atom_feature` (``feature_irreps = "{F}x0e"``) because the last convolution emits :math:`\ell=0` only. Default stacks use :code:`SimpleReadout` (:code:`head_type: dense`, :code:`use_bias: false`) + :code:`AtomicAffine` + :code:`EnergyReduce` + :code:`Force`. Set :code:`avg_num_neighbors` to the mean neighbor count inside the cutoff. Presets ``S`` / ``M`` / ``L`` / ``XL`` match the current foundation widths (natural parity); ``paper`` uses both parities and a depth-2 radial MLP of width 64. Checkpoints from the ``nequip`` package are not loaded. Example: :code:`enerzyme/config/nequip_layers_example.yaml`. Tests: :code:`test/test_nequip_core.py`. Enerzymette only needs :code:`architecture: nequip` plus a resolved :code:`config.yaml`.

**FACTO** (:code:`architecture: facto`) is an FP32 implementation of Liu et al. (arXiv:2611.03276) under :code:`enerzyme/models/facto/`. No reference code is published. The Core keeps learned state on atoms: a scalar stream and per-degree tensor streams, coupled by factorized scalar attention, tensor-to-scalar feedback, geometric tensor injection, and (for presets ``L`` and ``XL``) tensor transport. Shared geometry — the degree-9 pair window :code:`facto`, Gaussian filters, and integral real spherical harmonics from :code:`enerzyme.models.so3` — is built once per forward. The Core emits hierarchical :code:`atom_feature` ``(N, D, K)`` and :code:`isolated_atom_feature`. :code:`FACTOEnergyReadout` applies the paper head (bias-free map of the isolated-atom residual, elemental scale and shift). :code:`ZBLRepulsionEnergy` is the short-range pair term and is added by :code:`EnergyReduce`. Preset ``S`` matches the smallest published backbone (``D=384``, 5 blocks, no transport). Fixed DFT elemental references belong in the datahub atomic-energy transform; the readout reference buffer defaults to zero. The paper's CUDA aggregation kernels, BF16×3 matmul emulation, and Muon optimizer are not part of this port. Example: :code:`enerzyme/config/facto_layers_example.yaml`. Tests: :code:`test/test_facto_ops.py`, :code:`test/test_facto_core.py`. Enerzymette only needs :code:`architecture: facto` plus a resolved :code:`config.yaml`.

**GRACE** (:code:`architecture: grace`) is an original PyTorch implementation of the graph atomic cluster expansion (Bochkarev, Lysogorskiy, Drautz, Phys. Rev. X 2024; Lysogorskiy, Bochkarev, Drautz, arXiv:2508.17936) under :code:`enerzyme/models/grace/`. The reference package `grace-tensorpotential <https://github.com/ICAMS/grace-tensorpotential>`_ is Academic Software License, so it is not imported and its checkpoints are not weight-compatible. The Core reads :code:`RandomAtomEmbedding` and the short-range graph from :code:`RangeSeparation`. It builds a Chebyshev (or spherical-Bessel) radial embedding, a one-particle basis, and the published recursive products through order 4, then a species-dependent map to invariant densities ``atom_feature`` ``[N, n_densities]`` (``feature_irreps = "{n}x0e"``). A second layer passes an equivariant indicator to the neighbors. :code:`GRACEEnergyReadout` is the foundation head (first density plus an MLP of the rest) or the Finnis–Sinclair powers when ``preset: FS``. :code:`SimpleReadout` also accepts the densities. ZBL and other pair terms stay outside the Core. Presets ``1L``, ``1L-M``, ``1L-L``, ``2L``, ``2L-M``, and ``2L-L`` follow Supplementary Table S1 (6 Å, embedding 128). ``1L-latest`` / ``2L-latest`` follow the current public preset defaults. ``FS`` recommends embedding 64 and a 7 Å cutoff. The powered cosine window is :func:`enerzyme.models.cutoff.powered_cosine_window`. Example: :code:`enerzyme/config/grace_layers_example.yaml`. Tests: :code:`test/test_grace_ops.py`, :code:`test/test_grace_core.py`. Enerzymette only needs :code:`architecture: grace` plus a resolved :code:`config.yaml`.

**eSCN** (:code:`architecture: escn`) is a native port of Passaro & Zitnick (2023) SO(3)→SO(2) convolutions under :code:`enerzyme/models/escn/`, backed by shared primitives in :code:`enerzyme/models/so3/`. The Core emits scalar :code:`atom_feature` (spherical :code:`l=0`, with :code:`feature_irreps: "Cx0e"` and :code:`dim_feature_out = C`) and :code:`atom_sphere_feature` (full :code:`(lmax+1)^2` SH coefficients after message :code:`rotate_inv`; :code:`mmax` only reduces edge-frame SO(2)). Default stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` (energy-conserving :code:`Fa=-∇E`, so edge frames stay in the autograd graph). Opt-in :code:`SphereSampleReadout` applies the paper's S² sampling head to any atomic property fields in :code:`output_fields`; :code:`vector_output_fields: [Fa]` is the paper-style direct-force alternative. Examples: :code:`enerzyme/config/escn_layers_example.yaml`, :code:`escn_sphere_readout_example.yaml`. No fairchem dependency. Ops/block numerical parity vs vendored fairchem v1 lives in :code:`test/test_escn_parity_*.py` (forward only; force conservation is covered by unit tests).

**eSEN** (:code:`architecture: esen`) is a native port of Fu et al. (2025, arXiv:2502.12147) under :code:`enerzyme/models/esen/native/`. It reuses the shared SO(2) convolution (:code:`SO2_Convolution`), SiLU :code:`GateActivation`, :code:`SO3Linear`, equivariant norms, and :class:`RadialFunction`. The radial basis stays outside the Core: :code:`GaussianRBF` with :code:`flavor: esen` (fairchem width scalar 2; MPTrj / 30M) or :code:`BesselRBF` with :code:`flavor: esen` (enveloped spherical Bessel; SPICE). The Core owns source/target edge embeddings, the polynomial envelope on messages (order 5), two SO(2) convolutions, and a spectral node FFN. It does not sample an S² grid unless :code:`mlp_type: grid` or :code:`act_type: s2`. Blocks still run when a system has no neighbors, so a message that the envelope has driven to zero matches the graph with that neighbor removed. Default stacks use :code:`SimpleReadout` (:code:`head_type: mlp`, two SiLU hidden layers) + :code:`EnergyReduce` + :code:`Force`. Do not set a maximum neighbor count. Published widths are :code:`PUBLISHED_MP_3M_CORE_PARAMS`, :code:`PUBLISHED_SPICE_6M_CORE_PARAMS`, and :code:`PUBLISHED_30M_CORE_PARAMS`. The checked-in stack is smaller. Example: :code:`enerzyme/config/esen_layers_example.yaml`. Tests: :code:`test/test_esen_core.py`. Enerzymette only needs :code:`architecture: esen` plus a resolved :code:`config.yaml`. This is not :code:`uma_qs`.

**EquiformerV2** (:code:`architecture: equiformer_v2`) is a native port of Liao et al. (ICLR 2024) under :code:`enerzyme/models/equiformer_v2/`. It reuses shared :code:`enerzyme/models/so3/` (with EquiformerV2 ``mmax`` rescale / component grids / :code:`SO3_LinearV2`) and implements SO(2) equivariant graph attention + S²/gate feed-forward blocks. The Core emits the same latent contract as eSCN (:code:`atom_feature` / :code:`atom_sphere_feature`). Default production stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. Opt-in :code:`EquiformerV2FeedForwardReadout` wraps the paper energy FFN on :code:`atom_sphere_feature`. All external EquiformerV2 readouts accept :code:`shallow_ensemble_size` (widen last linear → :code:`ShallowEnsembleReduce`). Distinct from Equiformer V1 (e3nn TP attention) and from paper eSCN (message SO(2) without transformer attention). Examples: :code:`enerzyme/config/equiformer_v2_layers_example.yaml`, :code:`equiformer_v2_ffn_readout_example.yaml`, :code:`equiformer_v2_shallow_ensemble_example.yaml`. Parity vs vendored upstream nets: :code:`test/test_equiformer_v2_parity_*.py`.

**EquFlashV2** (:code:`architecture: equflashv2` or :code:`equflash_v2`) is an Enerzyme reimplementation of Samsung Electronics' EquFlashV2 (`SamsungDS/GGNN <https://github.com/SamsungDS/GGNN>`_, CC BY-NC-SA 4.0) in :code:`enerzyme/models/equflash/core_v2.py`. It shares :code:`irreps.py`, the equflash Bessel flavor, and the XPLOR window with EquFlash. The reference code's cuequivariance / FlashTP kernels are not required. The Core is the channelwise Clebsch-Gordan interaction and feed-forward stack. It reads :code:`RandomAtomEmbedding` and :code:`BesselRBF` with :code:`flavor: equflash` (``(2/c) sin(nπr/c) / r``) windowed by :code:`cutoff_fn: xplor` (:code:`cuton` is the reference ``cutoff_on``). It emits flat :code:`atom_feature` with :code:`feature_irreps` (the published schedule ends at even scalars, so :code:`dim_feature_out` is the channel count). Default stacks use :code:`SimpleReadout` (``two_layer``, ``pyramidal``, ``activate_hidden: false``, ``use_bias: false``), :code:`AtomicAffine` with ``order: scale_then_shift``, :code:`EnergyReduce`, and :code:`Force`. ``conv_type: full`` and ``efficient`` share that e3nn convolution. Example: :code:`enerzyme/config/equflashv2_layers_example.yaml`. Tests: :code:`test/test_equflash_v2_core.py`. Enerzymette only needs :code:`architecture: equflashv2` plus a resolved :code:`config.yaml`. Periodic stress from the reference edge autograd is not a separate layer; energy-conserving forces use :code:`Force`.

**Nequix** (:code:`architecture: nequix`) is a native port of Koker and Smidt (arXiv:2508.16067) under :code:`enerzyme/models/nequix/`, following `atomicarchitects/nequix <https://github.com/atomicarchitects/nequix>`_ (MIT). The Core is the separable ``uvu`` convolution stack only: species-independent linear residual, eSEN :code:`RMSLayerNorm` (centering off), and a SiLU/tanh gate. The last block drops non-scalars. It reads :code:`OneHotAtomEmbedding` and :code:`BesselRBF` with :code:`flavor: equflash` windowed by :code:`cutoff_fn: nequip` (polynomial envelope, ``p = 6``). It emits flat even-scalar :code:`atom_feature`. Default stacks use a bias-free :code:`SimpleReadout` (``head_type: dense``), :code:`AtomicAffine` with ``order: scale_then_shift``, :code:`EnergyReduce`, and :code:`Force`. The checked-in widths are a small stack. The default neighbor normalization is the MPtrj mean; published widths and the dataset scale/shift are :code:`PUBLISHED_MP_CORE_PARAMS`, :code:`PUBLISHED_MP_SCALE`, and :code:`PUBLISHED_MP_SHIFT`. :code:`MPTRJ_ATOMIC_NUMBERS` is the 89-element vocabulary (Z = 1..94 except 84..88). Example: :code:`enerzyme/config/nequix_layers_example.yaml`. Tests: :code:`test/test_nequix_core.py`. Enerzymette only needs :code:`architecture: nequix` plus a resolved :code:`config.yaml`. This is separate from the external :code:`nequip` wrapper, which still calls the NequIP package. OpenEquivariance kernels and JAX checkpoints are not loaded here.

**EquiformerV3** (:code:`architecture: equiformer_v3`) is a native port of Liao et al. (2026, arXiv:2604.09130) under :code:`enerzyme/models/equiformer_v3/`. It extends shared :code:`so3/` with merged layer norm, SwiGLU-S² activations, fused SO(2) linears (:code:`SO2Linear`), polynomial attention envelopes, and asymmetric S² grids. The Core keeps the same :code:`atom_feature` / :code:`atom_sphere_feature` contract as eSCN/V2. Default stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` (DeNS / stress / direct force heads stay outside the Core). Example: :code:`enerzyme/config/equiformer_v3_layers_example.yaml`. Parity: :code:`test/test_equiformer_v3_parity_*.py`. Enerzymette only needs :code:`architecture: equiformer_v3` plus a resolved :code:`config.yaml`.

**E2Former** (:code:`architecture: e2former`) is a native port of Li et al. (NeurIPS 2025 Spotlight, arXiv:2501.19216) under :code:`enerzyme/models/e2former/`, adapted from `liyy2/E2Former <https://github.com/liyy2/E2Former>`_ (MIT). The Core uses **Wigner-6j** factorization to shift expensive equivariant tensor products from edges toward nodes, with QK attention and configurable order (:code:`attn_type: first-order` default; also :code:`zero-order` / :code:`second-order`). It reuses shared :code:`so3` RMSNorm / :code:`SO3Linear`, :code:`blocks.radial_mlp.RadialFunction`, and EquiformerV2's S² :code:`FeedForwardNetwork`. Requires equal channel multiplicity across degrees (e.g. :code:`64x0e+64x1e+64x2e`). Emits the same :code:`atom_feature` / :code:`atom_sphere_feature` latent pair as eSCN/EquiformerV2. Default stacks use :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. No fairchem dependency. Example: :code:`enerzyme/config/e2former_layers_example.yaml`. Tests: :code:`test/test_e2former_core.py`, :code:`test/test_e2former_wigner6j.py`, :code:`test/test_e2former_parity_ops.py`. Enerzymette only needs :code:`architecture: e2former` plus a resolved :code:`config.yaml`.

**E2Former-V2** (:code:`architecture: e2former_v2`) ports Huang et al. (2026, arXiv:2601.16622) from `UBio-MolFM <https://github.com/IQuestLab/UBio-MolFM>`_ (MIT). It **reuses** :code:`E2FormerCore` with defaults in :code:`e2former/v2.py`: :code:`attn_type: so2-first-order` (EAAS / SO2 first-order TP + on-the-fly Wigner frames) and :code:`tp_type: QK_alpha+triton` (optional N×K Triton sparse QK/V; PyTorch fallback when CUDA is unavailable). Same latent contract and post-core stack as V1. Example: :code:`enerzyme/config/e2former_v2_layers_example.yaml`. Tests: :code:`test/test_e2former_v2_core.py`, :code:`test/test_e2former_so2_tp.py`, :code:`test/test_e2former_parity_ops.py`. Enerzymette: :code:`architecture: e2former_v2` + resolved :code:`config.yaml`.
**E2Former-LSR** (:code:`architecture: e2former_lsr`) ports Wang et al. (arXiv:2601.03774) from the `UBio-MolFM <https://github.com/IQuestLab/UBio-MolFM>`_ tag :code:`E2Former-LSR` (distinct from E2Former-V2 on :code:`main`, and from SO3LR physics priors). It reuses the short-range E2Former package, then adds atom–fragment bipartite long-range attention (:code:`cutoff_lr≈15` Å, :code:`long_layers`), mean-pool fragment irreps, and late fusion (:code:`SO3Linear(cat(h_short, x_long))`). Fragmentation defaults to online k-means (:code:`fragment_mode: kmeans`); paper-style BRICS uses :code:`fragment_mode: precomputed` with Datahub field :code:`cluster_ids` (centers are recomputed as means of current :code:`Ra` so COM frames stay consistent). Still emits :code:`atom_feature` / :code:`atom_sphere_feature`. Example: :code:`enerzyme/config/e2former_lsr_layers_example.yaml`. Tests: :code:`test/test_e2former_lsr_core.py`. Enerzymette only needs :code:`architecture: e2former_lsr` plus a resolved :code:`config.yaml`.

**DPA4** (:code:`architecture: dpa4`) implements the EMFA SO(2) descriptor of Li et al. (2026, arXiv:2606.02419) under :code:`enerzyme/models/dpa4/` (:code:`core.py`, :code:`interaction.py`, :code:`so2.py`). Shared SO(3) helpers (C³ envelope, Apache e3x Lebedev tables / :code:`S2LebedevProjector`, packed indexing, :code:`FocusSO2Linear`, focus gated activation, degree RMSNorm, Bessel×C³ RBF, quaternion edge frames + shared e3nn/:code:`Jd` Wigner-D via :code:`A R Aᵀ` into DPA4's Cartesian basis; flat :code:`SO3Grid` / Lebedev behind :code:`S2GridProjector`) live in :code:`enerzyme/models/so3/`. The Core owns the radial cache, geometric/environment seed, m-truncated interactions, and Lebedev-grid equivariant FFN. It emits :code:`atom_feature` / :code:`atom_sphere_feature`, so the standard :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force` stack applies. Example: :code:`enerzyme/config/dpa4_layers_example.yaml`. Tests: :code:`test/test_dpa4_core.py`, :code:`test/test_dpa4_parity_ops.py`. Enerzymette only needs :code:`architecture: dpa4` plus a resolved :code:`config.yaml`.

**TACE** (:code:`architecture: tace`) is a native port of Xu et al. (arXiv:2509.14961; Cartesian-3j arXiv:2512.16882) under :code:`enerzyme/models/tace/` (:code:`core.py`, :code:`interaction.py`), adapted from `xvzemin/tace <https://github.com/xvzemin/tace>`_ (MIT). Shared flat-Irreps helpers live in :code:`enerzyme/models/e3nn_nn/`; radial MLPs use :code:`blocks.radial_mlp.RadialMLP`. The Core is the ACE message/product stack only (edge update + CgtpInteraction / Cartesian ICT + CgtpACE); embeddings, property heads, ZBL/LES/UIE stay outside. Switch backends with Core param :code:`tensor_basis: spherical|cartesian` (default spherical). TECE / SO2 / RRA live in the separate :code:`tece` architecture. Emits :code:`atom_feature` with :code:`feature_irreps` / :code:`dim_feature_out` for :code:`SimpleReadout`. Examples: :code:`enerzyme/config/tace_layers_example.yaml`, :code:`tace_cartesian_layers_example.yaml`. Tests: :code:`test/test_tace_core.py`, :code:`test/test_tace_spherical_ops.py`, :code:`test/test_tace_cartesian_ops.py`, :code:`test/test_tace_parity_ops.py`. Enerzymette only needs :code:`architecture: tace` plus a resolved :code:`config.yaml`.

**TECE** (:code:`architecture: tece`) is a native port of Xu et al. (arXiv:2607.10664) under :code:`enerzyme/models/tece/` (:code:`core.py`, :code:`interaction.py`), adapted from the same `xvzemin/tace <https://github.com/xvzemin/tace>`_ v0.2.0 SO2 path (MIT). The Core runs a tensor seed (scalar → equivariant) then uvSO2 layers with **Edge Cluster Expansion** (:code:`ComplexProductBasis`) and **Radial Rotary Attention** (complex Q·K + radial phase), followed by node-side :code:`CgtpACE` reused from TACE. Shared SO(2) primitives (:code:`WignerD`, :code:`uvSO2Linear`, :code:`SO2Gate`, :code:`LayoutTransform`) live in :code:`enerzyme/models/so3/`. Requires :code:`Lmax == lmax`. Distinct from EFA's Euclidean RoPE. Emits :code:`atom_feature` for :code:`SimpleReadout`. Example: :code:`enerzyme/config/tece_layers_example.yaml`. Tests: :code:`test/test_tece_core.py`, :code:`test/test_tece_ops.py`. Enerzymette only needs :code:`architecture: tece` plus a resolved :code:`config.yaml`.

**So3krates** (:code:`architecture: so3krates`) is a native port of Frank et al. (NeurIPS 2022) under :code:`enerzyme/models/so3krates/`, following the So3krates-torch EuclideanTransformer (fused FeatureBlock + GeometricBlock + InteractionBlock). Shared :code:`RealSphericalHarmonics` and :code:`L0Contraction` live in :code:`enerzyme/models/so3/`. The Core emits :code:`atom_feature` (invariant stream ``x``, :code:`feature_irreps: "Fx0e"`) and :code:`atom_sphere_feature` (SPHC ``χ`` with shape ``[N, m_tot]``, **not** eSCN/EquiformerV2's ``[N, (lmax+1)^2, C]`` — :code:`SphereSampleReadout` does not apply). Default stacks use :code:`BernsteinRBF` + :code:`SimpleReadout` + :code:`EnergyReduce` + :code:`Force`. Long-range physics (ZBL / electrostatics / dispersion) belong in post-core layers, not inside the Core. Example: :code:`enerzyme/config/so3krates_layers_example.yaml`. Parity: :code:`test/test_so3krates_parity_ops.py`.

**SO3LR** (:code:`architecture: so3lr`) is a So3krates **variant** (Kabylda et al., JACS 2025): the same :code:`So3kratesCore` plus charge/spin embeds and shared physics layers configured for SO3LR — :code:`ZBLRepulsionEnergy` with :code:`switch_off: 1.5`, :code:`ElectrostaticEnergy` with :code:`flavor: SO3LR` (``erf(r/σ)/r``, pretrained ``σ=4``), and :code:`TSQDODispersionEnergy` (Hirshfeld-scaled TS + vdW-QDO; **not** Grimme D3/D4). Post-core heads: :code:`SimpleReadout(Qa)` + :code:`AtomicAffine` (unit scale; element shift), :code:`HirshfeldReadout` (``ha``). Defaults match the public pretrained hyperparams (``r_max=4.5``, ``L≤4``, ``H=128``, ``T=3``, phys cutoff). Example: :code:`enerzyme/config/so3lr_layers_example.yaml`. Tests: :code:`test/test_so3lr.py`. Enerzymette only needs :code:`architecture: so3lr` plus a resolved :code:`config.yaml`.

**Euclidean Fast Attention (EFA)** is a linear-scaling geometry-aware nonlocal plug-in (Frank et al., arXiv:2412.08541) implemented under :code:`enerzyme/models/efa/` (PyTorch; no JAX). Three wiring modes:

1. **SpookyNet** — set Core :code:`use_efa: true` to replace geometry-free :code:`NonlocalInteraction` with :code:`EFABlock` (needs absolute :code:`Ra`). Default remains :code:`false` for checkpoint compatibility.
2. **So3krates lineage** — :code:`architecture: efa` (So3krates + EFA) and :code:`architecture: so3lr_efa` (SO3LR stack + EFA). Both reuse :code:`So3kratesCore` with :code:`era_use_in_iterations` (e.g. ``[0, 1]``). EFA updates the **invariant** stream only; SPHC stays local.
3. **Other / future Cores** — call :code:`EFABlock` or :code:`apply_efa_if_configured` on invariant ``[N, F]`` features with :code:`Ra` and :code:`batch_seg` (see developer guide). Do not feed eSCN/EquiformerV2 ``atom_sphere_feature`` layouts into L=0 EFA.

Examples: :code:`enerzyme/config/efa_layers_example.yaml`, :code:`so3lr_efa_layers_example.yaml`. Tests: :code:`test/test_efa.py`.

**UMA** (:code:`architecture: uma_qs`) requires the :code:`fairchem` package. The Core wraps Meta's UMA / eSCN-MD backbone as an atom descriptor under :code:`enerzyme/models/esen/` (the package also holds native eSEN; this wrapper is **not** :code:`architecture: esen` and **not** the 2023 paper eSCN). Shared layers such as :code:`SimpleReadout`, :code:`HierachicalReadout`, and :code:`SpinConservation` predict atomic or molecular charge/spin outside the Core. Pair with :code:`aselmdb` datasets that provide :code:`Q` / :code:`S` (and optionally :code:`Qa` / :code:`Sa`).

.. warning::

   **AllScAIP** (:code:`architecture: AllScAIP`) in Enerzyme is an **experimental, heavily adapted (魔改)** port of fairchem-style attention IP, not a drop-in of the upstream model.
   It is **not recommended as a primary production architecture** yet. Prefer PhysNet, SpookyNet, MACE, or UMA for real campaigns; keep AllScAIP for research / ablation only (:code:`train.yaml` FF08 stays :code:`active: false` by default). The Core emits :code:`atom_feature` only — always attach :code:`SimpleReadout` (or NSE heads) before energy/charge physics layers.

Selection guidelines
--------------------

**Baseline / tutorial**
    SchNet — minimal dependencies, charge-aware stacks available.
    PaiNN — Cartesian scalar and vector features; energy from :code:`SimpleReadout`, dipole and polarizability from the shared tensor heads.
    DimeNet — directional messages with Bessel RBF; invariant node states.
    DimeNet++ — same stack with the Hadamard bottleneck when the bilinear tensor is expensive.

**Production QM-labeled clusters with charge and solvent**
    PhysNet or SpookyNet — long-range electrostatics, optional dispersion layers.

**Maximum accuracy on diverse geometries**
    MACE, NequIP, Nequix, Equiformer, EquiformerV2, EquiformerV3, E2Former, E2Former-V2, E2Former-LSR, So3krates, PaiNN, EquFlash, EquFlashV2, FACTO, or GRACE — equivariant message passing; tune cutoff and depth.
    NequIP is the E(3) tensor-product convolution (cutoff 5 Å, 4 layers, :math:`\ell_{\max}=1`, 8 Bessel functions in the default stack). Chemical embedding and the energy head stay outside the Core.
    Nequix is the simplified stack from Koker and Smidt: one-hot species, polynomial cutoff ``p = 6``, and RMSNorm. It is a separate Core from :code:`nequip`.
    FACTO keeps learned scalar and tensor state on the atoms (presets ``S`` through ``XL``).
    GRACE is the graph atomic cluster expansion (presets ``1L`` / ``2L`` and the Finnis–Sinclair student). Chemical embedding and the energy head stay outside the Core.
    PaiNN keeps equivariance in Cartesian vectors (cutoff 5 Å, 3 interactions, 20 Bessel functions in the paper) and is the path for dipole and polarizability spectra.
    EquFlash keeps equivariance in an e3nn irrep stack (cutoff 6 Å, XPLOR from 5.5 Å, 10 convolutions, lmax 3, 8 Bessel functions in the released config).
    EquFlashV2 is the channelwise Clebsch–Gordan stack on that same Bessel / XPLOR window.
    Equiformer uses SO(3) graph attention (default MD17-style stack with ``ExpNormalSmearing``
    and :code:`output_mode: feature` emitting full irreps plus :code:`feature_irreps`;
    production default is :code:`SimpleReadout` with :code:`head_type: two_layer` after 0e
    extract, optional :code:`EquiformerGraphAttentionReadout`); prefer smaller irreps /
    fewer layers for enzyme-scale clusters. EquiformerV2 uses SO(2)-reduced attention
    (default :code:`GaussianRBF` with SchNet width / no RBF envelope + :code:`atom_feature` / :code:`atom_sphere_feature`)
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
    E2Former Wigner-6j / SO2 TP by :code:`test/test_e2former_parity_ops.py`;
    TACE spherical / Cartesian by :code:`test/test_tace_parity_ops.py`;
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
