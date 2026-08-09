Developer Guide
===============

This guide is for contributors who modify Enerzyme core: adding models, extending tasks, changing data pipelines, or maintaining docs and tests. It does **not** repeat end-user tutorials in :doc:`/getting_started` or configuration reference in :doc:`/user_guide`. For class-level API details, see :doc:`/api`.

Development setup
-----------------

Editable install
^^^^^^^^^^^^^^^^

From the repository root:

.. code-block:: bash

    pip install -e .

This registers the :code:`enerzyme` console script via :code:`setup.py` and exposes the package for local development.

Environment files
^^^^^^^^^^^^^^^^^

Three dependency contexts matter:

- **Runtime** — :code:`setup.py` :code:`install_requires` (NumPy, PyTorch, ASE, RDKit, Lightning, etc.)
- **Development** — :code:`requirements.yaml` at the repo root (conda env for day-to-day coding; see :doc:`/getting_started/installation`)
- **Documentation** — :code:`docs/requirements.yaml` (Sphinx, pydata theme, editable install for autodoc)

For a first-time contributor setup, follow :doc:`/getting_started/installation`, then install in editable mode as above.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

Not every contributor needs all optional stacks. Install only what your change touches:

- **PhysNet parity tests** — TensorFlow 1.x compatibility mode and the reference PhysNet package
- **NequIP / XPaiNN** — :code:`nequip`, :code:`XequiNet` and their transitive deps
- **PLUMED workflows** — :code:`py-plumed` and a PLUMED-enabled build
- **QM annotation** — TeraChem executable/license, RDKit formal charges
- **Bond assignment** — QuantumPDB-style PDB inputs and template SDFs
- **Server mode** — Flask/Waitress (already in core :code:`install_requires`)

Smoke test after setup:

.. code-block:: bash

    python -c "import enerzyme; print(enerzyme.__file__)"
    enerzyme -h
    enerzyme train -h
    enerzyme predict -h
    enerzyme simulate -h

Repository map
--------------

CLI and command wrappers
^^^^^^^^^^^^^^^^^^^^^^^^

:code:`enerzyme/cli.py` defines argparse subcommands and dispatches to thin wrapper classes:

- :code:`train` → :code:`enerzyme/train.py` (:code:`FFTrain`)
- :code:`predict` → :code:`enerzyme/predict.py` (:code:`FFPredict`)
- :code:`simulate` → :code:`enerzyme/simulate.py` (:code:`FFSimulate`)
- :code:`extract` → :code:`enerzyme/extract.py` (:code:`FFExtract`, reuses :code:`FFPredict`)
- :code:`collect` → :code:`enerzyme/collect.py` (:code:`FFCollect`)
- :code:`annotate` → :code:`enerzyme/annotate.py` (:code:`QMAnnotate`)
- :code:`bond` → :code:`enerzyme/bond/bond.py`
- :code:`listen` / :code:`request` / :code:`kill` → :code:`enerzyme/listen.py`, :code:`enerzyme/request.py`, HTTP shutdown helper in :code:`cli.py`

Core packages
^^^^^^^^^^^^^

- :code:`enerzyme/data/` — dataset loading, standard fields, transforms, HDF5 preload cache (:code:`DataHub`, :code:`FieldDataset`)
- :code:`enerzyme/models/` — architecture cores, layer stack, :code:`ModelHub`, loss registration
- :code:`enerzyme/tasks/` — training, metrics, simulation, extraction, server, splitting, active-learning picking
- :code:`enerzyme/qm/` — QM driver adapters for :code:`annotate`
- :code:`enerzyme/bond/` — PDB bond-order assignment
- :code:`enerzyme/utils/` — logging, YAML I/O (:code:`YamlHandler`)

Reference YAML configs live in :code:`enerzyme/config/`. After training, the resolved :code:`config.yaml` in the output directory is the canonical model config for :code:`predict` and :code:`simulate`.

Runtime architecture
--------------------

Enerzyme is YAML-driven. User-facing behavior is defined by config sections; Python classes implement those sections.

.. code-block:: text

    enerzyme <command> -c config.yaml -o out/
        -> YamlHandler.read_yaml()
        -> DataHub / Trainer / task-specific wrapper
        -> ModelHub (for train) or loaded checkpoints (for predict/simulate/extract)
        -> artifacts under out/

Training path (:code:`FFTrain`):

.. code-block:: text

    config.yaml
        -> DataHub(dump_dir=out/, **Datahub)
        -> Trainer(out_dir, Metric, **Trainer)
        -> ModelHub(datahub, trainer, **Modelhub)
        -> FF_single / FF_committee.train() or .active_learn()
        -> out/config.yaml, processed_dataset_<hash>/, FFxx/, logs/

Prediction and simulation load the saved model config (:code:`-mc` or default :code:`model_dir/config.yaml`), rebuild active models, and run task code without retraining.

.. note::

   Treat YAML schema and output directory layout as **public contracts**. External workflows (including `Enerzymette <https://github.com/Benzoin96485/Enerzymette>`_) depend on checkpoint names, :code:`config.yaml`, and per-iteration folder names documented in :doc:`/user_guide/workflows/active_learning`.

Configuration development rules
-------------------------------

YAML handling
^^^^^^^^^^^^^

:code:`enerzyme/utils/config_handler.py` loads YAML into :code:`addict.Dict` objects. Training writes the resolved config back to :code:`out/config.yaml` at startup.

When adding or renaming a config field:

1. Add a default or example in the relevant file under :code:`enerzyme/config/`
2. Wire the field in the consumer (wrapper class or :code:`tasks/*` module)
3. Document semantics in :doc:`/user_guide` (user-facing) or this guide (developer-facing)
4. Add a smoke test or minimal pytest case if behavior is non-trivial

Config section ownership
^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`Datahub` — consumed by :code:`DataHub`; may be partially overridden in predict/extract configs
- :code:`Modelhub` — consumed by :code:`ModelHub` (train) or model rebuild (predict/simulate/extract)
- :code:`Trainer` / :code:`Metric` — training loop, early stopping, committee, active learning
- :code:`Simulation` / :code:`System` — :code:`enerzyme/tasks/simulator.py`
- :code:`Extractor` — :code:`enerzyme/tasks/extractor.py`
- :code:`Supplier` / :code:`QMDriver` — :code:`enerzyme/annotate.py`

Backward compatibility
^^^^^^^^^^^^^^^^^^^^^^

- **Published releases** — avoid breaking existing YAML keys or artifact paths without a migration note
- **Unreleased :code:`devel` branch** — schema changes may land directly, but update reference YAML and docs in the same PR

Data pipeline extension points
------------------------------

Standard fields
^^^^^^^^^^^^^^^

Registered in :code:`enerzyme/data/datatype.py` (:code:`N`, :code:`Ra`, :code:`Za`, :code:`E`, :code:`Fa`, :code:`Qa`, :code:`Q`, :code:`M2`, etc.). See :doc:`/user_guide/concepts/units_and_fields`.

Custom fields
^^^^^^^^^^^^^

Register under :code:`Datahub.fields` with :code:`is_atomic: true/false`. Map raw dataset keys via :code:`features` and :code:`targets`.

Transforms
^^^^^^^^^^

:code:`enerzyme/data/transform.py` applies dataset-level and global transforms during preload. Common developer touchpoints:

- :code:`negative_gradient` — sign flip for QC gradients
- :code:`atomic_energy` / :code:`total_energy_normalization` — energy offsets

Preload cache
^^^^^^^^^^^^^

:code:`SingleDataHub` hashes data path, neighbor-list mode, and transform settings into :code:`processed_dataset_<hash>/` with :code:`pre_transformed.hdf5` and :code:`datahub.yaml`. Changing hash inputs invalidates the cache — document this when adding transform keys.

Checklist: new data field or transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Register type in :code:`datatype.py` if it is a new standard name
2. Ensure loader supports the source format (:code:`pickle`, :code:`npz`, :code:`aselmdb`, or HDF5 cache)
3. Add transform logic in :code:`transform.py` if preprocessing is required
4. Update :code:`train.yaml` example and :doc:`/user_guide/data/datahub_reference`
5. Run :code:`enerzyme collect -c <yaml> -o <out>` to validate mapping without training

Model development
-----------------

Layer stack
^^^^^^^^^^^

:code:`enerzyme/models/ff.py` builds models from an ordered :code:`layers` list:

- :code:`get_ff_core(architecture)` returns :code:`Core`, :code:`DEFAULT_BUILD_PARAMS`, :code:`DEFAULT_LAYER_PARAMS`
- :code:`build_model()` instantiates each layer by name from :code:`enerzyme/models/layers/`
- Physics and readout layers (:code:`ChargeConservation`, :code:`ElectrostaticEnergy`, :code:`EnergyReduce`, :code:`Force`, :code:`ShallowEnsembleReduce`, etc.) are composed around the architecture :code:`Core`

:code:`ModelHub` creates :code:`FF_single` or :code:`FF_committee` per active entry in :code:`internal_FFs` / :code:`external_FFs`.

Adding an internal architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create :code:`enerzyme/models/<name>/` with :code:`core.py`, :code:`__init__.py`, and default params
2. Register in :code:`get_ff_core()` inside :code:`ff.py`
3. Add a minimal :code:`Modelhub` block to :code:`enerzyme/config/train.yaml` (or a dedicated example YAML)
4. Support required targets (:code:`E`, :code:`Fa`, optional :code:`Qa`, :code:`M2`) and loss/metric weights
5. Add tests under :code:`test/` (layer parity or forward-pass smoke)
6. Document in :doc:`/user_guide/models/architecture_catalog`

Recent internal example: :code:`So3krates` under :code:`enerzyme/models/so3krates/`
(scalar atom embedding + RBF as pre-core layers; Core emits invariant
:code:`atom_feature` and SPHC :code:`atom_sphere_feature`; shared
:code:`SimpleReadout` / physics layers after the Core). Also:
:code:`EquiformerV2` under :code:`enerzyme/models/equiformer_v2/`
(scalar atom embedding + RBF as pre-core layers; Core emits :code:`atom_feature` /
:code:`atom_sphere_feature` with :code:`feature_irreps`; shared :code:`SimpleReadout`
or optional :code:`EquiformerV2FeedForwardReadout` / physics layers after the Core),
:code:`EquiformerV3` under :code:`enerzyme/models/equiformer_v3/`
(same latent contract; merged LN + SwiGLU-S² + envelope attention via extended :code:`so3/`),
:code:`Equiformer` (:code:`enerzyme/models/equiformer/`), and
:code:`escn` (:code:`enerzyme/models/escn/`).

Adding an external wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^

External models (NequIP, XPaiNN) live under :code:`enerzyme/models/nequip/` and :code:`enerzyme/models/xpainn/`. Requirements:

- Lazy or guarded imports so core install still works
- Add package names to :code:`autosummary_mock_imports` in :code:`docs/conf.py` if autodoc cannot import them on RTD
- Clearly state optional dependencies in User Guide and Getting Started

Checkpoint resolution
^^^^^^^^^^^^^^^^^^^^^

:code:`get_pretrain_path()` in :code:`modelhub.py` resolves :code:`best` vs :code:`last`, version suffixes, and committee member ranks. Do not rename checkpoint files without updating this logic and downstream docs.

Task and CLI development
------------------------

Adding or changing a subcommand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Parser** — add subparser and arguments in :code:`cli.py` :code:`get_parser()`
2. **Dispatch** — branch in :code:`main()` and implement :code:`<command>(args)`
3. **Wrapper** — thin class in top-level :code:`enerzyme/<task>.py` that reads YAML and calls :code:`tasks/*`
4. **Config** — example YAML in :code:`enerzyme/config/` if the command is config-driven
5. **Docs** — Getting Started tutorial (if user-facing) + User Guide reference + this guide if extension points change

Task modules
^^^^^^^^^^^^

- :code:`trainer.py` — training loop, resume, EMA, Lightning multi-GPU, active learning hooks
- :code:`simulator.py` — ASE tasks (:code:`sp`, :code:`opt`, :code:`scan`, :code:`md`, :code:`neb`, :code:`plumed`, :code:`plumed_scan`)
- :code:`extractor.py` — uncertainty-based fragment picking
- :code:`server.py` — HTTP prediction server used by :code:`listen`
- :code:`splitter.py` — dataset partitions including :code:`withheld` for internal AL
- :code:`picker.py` — active-learning sample selection

Plugin patches
^^^^^^^^^^^^^^

:code:`FFSimulate` loads :code:`-cp` (calculator) and :code:`-pp` (PLUMED generator) via :code:`importlib` from user-supplied :code:`.py` files.

- **Calculator patches** — expose a factory function (e.g. :code:`get_uma_calculator`) referenced by :code:`external_calculator.name`.
- **PLUMED patches** — expose a :code:`PlumedConfigGenerator` subclass; YAML sets :code:`plumed_config_generator.name` and :code:`method`. Enerzyme also accepts legacy callables, but new Enerzymette plugins follow the class-based contract.

See :doc:`/user_guide/workflows/enhanced_sampling_plumed` and :doc:`/user_guide/integrations/enerzymette`.

Stable artifact paths
^^^^^^^^^^^^^^^^^^^^^

Avoid renaming without strong reason:

- :code:`out/config.yaml`
- :code:`out/processed_dataset_<hash>/`
- :code:`out/FF<id>-<arch>[-suffix]/` — :code:`model_best.pth` / :code:`model_last.pth` (committee: :code:`model0_best.pth`, …)
- Simulation outputs such as :code:`md.traj.xyz`, :code:`plumed.traj.xyz`, :code:`neb.xyz`

Testing strategy
----------------

Test layout
^^^^^^^^^^^

Tests live in :code:`test/`:

- :code:`test_scatter.py` — :code:`torch_scatter` equivalence (CPU/GPU parametrized)
- :code:`test_spookynet.py` — SpookyNet layer and forward tests
- :code:`test_physnet.py` — PhysNet parity against reference TensorFlow implementation (heavy optional stack)
- :code:`test_equiformer.py` — Equiformer forward smoke, feature mode, SO(3) energy/force checks
- :code:`test_equiformer_readout.py` — 0e extract, :code:`two_layer` SimpleReadout, GraphAttention readout
- :code:`test_equiformer_parity_*.py` — numerical parity vs vendored upstream Equiformer
  (ops / Core latent incl. feature ``get_output`` / direct E·F / grads /
  feature-mode + GraphAttention readout; no training loop; production
  :code:`SimpleReadout` Dense MLP is not LinearRS-parity)
- :code:`test_escn_parity_*.py` — numerical parity vs vendored fairchem v1 eSCN (SO3 ops / Message+Layer blocks; injected edge frames)
- :code:`test_equiformer_v2_core.py` — EquiformerV2 shapes, SimpleReadout contract, build_model E/F, SO(3) scalar invariance
- :code:`test_equiformer_v2_parity_*.py` — numerical parity vs vendored EquiformerV2 upstream (SO2 conv / LinearV2 / norms / FFN / TransBlockV2)
- :code:`test_equiformer_v3_core.py` — EquiformerV3 shapes, SimpleReadout contract, build_model E/F, SO(3) scalar invariance, YAML smoke, force finite-difference conservation / Wigner autograd
- :code:`test_equiformer_v3_parity_*.py` — numerical parity vs vendored EquiformerV3 upstream (MergeLN / SO2Linear / envelope / FFN / TransBlockV3); SwiGLU-S² ``_mem`` vs eager consistency
- :code:`test_e2former_core.py` — E2Former shapes, build_model E/F, SO(3) / translation checks, top-K truncation
- :code:`test_e2former_wigner6j.py` — Wigner-6j TP vs vanilla forward (orders 1–2)
- :code:`test_e2former_parity_ops.py` — E2Former Wigner-6j / SO2 TP numerical parity vs vendored UBio-MolFM fixtures
- :code:`test_e2former_v2_core.py` — E2Former-V2 SO2 attention shapes, build_model E/F, SO(3) / translation, YAML smoke
- :code:`test_e2former_so2_tp.py` — EAAS / SO2 TP shapes, rotation equivariance, Triton PyTorch fallback
- :code:`test_e2former_triton_parity.py` — QK index convention / CPU fallback guards; CUDA Triton vs PyTorch parity (skipped without GPU)
- :code:`test_e2former_lsr_core.py` — E2Former-LSR shapes, kmeans/precomputed fragments, bipartite graph batch isolation, build_model E/F, SO(3) / translation, YAML smoke
- :code:`test_dpa4_core.py` — DPA4 shapes, registration / YAML smoke, geometry autograd, SO(3) scalar invariance, build_model E/F, force finite-difference conservation
- :code:`test_dpa4_parity_ops.py` — DPA4 indexing / C³ envelope / SO2Linear / envelope-gated softmax algebraic checks (no runtime deepmd dependency)
- :code:`test_tace_core.py` — TACE registration / YAML smoke, spherical+Cartesian feature shapes, build_model E/F
- :code:`test_tace_spherical_ops.py` — CGTP path / scatter TP / CgtpACE smoke
- :code:`test_tace_cartesian_ops.py` — cartnn ICTD / harmonics / Cartesian contraction smoke
- :code:`test_tace_parity_ops.py` — TACE spherical / Cartesian numerical parity vs vendored tace v0.1.0 fixtures
- :code:`test_tece_core.py` — TECE registration / YAML smoke, feature shapes, build_model E/F, ECE/RRA flag sensitivity
- :code:`test_tece_ops.py` — WignerD / LayoutTransform / uvSO2Linear / SO2Gate / ComplexProductBasis / RRA path smoke
- :code:`test_so3_wigner_backend.py` — shared e3nn/Jd Wigner-D backend (packed orthogonality, SO3_Rotation / fused / DPA4 quaternion adapters, high-l smoke)
- :code:`test_so3_grid.py` — unified flat lat–long :code:`SO3Grid` / :code:`S2GridProjector` protocol (roundtrip, Lebedev duck-type, grid table)
- :code:`test_so3krates_core.py` — So3krates shapes, SimpleReadout contract, build_model E/F, SO(3) energy/force checks
- :code:`test_so3krates_parity_ops.py` — numerical parity vs vendored So3krates-torch (SH / L0 / FilterNet / attention / interaction)
- :code:`test_so3lr.py` — SO3LR priors (ZBL / erf-Coulomb / TS–QDO), readouts, and :code:`architecture: so3lr` build_model smoke test
- :code:`test_efa.py` — ERoPE / Lebedev / EFABlock, batch isolation, :code:`efa` / :code:`so3lr_efa` / SpookyNet :code:`use_efa` smoke tests
- :code:`test_sphere_sample_readout.py` — SphereSampleReadout shapes / scalar invariance smoke
- :code:`test_scatter_speed.py` — performance-oriented scatter checks

Suggested commands
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m pytest test/test_scatter.py -q
    python -m pytest test/test_spookynet.py -q

PhysNet parity tests require TensorFlow and the external PhysNet reference code — skip unless you are modifying PhysNet layers.

What to add for a PR
^^^^^^^^^^^^^^^^^^^^

- **Bug fix** — regression test when feasible
- **New layer or architecture** — at least forward-pass or numerical parity test
- **New CLI flag** — argparse help text + docs; optional integration smoke script
- **Config schema change** — update reference YAML and run :code:`enerzyme collect` or a minimal train job locally

Documentation workflow
----------------------

Local Sphinx build
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda env create -f docs/requirements.yaml   # once
    conda activate docs_Enerzyme                 # env name from file
    sphinx-build -b html docs docs/_build/html

Open :code:`docs/_build/html/index.html` and confirm the Developer Guide card and toctree link work.

Sphinx configuration
^^^^^^^^^^^^^^^^^^^^

- :code:`docs/conf.py` — extensions, autosummary, mock imports for optional deps
- :code:`.readthedocs.yaml` — RTD build uses :code:`docs/requirements.yaml`

When user-visible behavior changes, update in the same change set:

- :doc:`/getting_started` — tutorial path and copy-paste commands
- :doc:`/user_guide` — schema, tuning, troubleshooting
- :doc:`/api` — only if public modules/classes change (autosummary regenerates on build)

RST style notes
^^^^^^^^^^^^^^^

- Prefer bullet lists over wide grid tables (malformed tables break :code:`sphinx-build`)
- Section underlines must be at least as long as the title text
- Cross-link with :code:`:doc:\`/path\`` rather than hard-coded HTML paths

Contribution checklist
----------------------

New YAML field
^^^^^^^^^^^^^^

- [ ] Default in :code:`enerzyme/config/*.yaml`
- [ ] Read in wrapper/task code
- [ ] User Guide section updated
- [ ] Smoke command or test

New internal model
^^^^^^^^^^^^^^^^^^

- [ ] :code:`get_ff_core()` registration
- [ ] Example train config
- [ ] Tests in :code:`test/`
- [ ] Architecture catalog entry

New task or simulation mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- [ ] :code:`simulator.py` (or relevant task) implementation
- [ ] Example YAML
- [ ] Getting Started + User Guide workflow pages
- [ ] Artifact names documented

New QM driver or supplier
^^^^^^^^^^^^^^^^^^^^^^^^^

- [ ] :code:`enerzyme/qm/` adapter
- [ ] :code:`annotate.yaml` example
- [ ] Dependency and status notes (ORCA/PySCF/Psi4 if partial)

Docs-only change
^^^^^^^^^^^^^^^^

- [ ] :code:`sphinx-build` passes
- [ ] No broken :code:`:doc:` references

Out of scope for V1
-------------------

This Developer Guide intentionally does **not** duplicate:

- Full API listings (:doc:`/api`)
- Per-architecture theory (:doc:`/user_guide/models/architecture_catalog`)
- Release engineering or versioning policy (not formalized in-repo yet)
- Enerzymette internal development — only the integration boundary (:doc:`/user_guide/integrations/enerzymette`)

When to split into sub-pages
----------------------------

Keep this single page while the contributor surface is still evolving. Split into :code:`docs/developer_guide/` when any section grows past ~200 lines or needs its own deep dive (for example, a dedicated “Adding a new architecture” cookbook). The entry :code:`docs/developer_guide.rst` would then become a short overview plus layered toctree, mirroring :doc:`/user_guide`.

External UMA (:code:`uma_qs`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register in :code:`get_ff_core` like other architectures. Keep fairchem imports inside :code:`enerzyme/models/esen/` so non-UMA installs do not import it until selected. Prefer shared :code:`layers/readout.py` and :code:`layers/spin.py` for Q/S heads. Package name :code:`esen/` is historical (UMA / eSCN-MD lineage); it is **not** the 2023 paper eSCN.

Native eSCN (:code:`escn`)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Paper eSCN (Passaro & Zitnick, 2023) lives under :code:`enerzyme/models/escn/` with shared SO(2)/SO(3) primitives in :code:`enerzyme/models/so3/` (including :code:`Jd.pt` in package data). The Core emits :code:`atom_feature` (l=0 scalars with :code:`feature_irreps = "Cx0e"`) and :code:`atom_sphere_feature` (SH grid layout for :code:`SphereSampleReadout`); compose :code:`SimpleReadout` / :code:`SphereSampleReadout` / :code:`EnergyReduce` / :code:`Force` outside the Core. No fairchem dependency. Offline numerical parity against vendored fairchem :code:`fairchem_core-1.10.0` blocks is in :code:`test/test_escn_parity_*.py` (ops + Message/LayerBlock; injected edge frames; not OC20 E/F). Enerzymette and other YAML-driven workflows only need :code:`architecture: escn` plus a resolved :code:`config.yaml` — checkpoint layout is unchanged.

EquiformerV2 (:code:`equiformer_v2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Liao et al. (ICLR 2024) lives under :code:`enerzyme/models/equiformer_v2/` and extends shared :code:`so3/` (rotate-inv rescale, component grids, :code:`SO3_LinearV2`). The Core emits the same :code:`atom_feature` / :code:`atom_sphere_feature` contract as eSCN; compose :code:`SimpleReadout` or :code:`EquiformerV2FeedForwardReadout` plus :code:`EnergyReduce` / :code:`Force` outside the Core. Offline parity vs vendored :code:`atomicarchitects/equiformer_v2` nets is in :code:`test/test_equiformer_v2_parity_*.py`. Enerzymette only needs :code:`architecture: equiformer_v2` plus a resolved :code:`config.yaml`.

EquiformerV3 (:code:`equiformer_v3`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Liao et al. (2026, arXiv:2604.09130) lives under :code:`enerzyme/models/equiformer_v3/` and further extends shared :code:`so3/` (merged LN, SwiGLU-S², :code:`SO2Linear`, :code:`PolynomialEnvelope` / :code:`GraphSoftmax`, fused :code:`SO3RotationFused`, flat lat–long :code:`SO3Grid`). The Core emits the same :code:`atom_feature` / :code:`atom_sphere_feature` contract as eSCN/V2; compose :code:`SimpleReadout` plus :code:`EnergyReduce` / :code:`Force` outside the Core. Offline parity vs vendored :code:`atomicarchitects/equiformer_v3` experimental nets is in :code:`test/test_equiformer_v3_parity_*.py`. Enerzymette only needs :code:`architecture: equiformer_v3` plus a resolved :code:`config.yaml`.

E2Former (:code:`e2former`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Li et al. (NeurIPS 2025 Spotlight, arXiv:2501.19216) lives under :code:`enerzyme/models/e2former/`,
adapted from `liyy2/E2Former <https://github.com/liyy2/E2Former>`_ (MIT). Register via
:code:`get_ff_core("e2former")`. The Core uses Wigner-6j factorization for equivariant
attention, reuses shared :code:`so3` RMSNorm / :code:`SO3Linear` and EquiformerV2's S²
:code:`FeedForwardNetwork`, and emits :code:`atom_feature` / :code:`atom_sphere_feature`.
Compose :code:`SimpleReadout` + :code:`EnergyReduce` / :code:`Force` outside the Core.
Requires equal channel multiplicity across degrees. Example:
:code:`e2former_layers_example.yaml`. Tests: :code:`test/test_e2former_core.py`,
:code:`test/test_e2former_wigner6j.py`, :code:`test/test_e2former_parity_ops.py`.

E2Former-V2 (:code:`e2former_v2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Huang et al. (2026, arXiv:2601.16622) **reuses** :code:`E2FormerCore` with defaults in
:code:`e2former/v2.py` (:code:`attn_type: so2-first-order`, optional Triton sparse QK).
Same latent contract and post-core stack as V1. Example:
:code:`e2former_v2_layers_example.yaml`. Tests: :code:`test/test_e2former_v2_core.py`,
:code:`test/test_e2former_so2_tp.py`, :code:`test/test_e2former_triton_parity.py`.

E2Former-LSR (:code:`e2former_lsr`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wang et al. (arXiv:2601.03774) adds atom–fragment bipartite long-range attention on top of
the short-range E2Former package (:code:`cutoff_lr`, :code:`long_layers`, late fusion).
Fragmentation defaults to online k-means; :code:`fragment_mode: precomputed` uses Datahub
:code:`cluster_ids`. Still emits :code:`atom_feature` / :code:`atom_sphere_feature`.
Example: :code:`e2former_lsr_layers_example.yaml`. Tests: :code:`test/test_e2former_lsr_core.py`.

DPA4 (:code:`dpa4`)
^^^^^^^^^^^^^^^^^^^^^

DPA4 (Li et al., 2026, arXiv:2606.02419) lives under :code:`enerzyme/models/dpa4/` as :code:`core.py` / :code:`interaction.py` / :code:`so2.py`. EMFA orchestration (:code:`SO2Convolution`, :code:`DynamicRadialDegreeMixer`, :code:`EdgeCache`) stays local. Shared pieces in :code:`enerzyme/models/so3/` include :code:`C3CutoffEnvelope`, Apache e3x Lebedev tables (:code:`lebedev_grids.npz`, also used by EFA) / :code:`S2LebedevProjector`, packed/m-major :code:`indexing`, :code:`FocusSO2Linear`, :code:`SO3FocusLinear`, :code:`SO3GatedActivation`, :code:`EquivariantDegreeRMSNorm`, :code:`BesselC3RadialBasis` / :code:`RadialMLP`, and quaternion edge frames (:code:`build_edge_quaternion`) that share the e3nn/:code:`Jd` Wigner-D backend (:code:`wigner_from_rotation_matrix`) with eSCN / EquiformerV2 / EquiformerV3. :code:`WignerDCalculator` maps :code:`R(q)` into DPA4's historical Cartesian basis (:code:`A R Aᵀ`) before that backend so SO(2) / GIE stay equivariant for any :code:`lmax` in :code:`Jd.pt`. Flat lat–long :code:`SO3Grid` and Lebedev :code:`S2LebedevProjector` share the :code:`S2GridProjector` contract (:code:`to_grid` / :code:`from_grid`); EquiformerV3 FFN uses :code:`SO3Grid`, DPA4 FFN defaults to Lebedev. The Core emits :code:`atom_feature` / :code:`atom_sphere_feature`; compose :code:`SimpleReadout`, :code:`EnergyReduce`, and :code:`Force` outside it. Tests: :code:`test/test_dpa4_core.py`, :code:`test/test_dpa4_parity_ops.py`, :code:`test/test_so3_wigner_backend.py`, :code:`test/test_so3_grid.py`. Enerzymette only needs :code:`architecture: dpa4` plus a resolved :code:`config.yaml`.

TACE (:code:`tace`)
^^^^^^^^^^^^^^^^^^^

Xu et al. (arXiv:2509.14961; Cartesian-3j arXiv:2512.16882) lives under :code:`enerzyme/models/tace/` (:code:`core.py`, :code:`interaction.py`), adapted from `xvzemin/tace <https://github.com/xvzemin/tace>`_ (MIT). Shared flat-Irreps helpers (:code:`IrrepsLinear`, :code:`generate_paths`, :code:`O3ScatterTensorProduct`, :code:`get_gated_nonlinear`, …) live in :code:`enerzyme/models/e3nn_nn/`; radial channel MLPs use :code:`blocks.radial_mlp.RadialMLP`. Register via :code:`get_ff_core("tace")`. Core param :code:`tensor_basis` selects spherical e3nn CGTP or Cartesian ICT (:code:`cartnn` vendored from tace v0.1.0 + :code:`cartesian/`). Scope includes edge embedding/update, BB element-aware residual, and density/avg scatter-norm; TECE / SO2 / RRA live under :code:`architecture: tece`; ZBL / LES / UIE stay out. Emit :code:`atom_feature` for :code:`SimpleReadout`. Examples: :code:`tace_layers_example.yaml`, :code:`tace_cartesian_layers_example.yaml`. Tests: :code:`test/test_tace_*.py`.

TECE (:code:`tece`)
^^^^^^^^^^^^^^^^^^^

Xu et al. (arXiv:2607.10664) lives under :code:`enerzyme/models/tece/` (:code:`core.py`, :code:`interaction.py`), adapted from `xvzemin/tace <https://github.com/xvzemin/tace>`_ v0.2.0 (MIT). Register via :code:`get_ff_core("tece")`. The Core seeds equivariant features from scalar embeddings, then stacks uvSO2 interactions with Edge Cluster Expansion (:code:`ComplexProductBasis`) and Radial Rotary Attention, plus node-side :code:`CgtpACE` reused from TACE. Shared SO(2) primitives (:code:`WignerD` recursive/direct, :code:`uvSO2Linear`, :code:`SO2Gate`, :code:`LayoutTransform`) live in :code:`enerzyme/models/so3/`. Requires :code:`Lmax == lmax`. Do not conflate RRA with EFA Euclidean RoPE. Emit :code:`atom_feature` for :code:`SimpleReadout`. Example: :code:`tece_layers_example.yaml`. Tests: :code:`test/test_tece_*.py`. Enerzymette: :code:`architecture: tece` + resolved :code:`config.yaml`.

So3krates (:code:`so3krates`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frank et al. (NeurIPS 2022) lives under :code:`enerzyme/models/so3krates/` with shared :code:`RealSphericalHarmonics` and :code:`L0Contraction` (:code:`cgmatrix.npz`) in :code:`enerzyme/models/so3/`. The Core emits invariant :code:`atom_feature` and SPHC :code:`atom_sphere_feature` (``[N, m_tot]``, not eSCN/EquiformerV2 channel layout). Compose :code:`SimpleReadout` + :code:`EnergyReduce` / :code:`Force` (and optional ZBL / electrostatics / dispersion) outside the Core. Offline parity vs So3krates-torch fixtures: :code:`test/test_so3krates_parity_ops.py`. Enerzymette only needs :code:`architecture: so3krates` plus a resolved :code:`config.yaml`.

SO3LR (:code:`so3lr`)
^^^^^^^^^^^^^^^^^^^^^

Kabylda et al. (JACS 2025) is registered as :code:`architecture: so3lr` but **reuses** :code:`So3kratesCore`. Defaults and layers live in :code:`enerzyme/models/so3krates/so3lr.py`. Physics uses shared modules with SO3LR options: :code:`ZBLRepulsionEnergy` (:code:`switch_off`), :code:`ElectrostaticEnergy` (:code:`flavor: SO3LR`), :code:`TSQDODispersionEnergy` under :code:`enerzyme/models/layers/dispersion/`, plus :code:`SimpleReadout(Qa)` / :code:`AtomicAffine` / :code:`HirshfeldReadout` / :code:`ChargeSpinEmbedding`. Grimme D3/D4 remain for PhysNet/SpookyNet stacks. Cutoff alias :code:`phys` → polynomial. Tests: :code:`test/test_so3lr.py`. Enerzymette: :code:`architecture: so3lr` + resolved :code:`config.yaml` (see :code:`enerzyme/config/so3lr_layers_example.yaml`).

EFA (:code:`efa` / :code:`so3lr_efa`) and SpookyNet :code:`use_efa`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frank et al. (arXiv:2412.08541) — shared package :code:`enerzyme/models/efa/` (ERoPE + Lebedev linear attention). Lebedev point tables live in :code:`enerzyme/models/so3/data/lebedev_grids.npz` (e3x Apache-2.0) and are imported from :code:`enerzyme.models.so3.lebedev` (also re-exported on the :code:`efa` package).

* Register :code:`efa` / :code:`so3lr_efa` in :code:`get_ff_core` (both use :code:`So3kratesCore` with :code:`era_use_in_iterations`).
* SpookyNet: Core / :code:`InteractionModule` flag :code:`use_efa` swaps :code:`NonlocalInteraction` for :code:`EFABlock` (pass :code:`Ra`).
* Adding EFA to a **new** Core: (1) include :code:`Ra` and :code:`batch_seg` in Core :code:`input_fields`; (2) :code:`build_efa_blocks(...)` or construct :code:`EFABlock`; (3) after a local layer, :code:`x = x + apply_efa_if_configured(x, Ra, batch_seg, block)`. Keep EFA inside the Core, not as a post-core YAML physics layer.
* Tests: :code:`test/test_efa.py`. Examples: :code:`efa_layers_example.yaml`, :code:`so3lr_efa_layers_example.yaml`.

AllScAIP (:code:`AllScAIP`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enerzyme's AllScAIP is an **experimental, modified (魔改)** attention Core under :code:`enerzyme/models/allscaip/`. Do **not** treat it as the recommended production model; document regressions and keep example FF entries inactive unless deliberately testing. The Core returns :code:`atom_feature` only — YAML stacks must include :code:`SimpleReadout` / NSE heads (see :code:`DEFAULT_LAYER_PARAMS`).

Flow matching (:code:`uma_flow_qs`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Requires optional :code:`torchdiffeq` (:code:`pip install -e ".[flow]"`) for ODE integration in :code:`enerzyme/tasks/generator_ode.py`. Keep ODE utilities in tasks/, not inside Core modules.
