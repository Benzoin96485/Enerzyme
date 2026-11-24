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
- **XPaiNN** — :code:`XequiNet` and its transitive deps
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

The architecture registry in :code:`enerzyme.models.ff` resolves the preset.
:code:`enerzyme.models.pipeline` builds its ordered :code:`layers` list.

- :code:`get_ff_core(architecture)` returns :code:`Core`, :code:`DEFAULT_BUILD_PARAMS`, :code:`DEFAULT_LAYER_PARAMS`
- :code:`build_model()` instantiates each layer by name from :code:`enerzyme/models/layers/`
- Physics and readout layers (:code:`ChargeConservation`, :code:`ElectrostaticEnergy`, :code:`EnergyReduce`, :code:`Force`, :code:`ShallowEnsembleReduce`, etc.) are composed around the architecture :code:`Core`

:code:`ModelHub` creates :code:`FF_single` or :code:`FF_committee` per active entry in :code:`internal_FFs` / :code:`external_FFs`.

Adding an internal architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Define the preset factory in :code:`enerzyme/models/presets/`. Compose an
   :code:`Embed`, a sequence of :code:`Block` modules and a :code:`Head` in
   :code:`ComposableCore`. Set :code:`GraphRequires` and the input field names.
2. Set the default build parameters and YAML layers. Add the factory, defaults
   and aliases to :code:`enerzyme/models/presets/builtin.py`. Extensions can
   also use :code:`register_architecture`.
3. Set the head's output fields and the parameters that its downstream readout
   needs, such as :code:`dim_feature_out` and :code:`feature_irreps`.
4. Add a runnable configuration and tests for forward values, forces,
   symmetries and checkpoint parity. Record the preset in the
   :doc:`/user_guide/models/architecture_catalog`.

The shared :code:`edge_rotation` function defaults to the recursive Clebsch–Gordan
backend. It had the shortest median time for the forward pass and coordinate
gradient at 64 and 1,024 edges, for every tested degree from 1 to 4 in both
precisions on a Xeon Gold 6130 and a Tesla V100. The benchmark supplies the
same Cartesian frame to all backends. Existing presets select their upstream
frame convention explicitly. The measurements in :code:`test/fixtures/bench_rotations.json`
can be reproduced with :code:`scripts/bench_rotations.py`.

The paired GPU measurements in :code:`test/fixtures/bench_gpu/` cover the
three acceptance systems in both precisions. Run
:code:`python scripts/report_eager_gates.py` to summarize the archived ratios
against Phase 0 and the main checkpoints, including missing comparisons.

For ordinary :code:`torch.compile`, the paired V100 results are in
:code:`test/fixtures/bench_gpu/raw_compile_v100.json`. Its :code:`assessment`
counts comparable timings and peak allocations and identifies values over
the 3% threshold. To reproduce a pair, run the same driver in two fresh
processes on the same GPU. Set :code:`BASELINE_TREE` to a checkout of
:code:`1aa0f8b` and use the original frozen snapshot for both runs.

.. code-block:: shell

   python scripts/bench_frozen_compile.py --model-tree "$BASELINE_TREE" --snapshot test/fixtures/presnapshot/schnet.pt --precision float64 --out baseline.json
   python scripts/bench_frozen_compile.py --model-tree . --snapshot test/fixtures/presnapshot/schnet.pt --precision float64 --out candidate.json

Each process runs the molecular batch, periodic cell and 2,000-atom cluster
in order. It checks the first compiled run against its own eager energy,
forces and active weight gradients, then records the median time and peak
allocation after warmup. A model can reuse a graph captured for a previous
system. The per-system graph counters report only newly captured operations.

To check force training with the installed Inductor backend, run the following
from the checkout. The report compares the energy, forces, and every parameter
gradient for identical initial weights and coordinates. It records captured
operations, numerical errors, and changes between an absent and a zero gradient.
The command exits with a nonzero status on a mismatch. For dropout comparisons,
add :code:`--match-rng` to use matching random draws. For presets using the
optional e3nn FX kernels, pass :code:`--e3nn-codegen fx`.
Use :code:`--mode inference` to compare the reported energy and forces and
differentiate the returned energy twice with respect to the coordinates.
On PyTorch versions with :code:`donated_buffer`, the two additional gradient
calls request reusable saved tensors with
:code:`torch._functorch.config.patch(donated_buffer=False)`.

If a forward hook rescales a force or replaces its input energy, its changes
also affect the force-loss gradients. Ordinary compiled training invokes the
original :code:`ForceLayer` whenever that layer has hooks. Use ordinary
:code:`torch.compile(model)` when installing such hooks, including hooks added
after the first call.

On CPU, ordinary Inductor compiles the CACE and GRACE force calculations with
scalar C++ kernels. This preserves their coordinate and parameter derivatives
on PyTorch 2.13, where the vectorized kernels can terminate the process. The
default EquiformerV3 and CAMP training graphs and the Nequix and EquFlashV2 FX
training graphs use the same setting. The model restores the compiler settings
before returning. An explicit conflicting backend override uses native
derivatives for that call.

.. code-block:: shell

   PYTHONPATH=. python scripts/check_compiled_forces.py mattersim --dtype float64
   PYTHONPATH=. python scripts/check_compiled_forces.py chgnet --dtype float64 --prepared-energy-forces

The per-preset CPU results are in
:code:`test/fixtures/validation/compiled_forces_cpu.json`. Each entry identifies
the revision, either directly or through its enclosing run. A numerical match
without a captured learned operator is recorded as native differentiation.
The report keeps the original sweep and subsequent checks with corrected
models or matched dropout masks. An auxiliary weight with an explicit zero
gradient can receive an optimizer update even when the corresponding eager
gradient is absent. The inference results in
:code:`test/fixtures/validation/compiled_inference_cpu.json` include the energy,
forces, both coordinate derivatives, and the revision used for each check.
The second command prepares the batch's discrete neighbors before compiling
the graph that returns energy and forces for the loss. On CHGNet this leaves
the magnetic-moment head's gradient absent during energy/force fitting.

:code:`GraphContext` caches geometry for the duration of one forward pass.
Each block returns a :code:`BlockState` carrying node, edge or auxiliary
features. The head can consume the final state or request intermediate
states. The SchNet preset in :code:`enerzyme/models/presets/scalar.py`
shows a complete scalar example. Its energy and force pipeline can be built
from a checkout using the training example below.

.. code-block:: python

    from pathlib import Path
    import yaml
    from enerzyme.models.ff import build_model
    from enerzyme.models.presets import available_architectures

    path = Path("example/L3-COMT-aselmdb-smoke/config/train_otf_batch_schnet.yaml")
    entry = yaml.safe_load(path.read_text())["Modelhub"]["internal_FFs"]["FF_OTF_SchNet"]
    model = build_model(entry["architecture"], entry["layers"], entry["build_params"])
    assert "schnet" in available_architectures("scalar")

Recent internal example: :code:`DimeNet` under :code:`enerzyme/models/dimenet/`
(Bessel RBF + atom embedding as pre-core layers; Core emits hierarchical
:code:`atom_feature`; shared :code:`HierachicalReadout` / physics layers after
the Core). Also :code:`CHGNet` under :code:`enerzyme/models/chgnet/`
(two GemNet Bessel bases and the species embedding outside the Core;
undirected bonds and the angle line graph inside; :code:`Ma` from
:code:`AbsoluteLinearReadout`). Also :code:`GemNet` under :code:`enerzyme/models/gemnet/`
(GemNet Bessel flavor and atom embedding outside the Core; directional
triplets, and dihedral quadruplets for GemNet-Q, inside the Core; shared
bias-free :code:`HierachicalReadout`). GemNet-OC is the same package
(:code:`stages_oc.py`): a ``gemnet-oc`` Gaussian stays outside the Core, and
the interaction hierarchy reuses the GemNet blocks. Also :code:`So3krates` under :code:`enerzyme/models/so3krates/`
(scalar atom embedding + RBF as pre-core layers; Core emits invariant
:code:`atom_feature` and SPHC :code:`atom_sphere_feature`; shared
:code:`SimpleReadout` / physics layers after the Core). Also:
:code:`EquiformerV2` under :code:`enerzyme/models/equiformer_v2/`
(scalar atom embedding + RBF as pre-core layers; Core emits :code:`atom_feature` /
:code:`atom_sphere_feature` with :code:`feature_irreps`; shared :code:`SimpleReadout`
or optional :code:`EquiformerV2FeedForwardReadout` / physics layers after the Core),
:code:`EquiformerV3` under :code:`enerzyme/models/equiformer_v3/`
(same latent contract; merged LN + SwiGLU-S² + envelope attention via extended :code:`so3/`),
:code:`EquFlash` and :code:`EquFlashV2` under :code:`enerzyme/models/equflash/`
(shared irreps, Bessel flavor, and XPLOR window; each Core emits flat
:code:`atom_feature`),
:code:`Nequix` under :code:`enerzyme/models/nequix/`
(one-hot species + equflash Bessel / NequIP envelope outside the Core;
separable convolution emits scalar :code:`atom_feature`),
:code:`EQNorm` under :code:`enerzyme/models/eqnorm/`
(same one-hot and radial window; the Core's species linear starts a tensor
branch and a scalar accumulator),
:code:`Equiformer` (:code:`enerzyme/models/equiformer/`), and
:code:`escn` (:code:`enerzyme/models/escn/`), and
:code:`eSEN` under :code:`enerzyme/models/esen/native/`
(UMA wrappers stay in the same package and still load fairchem only when
:code:`uma_qs` / :code:`uma_flow_qs` is selected). Also :code:`Geodite`
under :code:`enerzyme/models/geodite/` (Bessel basis, covalent window, and
screened ZBL outside the Core; hierarchical scalars plus a polynomial
steerable state; shared :code:`HierachicalReadout`). Also :code:`CACE`
under :code:`enerzyme/models/cace/` (default Bessel basis and chemical
embedding outside the Core; Cartesian invariants inside; linear-plus-MLP
readout outside). Also :code:`CAMP` under :code:`enerzyme/models/camp/`
(chemical embedding outside the Core; a species-pair Chebyshev basis inside
each moment layer; per-layer linear / final MLP readout outside). Also :code:`TensorNet` and :code:`TensorNet2` under :code:`enerzyme/models/tensornet/`
(exponential-normal basis and species embedding outside the Core; Cartesian
3×3 messages inside; TensorNet2 equilibrates charge channels inside the
messages; shared :code:`SimpleReadout` and :code:`WeightedCoulombEnergy`
after the norm embedding).

Compiled energy and force losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When fitting only energy and forces, the compiled function can select those
two outputs from the usual model dictionary. This lets AOTAutograd see which
auxiliary parameters contribute to the loss. AdamW skips a parameter whose
gradient is :code:`None` and applies weight decay to a parameter with an
explicit zero gradient.

The following CPU example uses the same model, weights, two structures and
squared loss as the recorded checks. Run it from the repository checkout.
Keep the dynamic-shape and scalar capture settings active when executing the
compiled function. The coordinate tensor must require gradients.

.. code-block:: python

    import torch
    from scripts.core_harness import build, make_batch, random_molecules

    torch.manual_seed(142)
    model = build("visnet").float().train()
    batch = make_batch(random_molecules(107, [5, 4], density=0.12), torch.float32)
    batch["Ra"].requires_grad_(True)
    capture = dict(capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True)

    def energy_forces(inputs):
        outputs = model(inputs)
        return outputs["E"], outputs["Fa"]

    with torch._dynamo.config.patch(**capture):
        predict = torch.compile(energy_forces, fullgraph=True)
        energy, forces = predict(batch)
        (energy.square().sum() + forces.square().sum()).backward()

To include the loss inside the graph, compile the following function instead.
A supervised fit can pass its energy and force targets as additional inputs.

.. code-block:: python

    def force_loss(inputs):
        energy, forces = energy_forces(inputs)
        return energy.square().sum() + forces.square().sum()

    model.zero_grad(set_to_none=True)
    with torch._dynamo.config.patch(**capture):
        torch.compile(force_loss, fullgraph=True)(batch).backward()

On CPU with PyTorch 2.13.0, both wrappers preserved all 197 participating
ViSNet parameter gradients and all 223 FreeCG gradients. The unused final-layer
parameters retained :code:`grad is None` (two for ViSNet, three for FreeCG).
For the energy/force wrapper, maximum force errors against eager FP32 were
:code:`4.77e-7` and :code:`1.49e-7`, respectively. Both wrappers also preserved
ViSNet-LSRM's 182 participating gradients and ten absent gradients. Four LSRM
bias tensors had individual FP32 elements outside the recorded tolerance.
The LSRM energy/force wrapper passed all 182 gradients in FP64 with
:code:`atol=1e-10` and :code:`rtol=1e-8`. Its maximum force error was
:code:`2.06e-15`.

QuinNet preserves all 182 participating gradients and both absent gradients
with either wrapper in both precisions at revision :code:`b6796ea`.
The maximum FP64 energy and force errors were each below :code:`7e-16`.
The ordinary fullgraph call on CHGNet at revision :code:`2fa0e3d` stops when
constructing reverse edges from a data-dependent edge count. Per-model
results, tolerances and the independent FP64 controls are recorded in
:code:`test/fixtures/validation/vector_force_inductor_cpu.json`.

Adding an external wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^

The optional xPaiNN preset in :code:`enerzyme/models/presets/xpainn.py` composes XequiNet's embedding, message/update pairs, and output head. Its defaults live in :code:`enerzyme/models/xpainn/`. The two UMA wrappers load their external models directly. A wrapper should meet the following requirements.

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
- :code:`test_dpa3_ops.py` — DPA3 polynomial / exponential gates, SiLUT join, symmetrizer, angle pairs, residual combiners
- :code:`test_dpa3_core.py` — DPA3 registration / YAML smoke, rotation / inversion / translation / permutation, batch isolation, build_model E/F, force finite-difference conservation
- :code:`test_dpa4_core.py` — DPA4 shapes, registration / YAML smoke, geometry autograd, SO(3) scalar invariance, build_model E/F, force finite-difference conservation
- :code:`test_dpa4_parity_ops.py` — DPA4 indexing / C³ envelope / SO2Linear / envelope-gated softmax algebraic checks (no runtime deepmd dependency)
- :code:`test_dpa4c_ops.py` — DPA4C degree profile, Cartesian harmonics, STF isometry, raw Bessel, pair FiLM bounds, Gram isometry, residual fitting head
- :code:`test_dpa4c_core.py` — DPA4C registration / YAML smoke, rotation / inversion / translation / permutation, batch isolation, calibration, build_model E/F, force finite-difference conservation
- :code:`test_tace_core.py` — TACE registration / YAML smoke, spherical+Cartesian feature shapes, build_model E/F
- :code:`test_tace_spherical_ops.py` — CGTP path / scatter TP / CgtpACE smoke
- :code:`test_tace_cartesian_ops.py` — cartnn ICTD / harmonics / Cartesian contraction smoke
- :code:`test_tace_parity_ops.py` — TACE spherical / Cartesian numerical parity vs vendored tace v0.1.0 fixtures
- :code:`test_tece_core.py` — TECE registration / YAML smoke, feature shapes, build_model E/F, ECE/RRA flag sensitivity
- :code:`test_equflash_core.py` — EquFlash irrep schedule, XPLOR / Bessel flavor, registration, SO(3) feature equivariance, build_model E/F and finite-difference forces
- :code:`test_equflash_v2_core.py` — EquFlashV2 registration, bilinear-gate equivariance, rotation, and energy/force finite difference
- :code:`test_tece_ops.py` — WignerD / LayoutTransform / uvSO2Linear / SO2Gate / ComplexProductBasis / RRA path smoke
- :code:`test_so3_wigner_backend.py` — shared e3nn/Jd Wigner-D backend (packed orthogonality, SO3_Rotation / fused / DPA4 quaternion adapters, high-l smoke)
- :code:`test_so3_grid.py` — unified flat lat–long :code:`SO3Grid` / :code:`S2GridProjector` protocol (roundtrip, Lebedev duck-type, grid table)
- :code:`test_dimenet_ops.py` — DimeNet envelope / Bessel flavor / SBF / triplets / bilinear interaction
- :code:`test_dimenet_core.py` — DimeNet registration / YAML smoke, hierarchical features, SO(3) invariance, build_model E/F conservation
- :code:`test_dimenet_parity_ops.py` — numerical parity vs PyG original DimeNet (not ++); energy uses PyG angles at atom ``i``; production Core uses paper angles at ``j``
- :code:`test_dimenetpp_core.py` — DimeNet++ registration / YAML smoke, hierarchical features at ``out_emb_size``, SO(3) invariance, build_model E/F, pair graph with no triplets
- :code:`test_dimenetpp_parity_ops.py` — numerical parity of ``DimeNetPPInteractionBlock`` vs PyG ``InteractionPPBlock`` (all-edge down-projection, then triplet gather), including an empty triplet list
- :code:`test_gemnet_ops.py` — GemNet Bessel flavor, reverse edges, receiver triplets, dihedral quadruplets, two-cutoff central bonds, real harmonics, bilinear aggregation
- :code:`test_gemnet_core.py` — GemNet-T/Q registration, hierarchical features, rotation / inversion / translation, batch isolation, build_model energy and conservative forces, example YAML cutoffs
- :code:`test_gemnet_oc_ops.py` — GemNet-OC scaled Gaussian, neighbor cap, canonical-half symmetrization, mixed-image triplets, legendre outer product, periodic quadruplet images
- :code:`test_gemnet_oc_core.py` — GemNet-OC registration / YAML smoke, isolated atom, rotation / inversion / translation / edge order, batch isolation, neighbor cap, build_model energy and conservative forces
- :code:`test_m3gnet_ops.py` — M3GNet spherical Bessel (``l = 0`` closed form, zero at the cutoff), ``r_ik`` joint basis, receiver triplets including periodic images, gated MLP
- :code:`test_m3gnet_core.py` — M3GNet registration / YAML smoke, three-body cutoff, rotation / inversion / translation / edge order, batch isolation, build_model energy and conservative forces
- :code:`test_mattersim_ops.py` — Graphormer segment softmax, centrality sum, post-softmax polynomial mask
- :code:`test_mattersim_core.py` — MatterSim registration / aliases / YAML smoke, M3GNet reduction when attention and centrality are off, attention cutoff, rotation / inversion / translation / edge order, batch isolation, build_model energy and conservative forces
- :code:`test_chgnet_ops.py` — CHGNet Fourier order, gemnet smooth Bessel, undirected bonds including periodic images, bond-graph cutoff, gated MLP layer norm
- :code:`test_chgnet_core.py` — CHGNet registration / YAML smoke, bond cutoff, charge latent, rotation / inversion / translation / edge order, batch isolation, build_model energy, magmom, and conservative forces
- :code:`test_alignn_ops.py` — ALIGNN Gaussian width ``1/Δ``, directed line graph and cosines, periodic images, per-channel edge gate, code versus paper layer order
- :code:`test_alignn_core.py` — ALIGNN registration / YAML smoke, when angles reach the atoms, rotation / inversion / translation / edge order, batch isolation, envelope, build_model energy and conservative forces including a finite difference
- :code:`test_spinconv_ops.py` — SpinConv Gaussian width ``σ = 3Δ``, symmetric neighbor cap, equivariant edge frame, leaf and cutoff frames, latitude mass under a roll, longitude pool versus a reflection
- :code:`test_spinconv_core.py` — SpinConv registration / YAML smoke, angles at fixed bond lengths, rotation / translation / edge order, inversion sensitivity, batch isolation, pair-distance scale, build_model energy and autograd forces
- :code:`test_et_ops.py` — ET neighbor embedding, attention versus the TorchMD-Net formula, vector cutoff, self pairs
- :code:`test_et_core.py` — ET registration / aliases / YAML smoke, rotation / inversion / translation / edge order, batch isolation, neighbor embedding, build_model energy, dipole, and conservative forces
- :code:`test_quinnet_ops.py` — QuinNet six-term mix, sender scatter of five-body II, distance-influence filters
- :code:`test_quinnet_core.py` — QuinNet registration / aliases / YAML smoke, rotation / inversion / translation / edge order, batch isolation, neighbor embedding, build_model energy, dipole, and conservative forces
- :code:`test_freecg_ops.py` — FreeCG four CG paths, group weights, 1.5-group channel roll, rotation and reflection, zero-vector match to ViSNet
- :code:`test_freecg_core.py` — FreeCG registration / aliases / YAML smoke, rotation / reflection / edge order, batch isolation, build_model energy, dipole, and conservative forces
- :code:`test_visnet_lsrm_ops.py` — ViSNet-LSRM fragment radius graph, Z-weighted centers, bipartite block rotation
- :code:`test_visnet_lsrm_core.py` — ViSNet-LSRM registration / YAML smoke, precomputed and k-means fragments, batch isolation, rotation / translation, build_model energy, dipole, and conservative forces
- :code:`test_tensornet_ops.py` — TensorNet decomposition, skew roundtrip, traceless outer product, O(3) product, TorchMD-Net pure-PyTorch op parity
- :code:`test_tensornet_core.py` — TensorNet registration / YAML smoke, self pairs, charge scale, rotation / inversion / translation / edge order, batch isolation, periodic image, build_model energy, dipole head, and conservative forces
- :code:`test_tensornet2_ops.py` — neutral charge equilibration, AIMNet damping, weighted Coulomb, reaction field, charge-conditioned message, fixed ZBL pair
- :code:`test_tensornet2_core.py` — TensorNet2 registration / YAML smoke, per-channel charge sum, rotation / inversion / batch isolation, build_model energy and conservative forces
- :code:`test_facto_ops.py` — FACTO cutoff derivative, Gaussian window, descriptor table, and algebraic attention / feedback / injection / transport reductions
- :code:`test_facto_core.py` — FACTO registration / YAML smoke, isolated-atom head, batch isolation, rotation / inversion / translation, force finite-difference conservation
- :code:`test_grace_ops.py` — GRACE Chebyshev / spherical Bessel / cutoff, channel-aligned CG equivariance, Finnis–Sinclair powers, linear density head
- :code:`test_grace_core.py` — GRACE registration / YAML smoke, preset table, batch isolation, 2L message ablation, rotation / inversion / translation, force finite-difference conservation, SimpleReadout contract
- :code:`test_cace_ops.py` — CACE monomials, invariant counts, two-body dot-product identity, rotation / inversion of the contraction, per-shell radial mix
- :code:`test_cace_core.py` — CACE registration / YAML smoke, cacefit preset widths, batch isolation, scalar-message ablation, rotation / inversion / translation, force finite-difference conservation, SimpleReadout contract
- :code:`test_camp_ops.py` — CAMP Chebyshev on ``r/rc``, species-pair envelope, supplement moment rules, dyad contraction, one-neighbor atomic moment
- :code:`test_camp_core.py` — CAMP registration / YAML smoke, Table S5 widths, batch isolation, edge order, rotation / inversion / translation, force finite-difference conservation, SimpleReadout on the last layer
- :code:`test_geodite_ops.py` — polynomial envelope, covalent attenuation, radial window, screened ZBL pair formula, Geodite spherical harmonics, vector rejection, zero-basis isolated match
- :code:`test_geodite_core.py` — Geodite registration / YAML smoke, published widths, batch isolation, isolated-atom energy, rotation / inversion / translation, force finite-difference conservation
- :code:`test_so3krates_core.py` — So3krates shapes, SimpleReadout contract, build_model E/F, SO(3) energy/force checks
- :code:`test_so3krates_parity_ops.py` — numerical parity vs vendored So3krates-torch (SH / L0 / FilterNet / attention / interaction)
- :code:`test_so3lr.py` — SO3LR priors (ZBL / erf-Coulomb / TS–QDO), readouts, and :code:`architecture: so3lr` build_model smoke test
- :code:`test_efa.py` — ERoPE / Lebedev / EFABlock, batch isolation, :code:`efa` / :code:`so3lr_efa` / SpookyNet :code:`use_efa` smoke tests
- :code:`test_sphere_sample_readout.py` — SphereSampleReadout shapes / scalar invariance smoke
- :code:`test_extractor.py` — fragment extraction: single C–C bonds stay capped when the search reaches a carbon by two routes (e.g. around a ring)
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

- [ ] :code:`presets/builtin.py` registration
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

DimeNet (:code:`dimenet`)
^^^^^^^^^^^^^^^^^^^^^^^^^

Gasteiger et al. (ICLR 2020) lives under :code:`enerzyme/models/dimenet/`
(:code:`interaction.py`, :code:`basis.py`, :code:`triplets.py`),
adapted from `gasteigerjo/dimenet <https://github.com/gasteigerjo/dimenet>`_ (MIT).
This is **original DimeNet** (bilinear SBF), not DimeNet++. Shared
:code:`DimeNetEnvelope` lives in :code:`enerzyme/models/mp/basis/envelope.py` with the
other radial envelopes. YAML :code:`BesselRBF` uses :code:`flavor: dimenet`.
The Core emits hierarchical :code:`atom_feature`; compose
:code:`HierachicalReadout` (:code:`head_type: mlp`) + :code:`EnergyReduce` /
:code:`Force` outside the Core. SBF zeros / normalizers use :code:`scipy`
(:code:`scipy.special.spherical_jn` + Brent). PyG parity
(:code:`test/test_dimenet_parity_ops.py`) uses optional
:code:`torch_geometric` (numpy 2 ``np.math`` shim in the test helper;
``DimeNet.forward`` is not called — no torch-sparse / torch-cluster).
Enerzymette only needs :code:`architecture: dimenet` plus a resolved
:code:`config.yaml`.

GemNet (:code:`gemnet`, :code:`gemnet-q`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gasteiger, Becker, and Günnemann (NeurIPS 2021, arXiv:2106.08903) live under
:code:`enerzyme/models/gemnet/` (:code:`layers.py`,
:code:`basis.py`, :code:`geometry.py`). This is an original implementation.
`TUM-DAML/gemnet_pytorch <https://github.com/TUM-DAML/gemnet_pytorch>`_ is
Hippocratic License 2.0; do not vendor it, and do not import it from tests.
Register :code:`gemnet` / :code:`gemnet-t` / :code:`gemnett` and
:code:`gemnet-q` / :code:`gemnetq` in :code:`get_ff_core`. Reuse
:code:`BesselRBF(flavor="gemnet")`, :func:`enerzyme.models.cutoff.polynomial_envelope`,
DimeNet spherical-Bessel zeros and :code:`real_sph_harm_l0`,
:code:`RandomAtomEmbedding`, :code:`segment_sum_coo`, and
:meth:`BaseFFCore.adopt_radius_stack`. The triplet index is local: two
incoming edges share the receiver, which is not the DimeNet path triplet.
The Core emits hierarchical :code:`atom_feature`. Compose bias-free
:code:`HierachicalReadout` (:code:`head_type: dense`) + :code:`EnergyReduce` /
:code:`Force`. ``basis_cutoff`` must equal ``emb_cutoff``. GemNet-Q sets
``cutoff_sr`` and ``int_cutoff`` above that embedding cutoff. Variance scales
stay at 1. Upstream checkpoints are not loaded: the harmonic channel order,
the species-table initialization, and the fitted scales differ.
Enerzymette only needs :code:`architecture: gemnet-t` or
:code:`architecture: gemnet-q` plus a resolved :code:`config.yaml`.

GemNet-OC (:code:`gemnet-oc`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gasteiger et al. (TMLR 2022, arXiv:2204.02782) extend that package with
:code:`stages_oc.py` and :code:`oc.py`. The interaction block follows
`facebookresearch/fairchem <https://github.com/facebookresearch/fairchem/tree/ocp2.0/ocpmodels/models>`__
on branch ``ocp2.0`` (MIT). Do not vendor it, and do not import it from tests.
Register :code:`gemnet-oc` / :code:`gemnetoc` / :code:`gemnet_oc`. Reuse
:class:`~enerzyme.models.gemnet.layers.TripletInteraction`,
:class:`~enerzyme.models.gemnet.layers.QuadrupletInteraction`,
:class:`~enerzyme.models.gemnet.layers.BilinearAggregate`, and
:func:`enerzyme.models.cutoff.polynomial_envelope`. The radial basis is
:code:`GaussianRBF(flavor="gemnet-oc")` at ``cutoff_sr``; the Core evaluates
it again at each interaction cutoff. Bind that layer with
``object.__setattr__`` so it stays parented only by ``pre_sequence``.
Quadruplet images use :func:`enerzyme.models.gemnet.geometry.quadruplet_graph`
with ``images=True``. GemNet-Q keeps the default ``images=False``. The Core
emits hierarchical :code:`atom_feature`. Compose the same bias-free
:code:`HierachicalReadout` and conservative :code:`Force`. Direct forces and
``qint_tags`` are not implemented. fairchem checkpoints do not load: the
edge concatenation order, the missing tag filter, the unit variance scales,
and the energy head all differ. Enerzymette only needs
:code:`architecture: gemnet-oc` plus a resolved :code:`config.yaml`.

M3GNet (:code:`m3gnet`)
^^^^^^^^^^^^^^^^^^^^^^^

Chen and Ong (Nat. Comput. Sci. 2022) live under :code:`enerzyme/models/m3gnet/`
(:code:`interaction.py`, :code:`basis.py`, :code:`geometry.py`).
The update rules follow `materialyzeai/matgl <https://github.com/materialyzeai/matgl>`_
(BSD). Do not import matgl, and do not load its checkpoints. Register
:code:`m3gnet` in :code:`get_ff_core`. Reuse
:func:`enerzyme.models.cutoff.polynomial_envelope` (order 3 on the three-body
term), DimeNet spherical-Bessel zeros / :code:`spherical_jn` /
:code:`real_sph_harm_l0`, :class:`~enerzyme.models.mp.nn.mlp.GatedMLP`,
:code:`RandomAtomEmbedding`, :code:`segment_sum_coo`, and
:meth:`BaseFFCore.adopt_radius_stack`. The bond expansion is the layer
:code:`SphericalBesselRBF` (flavor ``m3gnet``), not a :code:`BesselRBF` flavor:
its width is ``max_l * num_rbf`` and it carries no envelope.
:code:`threebody_cutoff` must be at most :code:`cutoff_sr`. The Core emits
flat :code:`atom_feature`. Compose :code:`SimpleReadout` (:code:`head_type: gated`)
+ :code:`EnergyReduce` / :code:`Force`. The gated head's hidden width follows
Core :code:`readout_hidden_width` (``units``). An isolated atom is the species
embedding, so the baseline is in that head rather than a zero descriptor.
:code:`Force` returns zero when that energy does not depend on coordinates.
Example: :code:`m3gnet_layers_example.yaml`. Tests: :code:`test/test_m3gnet_ops.py`,
:code:`test/test_m3gnet_core.py`. Enerzymette only needs :code:`architecture: m3gnet`
plus a resolved :code:`config.yaml`.

MatterSim (:code:`mattersim`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yang et al. (arXiv:2405.04967) live under :code:`enerzyme/models/mattersim/`
(:code:`attention.py`). The :func:`~enerzyme.models.presets.triplet.mattersim` preset shares the M3GNet triplet-update components. Register :code:`mattersim`,
:code:`m3gnet-graphormer`, and :code:`m3gnet_graphormer`. The three-body
convolution, spherical Bessel basis, gated MLP, triplet index, and polynomial
envelope are the M3GNet ones. The added structural encoder is supplementary
section S1.3: a centrality sum :code:`Linear(m_ij Φ_ij)` on the receiver, then
multi-head attention whose logits add :code:`Linear(Φ_ij)` and whose weights
are multiplied by the order-3 polynomial mask after the softmax. :code:`Φ` is
:code:`GaussianRBF` with ``flavor: SchNet`` and ``apply_cutoff_fn: false``;
the Core renames that output to ``graphormer_rbf``. Keys and values are read
from ``idx_j``, which is the in-cell atom of a periodic image. ``num_attn_layers: 0``
and ``use_centrality: false`` is M3GNet. The paper's GeoMFormer direct-force
decoder and lattice stress head are not this potential. Forces stay on
:code:`Force`. :code:`microsoft/mattersim` ships M3GNet checkpoints only and
raises ``NotImplementedError`` for Graphormer; those checkpoints do not load.
Example: :code:`mattersim_layers_example.yaml`. Tests:
:code:`test/test_mattersim_ops.py`, :code:`test/test_mattersim_core.py`.
Enerzymette only needs :code:`architecture: mattersim` plus a resolved
:code:`config.yaml`.

CHGNet (:code:`chgnet`)
^^^^^^^^^^^^^^^^^^^^^^^

Deng et al. (Nat. Mach. Intell. 2023) live under :code:`enerzyme/models/chgnet/`
(:code:`interaction.py`, :code:`basis.py`, :code:`geometry.py`).
The update rules follow `CederGroupHub/chgnet <https://github.com/CederGroupHub/chgnet>`_.
Do not import that package, and do not load its checkpoints. Register
:code:`chgnet` in :code:`get_ff_core`. Reuse
:code:`BesselRBF(flavor="gemnet")` for both smooth radial bases,
:class:`~enerzyme.models.mp.nn.mlp.GatedMLP` (optional ``norm`` and
``dropout``), :func:`~enerzyme.models.m3gnet.geometry.receiver_triplets`,
:code:`RandomAtomEmbedding`, :code:`segment_sum_coo`, and
:meth:`BaseFFCore.adopt_radius_stack`. The angle Fourier series stays in the
Core because the bond graph is part of the interaction. The second Bessel
layer's ``cutoff_sr`` is ``bond_cutoff``; the Core renames its output to
``rbf_bg``. The neighbor list must be bidirectional. The Core emits
``atom_feature`` and ``atom_feature_charge``. Compose
:code:`AbsoluteLinearReadout` (field :code:`Ma`) and :code:`SimpleReadout`
(``head_type: mlp``, two hidden layers, ``use_bias_out: true``) +
:code:`EnergyReduce` / :code:`Force`. Energies are eV
(``Hartree_in_E ≈ 27.211``) and distances are Å. The reference layer norms
are ``gmlp_norm: layer`` and ``readout_norm: layer``. Example:
:code:`chgnet_layers_example.yaml`. Tests: :code:`test/test_chgnet_ops.py`,
:code:`test/test_chgnet_core.py`. Enerzymette only needs
:code:`architecture: chgnet` plus a resolved :code:`config.yaml`.

ALIGNN (:code:`alignn`)
^^^^^^^^^^^^^^^^^^^^^^^

Choudhary and DeCost (npj Comput. Mater. 2021) live under
:code:`enerzyme/models/alignn/` (:code:`interaction.py`,
:code:`geometry.py`). The update follows
`usnistgov/alignn <https://github.com/usnistgov/alignn>`_, including
``ALIGNNAtomWise``. Do not import that package, and do not load its
checkpoints. Register :code:`alignn` in :code:`get_ff_core`. Reuse
:func:`enerzyme.models.layers.rbf.alignn_gaussian_terms` through
:code:`GaussianRBF(flavor="alignn")` for the bond expansion,
:func:`enerzyme.models.cutoff.polynomial_envelope` only when
``multiply_envelope`` is set,
:code:`RandomAtomEmbedding`, :code:`segment_sum_coo`, and
:meth:`BaseFFCore.adopt_radius_stack`. Do not reuse
:func:`enerzyme.models.m3gnet.geometry.receiver_triplets` or the CHGNet bond
graph: ALIGNN's line graph is the directed head-to-tail graph, and the
message updates the outgoing bond. The cosine Gaussian on ``[-1, 1]`` stays
in the Core. ``update_order`` is ``code`` or ``paper``; the gate stays the
released per-channel sigmoid. One incoming edge cancels that gate, so the
angle reaches an atom only after the updated bond competes with another
incoming bond. The Core emits flat :code:`atom_feature`.
Compose :code:`SimpleReadout` (:code:`head_type: dense`) +
:code:`EnergyReduce` / :code:`Force`. The default widths are ALIGNN-FF.
``PUBLISHED_PROPERTY_CORE_PARAMS`` is the 4 + 4, hidden-256, batch-norm
property model. Batch norm needs more than one row; the force field uses
layer norm. Example: :code:`alignn_layers_example.yaml`. Tests:
:code:`test/test_alignn_ops.py`, :code:`test/test_alignn_core.py`.
Enerzymette only needs :code:`architecture: alignn` plus a resolved
:code:`config.yaml`.

SpinConv (:code:`spinconv`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shuaibi et al. (arXiv:2106.09575) live under :code:`enerzyme/models/spinconv/`
(:code:`interaction.py`, :code:`geometry.py`). The grid
convolution follows
`facebookresearch/fairchem <https://github.com/facebookresearch/fairchem/blob/ocp2.0/ocpmodels/models/spinconv.py>`__
on branch ``ocp2.0``. Do not import that module, and do not load its
checkpoints. Register :code:`spinconv` in :code:`get_ff_core`. Reuse
:func:`enerzyme.models.layers.rbf.spinconv_gaussian_terms` through
:code:`GaussianRBF(flavor="spinconv")`,
:func:`enerzyme.models.layers.rbf.gaussian_expansion`,
:func:`enerzyme.models.cutoff.cosine_transition` for the second-axis weight,
:func:`enerzyme.models.activation.get_activation_fn` with fixed Swish
(``beta = 1``), :code:`RandomAtomEmbedding`, :code:`segment_sum_coo`, and
:meth:`BaseFFCore.adopt_radius_stack`. The latitude–longitude projection and
the species mixture stay in the Core. One species table is shared; each
block owns the maps of that vector. ``residual_first`` defaults to false,
matching the released loop. The Core emits flat :code:`atom_feature` of
width ``mid_hidden_channels``. Compose :code:`SimpleReadout`
(:code:`head_type: dense`) + :code:`EnergyReduce` / :code:`Force`. Detach
the bilinear weights. That is the released ``force_estimator: grad`` path:
the bond partner sits on the pole, and the latitude / longitude Jacobian is
undefined there. When the second axis is parallel to the bond or has
zero weight, build the normal from a lab axis so the frame stays a
rotation. Do not add the released lab-frame ``1e-4``. The force-centric
direct head is not a layer. Example: :code:`spinconv_layers_example.yaml`.
Tests: :code:`test/test_spinconv_ops.py`, :code:`test/test_spinconv_core.py`.
Enerzymette only needs :code:`architecture: spinconv` plus a resolved
:code:`config.yaml`.

External UMA (:code:`uma_qs`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register in :code:`get_ff_core` like other architectures. Keep fairchem imports inside :code:`enerzyme/models/esen/` so non-UMA installs do not import it until selected. Prefer shared :code:`layers/readout.py` and :code:`layers/spin.py` for Q/S heads. Package name :code:`esen/` is historical (UMA / eSCN-MD lineage); it is **not** the 2023 paper eSCN.

Native eSEN (:code:`esen`)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fu et al. (2025, arXiv:2502.12147) lives under :code:`enerzyme/models/esen/native/`
and is registered as :code:`architecture: esen`. The parent package
:code:`enerzyme/models/esen/` still exports :code:`UMAWrapperQS` lazily, so
importing the native Core does not import fairchem. The Core reuses shared
:code:`SO2_Convolution`, :code:`GateActivation`, :code:`SO3Linear`,
:code:`get_normalization_layer`, :code:`RadialFunction`, and
:code:`polynomial_envelope`. Species and distance embeddings stay in
:code:`RandomAtomEmbedding` and :code:`GaussianRBF(flavor="esen")` or
:code:`BesselRBF(flavor="esen")`. Compose :code:`SimpleReadout` +
:code:`EnergyReduce` + :code:`Force` outside the Core. Example:
:code:`esen_layers_example.yaml`. Tests: :code:`test/test_esen_core.py`.
Enerzymette only needs :code:`architecture: esen` plus a resolved
:code:`config.yaml`.

Native eSCN (:code:`escn`)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Paper eSCN (Passaro & Zitnick, 2023) lives under :code:`enerzyme/models/escn/` with shared SO(2)/SO(3) primitives in :code:`enerzyme/models/mp/equivariant/` (including :code:`Jd.pt` in package data). The Core emits :code:`atom_feature` (l=0 scalars with :code:`feature_irreps = "Cx0e"`) and :code:`atom_sphere_feature` (SH grid layout for :code:`SphereSampleReadout`); compose :code:`SimpleReadout` / :code:`SphereSampleReadout` / :code:`EnergyReduce` / :code:`Force` outside the Core. No fairchem dependency. Offline numerical parity against vendored fairchem :code:`fairchem_core-1.10.0` blocks is in :code:`test/test_escn_parity_*.py` (ops + Message/LayerBlock; injected edge frames; not OC20 E/F). Enerzymette and other YAML-driven workflows only need :code:`architecture: escn` plus a resolved :code:`config.yaml` — checkpoint layout is unchanged.

EquiformerV2 (:code:`equiformer_v2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Liao et al. (ICLR 2024) lives under :code:`enerzyme/models/equiformer_v2/` and extends shared :code:`so3/` (rotate-inv rescale, component grids, :code:`SO3_LinearV2`). The Core emits the same :code:`atom_feature` / :code:`atom_sphere_feature` contract as eSCN; compose :code:`SimpleReadout` or :code:`EquiformerV2FeedForwardReadout` plus :code:`EnergyReduce` / :code:`Force` outside the Core. Offline parity vs vendored :code:`atomicarchitects/equiformer_v2` nets is in :code:`test/test_equiformer_v2_parity_*.py`. Enerzymette only needs :code:`architecture: equiformer_v2` plus a resolved :code:`config.yaml`.

EquFlashV2 (:code:`equflashv2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Samsung Electronics' EquFlashV2 lives next to EquFlash in
:code:`enerzyme/models/equflash/` (:code:`defaults_v2.py`, :code:`interaction_v2.py`,
shared :code:`irreps.py`). Register via :code:`get_ff_core("equflashv2")`
(alias :code:`equflash_v2`). The reference repository is CC BY-NC-SA 4.0 and
uses cuequivariance; this stack is an e3nn reimplementation and does not load
those checkpoints. It reuses EquFlash's :code:`BesselRBF(flavor="equflash")`,
:code:`xplor` / :code:`cuton`, and :meth:`BaseFFCore.adopt_radius_stack`.
The channelwise convolution writes onto the requested irreps, so it does not
use :func:`tp_out_irreps_with_instructions` (that helper keeps one slot per
path for EquFlash's following linear). Messages go through
:func:`segment_sum_coo`. :code:`EquivariantRMSNormSH`,
:code:`SimpleReadout` (``activate_hidden: false``, ``use_bias: false``), and
:code:`AtomicAffine(order="scale_then_shift")` stay outside the convolution.
The Core emits flat :code:`atom_feature`. Example:
:code:`equflashv2_layers_example.yaml`. Tests: :code:`test/test_equflash_v2_core.py`.
Enerzymette: :code:`architecture: equflashv2` plus a resolved :code:`config.yaml`.

Nequix (:code:`nequix`)
^^^^^^^^^^^^^^^^^^^^^^^

Koker and Smidt (arXiv:2508.16067) lives under :code:`enerzyme/models/nequix/`
(:code:`interaction.py`). Register via
:code:`get_ff_core("nequix")`. Reuse :code:`OneHotAtomEmbedding`,
:code:`BesselRBF(flavor="equflash")`, :code:`cutoff_fn: nequip`,
:code:`RMSLayerNorm` in :code:`e3nn_nn`, a bias-free :code:`SimpleReadout`,
and :code:`AtomicAffine(order="scale_then_shift")`. The external
:code:`nequip` wrapper is unchanged and is not this model. Published MPtrj
hyperparameters are :code:`PUBLISHED_MP_CORE_PARAMS`. The default stack keeps
those widths small and uses the MPtrj neighbor normalization. Example:
:code:`nequix_layers_example.yaml`. Tests:
:code:`test/test_nequix_core.py`. Enerzymette: :code:`architecture: nequix`
plus a resolved :code:`config.yaml`.

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

DPA3 (:code:`dpa3`)
^^^^^^^^^^^^^^^^^^^^^

Zhang et al. (arXiv:2506.01686) live under :code:`enerzyme/models/dpa3/` (:code:`interaction.py`, :code:`geometry.py`). Register via :code:`get_ff_core("dpa3")`. DeePMD-kit is not imported. The Core is the order-2 repflow update only. Reuse :func:`enerzyme.models.cutoff.polynomial_transition` (the DeePMD radial gate), :func:`enerzyme.models.cutoff.exponential_switch` (Eq. 12), :func:`enerzyme.models.layers.symmetrize.dp_symmetrize`, :func:`enerzyme.models.functional.reduce_edge`, :func:`enerzyme.models.activation.resolve_scalar_activation` (including :code:`silut`), and :meth:`BaseFFCore.adopt_radius_stack`. :code:`RandomAtomEmbedding` is the initial node state. :code:`SimpleReadout`, :code:`AtomicAffine`, :code:`EnergyReduce`, and :code:`Force` stay outside. ``e_sel`` / ``a_sel`` are :math:`N_m`, not a fixed neighbor cap. Published widths are :code:`PUBLISHED_CORE_PARAMS`. Example: :code:`dpa3_layers_example.yaml`. Tests: :code:`test/test_dpa3_ops.py`, :code:`test/test_dpa3_core.py`. Enerzymette: :code:`architecture: dpa3` plus a resolved :code:`config.yaml`.

DPA4 (:code:`dpa4`)
^^^^^^^^^^^^^^^^^^^^^

DPA4 (Li et al., 2026, arXiv:2606.02419) lives under :code:`enerzyme/models/dpa4/` as :code:`interaction.py` / :code:`so2.py`. EMFA orchestration (:code:`SO2Convolution`, :code:`DynamicRadialDegreeMixer`, :code:`EdgeCache`) stays local. Shared pieces in :code:`enerzyme/models/mp/equivariant/` include :code:`C3CutoffEnvelope`, Apache e3x Lebedev tables (:code:`lebedev_grids.npz`, also used by EFA) / :code:`S2LebedevProjector`, packed/m-major :code:`indexing`, :code:`FocusSO2Linear`, :code:`SO3FocusLinear`, :code:`SO3GatedActivation`, :code:`EquivariantDegreeRMSNorm`, :code:`BesselC3RadialBasis` / :code:`RadialMLP`, and quaternion edge frames (:code:`build_edge_quaternion`) that share the e3nn/:code:`Jd` Wigner-D backend (:code:`wigner_from_rotation_matrix`) with eSCN / EquiformerV2 / EquiformerV3. :code:`WignerDCalculator` maps :code:`R(q)` into DPA4's historical Cartesian basis (:code:`A R Aᵀ`) before that backend so SO(2) / GIE stay equivariant for any :code:`lmax` in :code:`Jd.pt`. Flat lat–long :code:`SO3Grid` and Lebedev :code:`S2LebedevProjector` share the :code:`S2GridProjector` contract (:code:`to_grid` / :code:`from_grid`); EquiformerV3 FFN uses :code:`SO3Grid`, DPA4 FFN defaults to Lebedev. The Core emits :code:`atom_feature` / :code:`atom_sphere_feature`; compose :code:`SimpleReadout`, :code:`EnergyReduce`, and :code:`Force` outside it. Tests: :code:`test/test_dpa4_core.py`, :code:`test/test_dpa4_parity_ops.py`, :code:`test/test_so3_wigner_backend.py`, :code:`test/test_so3_grid.py`. Enerzymette only needs :code:`architecture: dpa4` plus a resolved :code:`config.yaml`.

DPA4C (:code:`dpa4c`)
^^^^^^^^^^^^^^^^^^^^^

Li et al. (arXiv:2608.19041) live under :code:`enerzyme/models/dpa4c/` (:code:`geometry.py`, :code:`radial.py`, :code:`readout.py`). Register via :code:`get_ff_core("dpa4c")`. DeePMD-kit is not imported. The Core is one destination aggregation: a shared SwiGLU radial map, ordered-pair FiLM from the atom embedding, real Cartesian harmonics through degree 4, and fixed polynomial invariants (Grams, bispectrum, projected quartic). Reuse :class:`~enerzyme.models.cutoff.C3CutoffEnvelope`, :class:`~enerzyme.models.mp.basis.radial.BesselC3RadialBasis` with ``apply_envelope=False``, :func:`~enerzyme.models.functional.reduce_edge`, and :func:`~enerzyme.models.mp.equivariant.s2.lebedev.load_lebedev_rule` for the Gaunt tables. :code:`RandomAtomEmbedding` is the initial scalar feature and is concatenated again at the end of the descriptor. :code:`SimpleReadout` with ``head_type: mlp`` and ``identity_residual: true`` is the energy MLP. :code:`AtomicAffine`, :code:`EnergyReduce`, and :code:`Force` stay outside. Published grades are :code:`PUBLISHED_GRADES`. :meth:`DPA4CCore.fit_output_calibration` freezes the diagonal preconditioner; a geometric column with RMS at most ``1e-12`` keeps scale 1, and the default preconditioner is the identity. ``build`` rejects ``cutoff_fn`` and the forward rejects self-loops. Duo blocks, native spin, charge conditioning, and compressed CUDA inference are outside this port. Example: :code:`dpa4c_layers_example.yaml`. Tests: :code:`test/test_dpa4c_ops.py`, :code:`test/test_dpa4c_core.py`. Enerzymette: :code:`architecture: dpa4c` plus a resolved :code:`config.yaml`.

TACE (:code:`tace`)
^^^^^^^^^^^^^^^^^^^

Xu et al. (arXiv:2509.14961; Cartesian-3j arXiv:2512.16882) lives under :code:`enerzyme/models/tace/` (:code:`interaction.py`), adapted from `xvzemin/tace <https://github.com/xvzemin/tace>`_ (MIT). Shared flat-Irreps helpers (:code:`IrrepsLinear`, :code:`generate_paths`, :code:`O3ScatterTensorProduct`, :code:`get_gated_nonlinear`, …) live in :code:`enerzyme/models/mp/equivariant/e3nn/`; radial channel MLPs use :code:`blocks.radial_mlp.RadialMLP`. Register via :code:`get_ff_core("tace")`. Core param :code:`tensor_basis` selects spherical e3nn CGTP or Cartesian ICT (:code:`cartnn` vendored from tace v0.1.0 + :code:`cartesian/`). Scope includes edge embedding/update, BB element-aware residual, and density/avg scatter-norm; TECE / SO2 / RRA live under :code:`architecture: tece`; ZBL / LES / UIE stay out. Emit :code:`atom_feature` for :code:`SimpleReadout`. Examples: :code:`tace_layers_example.yaml`, :code:`tace_cartesian_layers_example.yaml`. Tests: :code:`test/test_tace_*.py`.

TECE (:code:`tece`)
^^^^^^^^^^^^^^^^^^^

Xu et al. (arXiv:2607.10664) lives under :code:`enerzyme/models/tece/` (:code:`interaction.py`), adapted from `xvzemin/tace <https://github.com/xvzemin/tace>`_ v0.2.0 (MIT). Register via :code:`get_ff_core("tece")`. The Core seeds equivariant features from scalar embeddings, then stacks uvSO2 interactions with Edge Cluster Expansion (:code:`ComplexProductBasis`) and Radial Rotary Attention, plus node-side :code:`CgtpACE` reused from TACE. Shared SO(2) primitives (:code:`WignerD` recursive/direct, :code:`uvSO2Linear`, :code:`SO2Gate`, :code:`LayoutTransform`) live in :code:`enerzyme/models/mp/equivariant/`. Requires :code:`Lmax == lmax`. Do not conflate RRA with EFA Euclidean RoPE. Emit :code:`atom_feature` for :code:`SimpleReadout`. Example: :code:`tece_layers_example.yaml`. Tests: :code:`test/test_tece_*.py`. Enerzymette: :code:`architecture: tece` + resolved :code:`config.yaml`.

Allegro (:code:`allegro`)
^^^^^^^^^^^^^^^^^^^^^^^^^

Musaelian et al. (Nat. Commun. 2023) live under :code:`enerzyme/models/allegro/` (:code:`interaction.py`, :code:`irreps.py`). Register via :code:`get_ff_core("allegro")`. Reuse :code:`BesselRBF(flavor="nequip")`, :class:`~enerzyme.models.mp.nn.mlp.VariancePreservingMLP` (the NequIP scalar-MLP init, also the :code:`EdgeReadout` head), :func:`enerzyme.models.nequip.interaction.tp_path_exists`, and :meth:`BaseFFCore.adopt_radius_stack`. The species tables are :code:`PairChemicalRadial`, not inside the Core. The Core emits ``edge_feature``. :code:`EdgeReadout` and :code:`EdgeEnergyReduce` are shared layers so a later pair model can reuse the same head and the same directed sum. ``avg_num_neighbors`` is a build parameter read by both the environment sum and the energy sum. Example: :code:`allegro_layers_example.yaml`. Tests: :code:`test/test_allegro_core.py`. Enerzymette: :code:`architecture: allegro` + resolved :code:`config.yaml`. Checkpoints from mir-group/allegro are not weight-compatible.

NequIP (:code:`nequip`)
^^^^^^^^^^^^^^^^^^^^^^^

Batzner et al. (Nat. Commun. 2022) live under :code:`enerzyme/models/nequip/` (:code:`interaction.py`). Register via :code:`get_ff_core("nequip")`. The external ``nequip`` wrapper has been removed; do not import that package. Reuse :code:`BesselRBF(flavor="nequip")` (sinc basis × :code:`PolynomialEnvelope` × :math:`2\pi/r_{\max}^2`), :code:`RandomAtomEmbedding`, :code:`tp_out_irreps_with_instructions` / e3nn :code:`Gate`, :func:`enerzyme.models.activation.resolve_scalar_activation`, and :meth:`BaseFFCore.adopt_radius_stack`. The species embedding is the initial scalar state and the fixed node attribute for the self-connection. The first layer omits that connection unless :code:`sc_first` (isolated atoms stay at zero). The last convolution emits :math:`\ell=0` only, so :code:`atom_feature` is ``{F}x0e`` for :code:`SimpleReadout`. Per-element shift is :code:`AtomicAffine`, not a head inside the Core. Presets ``S``–``XL`` follow the current foundation widths; ``paper`` is the 2022 radial MLP. Example: :code:`nequip_layers_example.yaml`. Tests: :code:`test/test_nequip_core.py`. Enerzymette: :code:`architecture: nequip` + resolved :code:`config.yaml`. Checkpoints from mir-group/nequip are not weight-compatible.

EQNorm (:code:`eqnorm`)
^^^^^^^^^^^^^^^^^^^^^^^

Chen, Fu, Yang, and Chen live under :code:`enerzyme/models/eqnorm/`
(:code:`interaction.py`, :code:`layernorm.py`). The update
follows `yzchen08/eqnorm <https://github.com/yzchen08/eqnorm>`_ (MIT). Do not
import that package, and do not load its checkpoints. Register :code:`eqnorm`
in :code:`get_ff_core`. Reuse :code:`OneHotAtomEmbedding`,
:code:`BesselRBF(flavor="equflash")` windowed by :code:`cutoff_fn: nequip`,
:func:`enerzyme.models.nequip.interaction.tp_path_exists`, e3nn :code:`Gate`
and :code:`FullyConnectedNet`, :func:`enerzyme.models.functional.reduce_edge`,
:class:`~enerzyme.models.mp.nn.mlp.VariancePreservingMLP` (the scalar head),
and :meth:`BaseFFCore.adopt_radius_stack`. The species linear stays in the
Core: both the tensor state and the scalar accumulator start there.
:func:`enerzyme.models.eqnorm.interaction.eqnorm_tp_paths` keeps the reference
instruction order; :func:`tp_out_irreps_with_instructions` sorts that order
for Nequix, so the two must not be swapped.
:class:`~enerzyme.models.eqnorm.layernorm.EqnormRMSNorm` is the reference
degree-balanced norm. eSEN :class:`~enerzyme.models.mp.equivariant.e3nn.rms_norm.RMSLayerNorm`
uses a different reduction and does not replace it. Each layer is a tensor
convolution, a gate, and a scalar convolution. The scalar convolution does
not write the tensor state. ``normalization`` is ``constant`` (MPtrj, and the
max variant with ``message_envelope``) or ``degree`` (OMat).
``polynomial_p`` is the second window and is read only when
``message_envelope`` is true. :func:`enerzyme.models.cutoff.nequip_envelope`
is fixed at exponent 6, so that Core value stays 6 and the two windows match.
The Core emits
flat :code:`atom_feature`. Compose :code:`SimpleReadout`
(``head_type: variance_preserving``, hidden width ``num_features // 2``) +
:code:`AtomicAffine` (``order: scale_then_shift``) + :code:`EnergyReduce` /
:code:`Force`. Published widths are :code:`PUBLISHED_MPTRJ_CORE_PARAMS`,
:code:`PUBLISHED_MAX_CORE_PARAMS`, and :code:`PUBLISHED_OMAT_CORE_PARAMS`.
Example: :code:`eqnorm_layers_example.yaml`. Tests:
:code:`test/test_eqnorm_ops.py`, :code:`test/test_eqnorm_core.py`.
Enerzymette only needs :code:`architecture: eqnorm` plus a resolved
:code:`config.yaml`.

FACTO (:code:`facto`)
^^^^^^^^^^^^^^^^^^^^^

Liu et al. (arXiv:2611.03276) live under :code:`enerzyme/models/facto/` (:code:`geometry.py`, :code:`interaction.py`, :code:`readout.py`). There is no upstream repository. Register via :code:`get_ff_core("facto")`. The degree-9 window is :func:`enerzyme.models.cutoff.facto_polynomial_cutoff` (YAML :code:`cutoff_fn: facto`), distinct from :class:`~enerzyme.models.cutoff.C3CutoffEnvelope`. Real spherical harmonics reuse :func:`enerzyme.models.mp.basis.angular.spherical_harmonics` with ``layout="e3nn"`` and integral normalization, which covers ``L=6``. The Core emits hierarchical scalars plus isolated-atom states; :code:`FACTOEnergyReadout` and :code:`ZBLRepulsionEnergy` stay outside it. The 31-d elemental table is an Enerzyme stand-in — the paper does not list the components — and stops at Z=94. CUDA neighbor kernels and BF16×3 emulation are intentionally not ported. Tests: :code:`test/test_facto_ops.py`, :code:`test/test_facto_core.py`. Enerzymette: :code:`architecture: facto` + resolved :code:`config.yaml`.

CACE (:code:`cace`)
^^^^^^^^^^^^^^^^^^^

Cheng (npj Comput. Mater. 2024) lives under :code:`enerzyme/models/cace/` (:code:`basis.py`, :code:`interaction.py`, :code:`readout.py`, :code:`presets.py`). Register via :code:`get_ff_core("cace")`. This is an original implementation of the published Cartesian contraction. `BingqingCheng/cace <https://github.com/BingqingCheng/cace>`_ is CC BY-NC 4.0 and is not imported; do not vendor it, and do not expect its checkpoints to load. Reuse :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``), :code:`RandomAtomEmbedding`, :func:`enerzyme.models.cutoff.polynomial_envelope`, and :code:`EnergyReduce` / :code:`Force`. The monomial list and the body-order contraction stay in the package: they are the Cartesian cluster expansion, and a spherical-harmonic tensor product is not used. The Core emits :code:`atom_feature` ``[N, F]``. :code:`CACEEnergyReadout` is the linear map plus the SiLU MLP; :code:`SimpleReadout` also fits. Presets follow the cacefit training scripts (``water``, ``water-local``, ``ethanol``, ``bpa``, ``hea``). Leave ``cutoff_fn`` unset so the envelope order stays on the Core. Tests: :code:`test/test_cace_ops.py`, :code:`test/test_cace_core.py`. Enerzymette: :code:`architecture: cace` + resolved :code:`config.yaml`.

CAMP (:code:`camp`)
^^^^^^^^^^^^^^^^^^^

Wen, Huang, Dai, and Adhikari (npj Comput. Mater. 2025) live under :code:`enerzyme/models/camp/` (:code:`moment.py`, :code:`radial.py`, :code:`rules.py`, :code:`readout.py`, :code:`presets.py`). Register via :code:`get_ff_core("camp")`. This is an original implementation of the published Cartesian moments. `wengroup/camp <https://github.com/wengroup/camp>`_ is not imported; do not vendor it, and do not expect its checkpoints to load. Reuse :code:`RandomAtomEmbedding`, :code:`RangeSeparation`, :func:`enerzyme.models.cutoff.polynomial_envelope`, and :code:`EnergyReduce` / :code:`Force`. The species-pair Chebyshev basis stays inside each moment layer because the reference constructs a new radial map per layer. It is not the GRACE Chebyshev argument. The moment rules are the supplement's full contraction onto the highest rank, not a spherical Clebsch–Gordan product. The Core emits :code:`atom_feature` ``[N, C, T]``. :code:`CAMPEnergyReadout` is the per-layer head; :code:`SimpleReadout` reads the last layer. Presets follow Table S5. ``dim_embedding`` is ``max_u + 1``. Leave ``cutoff_fn`` unset. Tests: :code:`test/test_camp_ops.py`, :code:`test/test_camp_core.py`. Enerzymette: :code:`architecture: camp` + resolved :code:`config.yaml`. Set ``atomic_numbers`` and ``avg_num_neighbors`` in that YAML.

GRACE (:code:`grace`)
^^^^^^^^^^^^^^^^^^^^^

Lysogorskiy, Bochkarev, and Drautz (arXiv:2508.17936; method in Phys. Rev. X 14, 021036) live under :code:`enerzyme/models/grace/` (:code:`basis.py`, :code:`coupling.py`, :code:`radial.py`, :code:`readout.py`, :code:`presets.py`). Register via :code:`get_ff_core("grace")`. This is an original implementation of the published recursion. `grace-tensorpotential <https://github.com/ICAMS/grace-tensorpotential>`_ is Academic Software License and is not imported; do not vendor it, and do not expect its checkpoints to load. Reuse :code:`RandomAtomEmbedding`, :code:`RangeSeparation`, real spherical harmonics (:code:`layout="e3nn"`), :func:`enerzyme.models.cutoff.powered_cosine_window`, :func:`enerzyme.models.functional.reduce_edge`, and :code:`EnergyReduce` / :code:`Force`. The channel-aligned Clebsch–Gordan product stays in the package: it is the ACE recursion, and e3nn's generated tensor product is not used. The Core emits :code:`atom_feature` ``[N, n_densities]``. :code:`GRACEEnergyReadout` implements the foundation linear+MLP head and the Finnis–Sinclair powers; :code:`SimpleReadout` also fits. Presets follow Table S1 and the public preset defaults. Tests: :code:`test/test_grace_ops.py`, :code:`test/test_grace_core.py`. Enerzymette: :code:`architecture: grace` + resolved :code:`config.yaml`.

Geodite (:code:`geodite`)
^^^^^^^^^^^^^^^^^^^^^^^^^

Reschützegger et al. (J. Chem. Theory Comput. 2026, doi:10.1021/acs.jctc.6c00712) live under :code:`enerzyme/models/geodite/` (:code:`geometry.py`, :code:`interaction.py`). Register via :code:`get_ff_core("geodite")`. Adapted from `IBM/materials <https://github.com/IBM/materials/tree/main/models/pos_egnn/mlip>`_ (Apache-2.0). IBM checkpoints do not load. Reuse :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``), :code:`RandomAtomEmbedding`, :code:`HierachicalReadout` (bias-free dense heads), :code:`EnergyReduce` / :code:`Force`, and :func:`enerzyme.models.cutoff.polynomial_envelope` (scalar or per-edge cutoff). New shared layers are :code:`RadialWindow`, :code:`IsolatedAtomResidual`, and :code:`ScreenedZBLRepulsionEnergy`. The polynomial spherical harmonics through :code:`lmax` 4 live in :mod:`enerzyme.models.mp.basis.angular` and are re-exported from the package. ViSNet uses the ``lmax <= 2`` prefix. The component order is the inner-product basis (``l = 1`` is ``(x, y, z)``). :func:`enerzyme.models.mp.ops.rejection.vector_rejection` is shared with ViSNet. The Core emits :code:`atom_feature` ``[N, C, K]``, :code:`isolated_atom_feature`, and :code:`atom_steerable_feature`. Leave ``cutoff_fn`` unset so the envelope stays in :code:`RadialWindow`. The default stack is Geodite-MP in eV. Example: :code:`geodite_layers_example.yaml`. Tests: :code:`test/test_geodite_ops.py`, :code:`test/test_geodite_core.py`. Enerzymette: :code:`architecture: geodite` + resolved :code:`config.yaml`.

PaiNN (:code:`painn`)
^^^^^^^^^^^^^^^^^^^^^

Schütt, Unke, and Gastegger (ICML 2021, arXiv:2102.03150) live under :code:`enerzyme/models/painn/` (:code:`interaction.py`), adapted from SchNetPack 2.0.4 (MIT). Register via :code:`get_ff_core("painn")`. Reuse :code:`BesselRBF(flavor="painn")`, :code:`cosine` cutoff values from :code:`RangeSeparation`, and :code:`RandomAtomEmbedding`. The Core emits :code:`atom_feature` and Cartesian :code:`atom_vector_feature` ``[N, 3, F]``. Scalar properties use :code:`SimpleReadout` (:code:`two_layer` + :code:`pyramidal: true` matches the SchNetPack two-layer width schedule; pass SiLU as ``swish`` with ``alpha=beta=1``, ``learnable: false``). Dipole and polarizability use :code:`EquivariantDipole` / :code:`EquivariantPolarizability`. The gated block itself is :code:`enerzyme.models.mp.nn.gate` so later Cartesian cores can share it. The optional :code:`xpainn` preset uses XequiNet. Examples: :code:`painn_layers_example.yaml`, :code:`painn_spectra_layers_example.yaml`. Tests: :code:`test/test_painn_*.py`. Enerzymette: :code:`architecture: painn` + resolved :code:`config.yaml`. The ASE calculator and :code:`enerzyme listen` copy :code:`polarizability` when the stack emits it.

REANN (:code:`reann`)
^^^^^^^^^^^^^^^^^^^^^

Zhang, Xia, and Jiang (J. Chem. Phys. 2022) live under :code:`enerzyme/models/reann/` (:code:`density.py`, :code:`network.py`). Register via :code:`get_ff_core("reann")`. Do not import `zhangylch/REANN <https://github.com/zhangylch/REANN>`_; its checkpoints do not load. Reuse :code:`RandomAtomEmbedding`, :func:`enerzyme.models.cutoff.cosine_transition` (squared inside the Core), :class:`~enerzyme.models.activation.ReluLike`, :class:`~enerzyme.models.activation.TanhLike`, :class:`~enerzyme.models.activation.Swish`, and :code:`SimpleReadout`. The new shared head is :class:`~enerzyme.models.layers.response.DerivativePolarizabilityLayer` for any scalar Core. ``num_iterations = 0`` is EANN. ``RangeSeparation`` ``cutoff_sr`` must equal the Core cutoff, because the neighbor list and the squared cosine window share that radius. Per-element coefficient networks need ``atomic_numbers``. The energy head stays a shared readout. :func:`enerzyme.models.reann.density.pair_gaussian_orbitals`, :func:`enerzyme.models.reann.density.aggregate_weighted_orbitals`, and :func:`enerzyme.models.reann.density.invariant_orbital_density` are the pieces EquiREANN reuses. Example: :code:`reann_layers_example.yaml`. Tests: :code:`test/test_reann_ops.py`, :code:`test/test_reann_core.py`. Enerzymette: :code:`architecture: reann` + resolved :code:`config.yaml`.

EquiREANN (:code:`equireann`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wu, Xia, Zhang, and Jiang (J. Phys. Chem. A 2024) live under :code:`enerzyme/models/equireann/` (:code:`message.py`). Register via :code:`get_ff_core("equireann")`. Aliases are :code:`equi-reann` and :code:`equi_reann`. Do not import `bjiangch/EquiREANN <https://github.com/bjiangch/EquiREANN>`_; its checkpoints do not load. The :func:`~enerzyme.models.presets.cartesian_tensor.equireann` and :func:`~enerzyme.models.presets.cartesian_tensor.reann` presets share their orbital-density construction and use their respective contraction tensors. The geometric orbitals, the aggregate, the invariant square, and the coefficient MLP stay in the REANN package. :func:`enerzyme.models.equireann.message.add_neighbor_orbital` is the only new tensor step: ``pair + fc * ψ_neighbor``. The Core still emits invariant :code:`atom_feature`, so :code:`SimpleReadout`, :code:`AtomicCharge2Dipole`, and :code:`DerivativePolarizability` apply unchanged. ``num_iterations = 0`` matches REANN. The contraction has one slice per iteration. Example: :code:`equireann_layers_example.yaml`. Tests: :code:`test/test_equireann_ops.py`, :code:`test/test_equireann_core.py`. Enerzymette: :code:`architecture: equireann` + resolved :code:`config.yaml`.

NewtonNet (:code:`newtonnet`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Haghighatlari et al. (Digital Discovery 2022, arXiv:2108.02913) live under :code:`enerzyme/models/newtonnet/` (:code:`interaction.py`, :code:`loss.py`). Register via :code:`get_ff_core("newtonnet")`. Do not import `THGLab/NewtonNet <https://github.com/THGLab/NewtonNet>`_; its checkpoints do not load. Reuse :code:`BesselRBF` (default flavor, ``apply_cutoff_fn: false``), :code:`RandomAtomEmbedding`, :func:`enerzyme.models.cutoff.polynomial_envelope`, :func:`enerzyme.models.functional.segment_sum_coo`, and :code:`SimpleReadout` (``head_type: mlp``, ``hidden_dims: [128, 64]``). The envelope is applied inside the Core after each layer's radial linear. The default order is 9. ``variant: paper`` is Eqs. 7–17, including a separate displacement. ``variant: repo`` is ``InteractionNet`` on the repository ``main`` branch; its energy head is two hidden layers of width ``F``. Both emit :code:`atom_feature`, Cartesian :code:`atom_vector_feature` ``[N, 3, F]``, and :code:`latent_force` ``[N, 3]``, so :code:`EquivariantDipole` / :code:`EquivariantPolarizability` apply unchanged. Conservative forces stay on :code:`Force`. ``layer_norm`` applies to the invariant features only. ``latent_force_cosine`` is installed for this architecture and removed when another architecture is loaded. Example: :code:`newtonnet_layers_example.yaml`. Tests: :code:`test/test_newtonnet_ops.py`, :code:`test/test_newtonnet_core.py`. Enerzymette: :code:`architecture: newtonnet` + resolved :code:`config.yaml`.

LEFTNet (:code:`leftnet`) and AlphaNet (:code:`alphanet`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Du et al. (NeurIPS 2023, arXiv:2304.04757) live under :code:`enerzyme/models/leftnet/`. Yin et al. (npj Comput. Mater. 2025, arXiv:2501.07155) live under :code:`enerzyme/models/alphanet/`. Register via :code:`get_ff_core("leftnet")` and :code:`get_ff_core("alphanet")`. Do not import `yuanqidu/LeftNet <https://github.com/yuanqidu/LeftNet>`_ or `zmyybc/AlphaNet <https://github.com/zmyybc/AlphaNet/tree/jax_and_zbl>`_; their checkpoints do not load. Shared frames, local substructure encoding, the radial filter, the Cartesian message, and the frame transition live in :code:`enerzyme/models/mp/message/frame/`. LEFTNet reads :code:`ExpNormalSmearing` with ``alpha: 1`` and centers each graph before the edge cross product. ``fte_scalarization: l2`` is the released norm. ``frame`` is the paper node-frame update. AlphaNet reads :code:`BesselRBF` (``flavor: equflash``, ``apply_cutoff_fn: false``), LayerNorms one species embedding, and keeps a complex matrix-product state beside the scalar stream. The released edge frame is unnormalized, and its first axis is the unit ``vij``. The default frame transition is ``l2``, so the Core stays rotationally equivariant. ``l1`` is the released sum of absolute Cartesian components and is opt-in. Atoms with no neighbors use the identity node frame. Both Cores emit :code:`atom_feature` and Cartesian :code:`atom_vector_feature` ``[N, 3, F]``. AlphaNet also emits :code:`quantum_feature`. :code:`SimpleReadout`, :code:`AtomicAffine`, :code:`EnergyReduce`, and :code:`Force` stay outside, so :code:`EquivariantDipole` applies. The reference ZBL term is not inside the Core. Examples: :code:`leftnet_layers_example.yaml`, :code:`alphanet_layers_example.yaml`. Tests: :code:`test/test_local_frame_ops.py`, :code:`test/test_leftnet_core.py`, :code:`test/test_alphanet_core.py`. Enerzymette: :code:`architecture: leftnet` or :code:`architecture: alphanet` + resolved :code:`config.yaml`.

Equivariant Transformer (:code:`et`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Thölke and De Fabritiis (ICLR 2022, arXiv:2202.02541) live under :code:`enerzyme/models/et/` (:code:`interaction.py`). Register via :code:`get_ff_core("et")`. Aliases are :code:`torchmd-et`, :code:`torchmd_et`, and :code:`equivariant-transformer`. The pure PyTorch rules follow ``torchmdnet.models.torchmd_et`` (MIT). Do not import that package, do not port the Warp neighbor kernels, and do not expect its checkpoints to load. Reuse :code:`ExpNormalSmearing`, :code:`cosine` windows from :code:`RangeSeparation`, :code:`RandomAtomEmbedding` (the atom state and, inside the neighbor block, the second species table), :func:`enerzyme.models.mp.ops.self_pairs.insert_missing_self_pairs`, :func:`enerzyme.models.layers.self_pairs.unit_direction`, :func:`enerzyme.models.functional.reduce_edge`, and :code:`SimpleReadout` (``two_layer`` + ``pyramidal: true``, exact SiLU). TensorNet calls the same self-pair helper. The Core emits invariant :code:`atom_feature` and Cartesian :code:`atom_vector_feature` ``[N, 3, F]``, so :code:`EquivariantDipole` and :code:`EquivariantPolarizability` apply. ``vector_cutoff: true`` is the current MD17/QM9 example; ``false`` is the 2022 paper. QM9 widths are :code:`PUBLISHED_QM9_CORE_PARAMS`. Example: :code:`et_layers_example.yaml`. Tests: :code:`test/test_et_ops.py`, :code:`test/test_et_core.py`. Enerzymette: :code:`architecture: et` + resolved :code:`config.yaml`.

QuinNet (:code:`quinnet`)
^^^^^^^^^^^^^^^^^^^^^^^^^

Wang et al. (NeurIPS 2023) live under :code:`enerzyme/models/quinnet/` (:code:`interaction.py`). Register via :code:`get_ff_core("quinnet")`. Aliases are :code:`quin-net` and :code:`quin_net`. The pure PyTorch rules follow ``src/models/quinnet.py`` in `Zun-Wang/QuinNet <https://github.com/Zun-Wang/QuinNet>`_ (MIT). Do not import that package and do not expect its checkpoints to load. Reuse :code:`ExpNormalSmearing`, :code:`cosine` windows from :code:`RangeSeparation`, :code:`RandomAtomEmbedding`, :class:`~enerzyme.models.et.interaction.ETNeighborEmbedding`, :class:`~enerzyme.models.visnet.interaction.EdgeEmbedding`, :func:`enerzyme.models.mp.ops.self_pairs.insert_missing_self_pairs`, :func:`enerzyme.models.mp.basis.angular.polynomial_spherical_harmonics`, :func:`enerzyme.models.mp.ops.rejection.vector_rejection`, and :code:`SimpleReadout` (``two_layer`` + ``pyramidal: true``, exact SiLU). The angular state is the 8-channel ``lmax`` 2 prefix and starts at zero, so the six geometric channels enter on the second block. The public vector is the ``l = 1`` slice, so the Cartesian tensor heads apply. ``atom_l2_feature`` is the other five channels. The three edge invariants scatter onto the sender. The reference neighbor cap of 32 is not applied. Aspirin widths are :code:`PUBLISHED_MD17_CORE_PARAMS`. Example: :code:`quinnet_layers_example.yaml`. Tests: :code:`test/test_quinnet_ops.py`, :code:`test/test_quinnet_core.py`. Enerzymette: :code:`architecture: quinnet` + resolved :code:`config.yaml`.

ViSNet (:code:`visnet`)
^^^^^^^^^^^^^^^^^^^^^^^

Wang et al. (Nat. Commun. 2024, doi:10.1038/s41467-023-43720-2) live under :code:`enerzyme/models/visnet/` (:code:`interaction.py`). Register via :code:`get_ff_core("visnet")`. Aliases are :code:`vis-net` and :code:`vis_net`. The pure PyTorch rules follow ``visnet.models.visnet_block`` on `microsoft/AI2BMD <https://github.com/microsoft/AI2BMD/tree/ViSNet>`_ branch ``ViSNet`` (MIT). Do not import that package and do not expect its checkpoints to load. Reuse :code:`ExpNormalSmearing`, :code:`cosine` windows from :code:`RangeSeparation`, :code:`RandomAtomEmbedding`, :class:`~enerzyme.models.et.interaction.ETNeighborEmbedding`, :func:`enerzyme.models.mp.ops.self_pairs.insert_missing_self_pairs`, :func:`enerzyme.models.mp.basis.angular.polynomial_spherical_harmonics`, :func:`enerzyme.models.mp.ops.rejection.vector_rejection`, and :code:`SimpleReadout` (``two_layer`` + ``pyramidal: true``, exact SiLU). ``lmax`` 1 or 2 only; VecLayerNorm accepts 3 or 8 angular channels. The public vector is the ``l = 1`` slice, so the Cartesian tensor heads apply at ``lmax: 2``. ``atom_l2_feature`` is the other five channels. ``vertex_type: none`` is the published YAML null. The reference neighbor cap of 32 is not applied. MD17 widths are :code:`PUBLISHED_MD17_CORE_PARAMS`. QM9 is :code:`PUBLISHED_QM9_CORE_PARAMS` (``vecnorm_type: max_min``). :meth:`ViSNetCore.encode_state` is the pre-norm stack ViSNet-LSRM calls. Example: :code:`visnet_layers_example.yaml`. Tests: :code:`test/test_visnet_ops.py`, :code:`test/test_visnet_core.py`. Enerzymette: :code:`architecture: visnet` + resolved :code:`config.yaml`.

FreeCG (:code:`freecg`)
^^^^^^^^^^^^^^^^^^^^^^^

Shao, Geng, Wang, and Cui (arXiv:2407.02263) live under :code:`enerzyme/models/freecg/` (:code:`interaction.py`, :code:`cg.py`). Register via :code:`get_ff_core("freecg")`. Aliases are :code:`free-cg` and :code:`free_cg`. The :func:`~enerzyme.models.presets.cartesian_vector.freecg` preset shares the ViSNet embedding and builds its blocks with :class:`~enerzyme.models.freecg.interaction.FreeCGInteraction`. Do not import `ShihaoShao-GH/FreeCG <https://github.com/ShihaoShao-GH/FreeCG>`_; its checkpoints do not load. Reuse :code:`ExpNormalSmearing`, :code:`cosine` windows, :code:`RandomAtomEmbedding`, :class:`~enerzyme.models.et.interaction.ETNeighborEmbedding`, :func:`enerzyme.models.mp.basis.angular.polynomial_spherical_harmonics`, :func:`enerzyme.models.mp.ops.rejection.vector_rejection`, and :code:`SimpleReadout`. ``lmax`` is 2. :class:`~enerzyme.models.freecg.cg.GroupSparseCG` is the grouped four-path product and the channel roll. It does not own the embedding or the energy MLP, so a later model can call it on abstract edges of shape ``[N, 8, F]``. ``num_groups`` must divide ``dim_embedding``. ``shuffle_scale: 1.5`` is ``1.5 * F / G`` channels forward. ``attention_enhancer`` is ``linear``, ``max``, or ``none``. Table 7 widths are :code:`PUBLISHED_MD17_CORE_PARAMS` (8 groups). Example: :code:`freecg_layers_example.yaml`. Tests: :code:`test/test_freecg_ops.py`, :code:`test/test_freecg_core.py`. Enerzymette: :code:`architecture: freecg` + resolved :code:`config.yaml`.

ViSNet-LSRM (:code:`visnet_lsrm`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Li et al. (ICLR 2024) live under :code:`enerzyme/models/visnet/lsr.py` and :code:`long_range.py`. Register via :code:`get_ff_core("visnet_lsrm")`. Aliases are :code:`visnet-lsrm`, :code:`visnet_lsr`, and :code:`visnet-lsr`. The :func:`~enerzyme.models.presets.cartesian_vector.visnet_lsrm` preset extends the shared ViSNet short-range blocks with fragment interactions. Do not import `liyy2/LSR-MP <https://github.com/liyy2/LSR-MP>`_; its checkpoints do not load. Fragment ids and the cluster-count heuristic live in :mod:`enerzyme.models.layers.fragments`, which E2Former-LSR re-exports from :mod:`enerzyme.models.e2former.cluster`. The long-range block is :class:`~enerzyme.models.visnet.long_range.FragmentBipartiteBlock`. It does not own the species table, the short-range basis, or the energy MLP. The long-range basis is another :class:`~enerzyme.models.layers.rbf.ExpNormalSmearing` stored on the Core, because those distances are atom–fragment distances. ``dim_feature_out`` is ``2 * dim_embedding``. ``atom_l2_feature`` stays width ``dim_embedding`` when ``lmax`` is 2. MD22 widths are :code:`PUBLISHED_MD22_CORE_PARAMS`. Example: :code:`visnet_lsrm_layers_example.yaml`. Tests: :code:`test/test_visnet_lsrm_ops.py`, :code:`test/test_visnet_lsrm_core.py`. Enerzymette: :code:`architecture: visnet_lsrm` + resolved :code:`config.yaml`. Optional :code:`cluster_ids` is the same Datahub field as E2Former-LSR.

TensorNet (:code:`tensornet`) and TensorNet2 (:code:`tensornet2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simeon and De Fabritiis (NeurIPS 2023, arXiv:2306.06482) live under :code:`enerzyme/models/tensornet/` (:code:`interaction.py`, :code:`ops.py`). Register via :code:`get_ff_core("tensornet")`. TensorNet2 (Farr et al., arXiv:2601.00581) is :code:`tensornet2` in :code:`enerzyme.models.presets.cartesian_tensor` with the charge update in :code:`charge.py`. Register via :code:`get_ff_core("tensornet2")`. It reuses :meth:`TensorNetCore._prepare_graph`, the embedding, and :func:`tensor_message_update`. The charge head and neutral charge equilibration stay inside the Core because the next message reads them. :code:`WeightedCoulombEnergy` and :code:`StandardZBLRepulsionEnergy` are shared layers. The pure PyTorch rules follow `torchmd/torchmd-net <https://github.com/torchmd/torchmd-net>`_ (MIT). Do not import that package, do not port the Warp kernels, and do not expect its checkpoints to load. Reuse :code:`ExpNormalSmearing`, :code:`cosine` windows from :code:`RangeSeparation`, :code:`RandomAtomEmbedding`, :func:`enerzyme.models.functional.reduce_edge`, and :code:`SimpleReadout` (``two_layer`` + ``pyramidal: true``, exact SiLU). The basis already contains a cosine window; the Core multiplies :code:`cutoff_values_sr` again, which is the repository product when both windows are the Behler cosine at the same cutoff. Self pairs are inserted inside the Core by :func:`enerzyme.models.mp.ops.self_pairs.insert_missing_self_pairs` because the shared neighbor list omits them. The same helper serves the Equivariant Transformer. The Core emits invariant :code:`atom_feature` ``[N, F]``, skew :code:`atom_vector_feature` ``[N, 3, F]``, and :code:`atom_tensor_feature` ``[N, 3, 3, F]``. TensorNet2 also emits :code:`charge_channels`. ``Q`` is the optional molecular total charge; a length-``N`` value must be constant inside each molecule. ``WeightedCoulombEnergy`` weights are checked against ``(L+1) q`` in :meth:`TensorNet2Core.build`. A charge channel with all-zero weights is shifted uniformly. Examples: :code:`tensornet_layers_example.yaml`, :code:`tensornet2_layers_example.yaml`. Tests: :code:`test/test_tensornet_ops.py`, :code:`test/test_tensornet_core.py`, :code:`test/test_tensornet2_ops.py`, :code:`test/test_tensornet2_core.py`. Enerzymette: :code:`architecture: tensornet` or :code:`architecture: tensornet2` + resolved :code:`config.yaml`.

EquFlash (:code:`equflash`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SamsungDS GGNN EquFlash lives under :code:`enerzyme/models/equflash/` (:code:`interaction.py`, :code:`irreps.py`). Register via :code:`get_ff_core("equflash")`. The Core is the convolution stack only. Reuse :code:`RandomAtomEmbedding`, :code:`BesselRBF(flavor="equflash")` (``(2/c) sin(nπr/c)/r``, trainable frequencies by default), and :code:`RangeSeparation` with :code:`cutoff_fn: xplor` plus :code:`cuton`. :func:`equflash_irreps` builds the scalar → natural-parity → scalar schedule; :code:`irreps_manual` overrides it. Messages use :func:`tp_out_irreps_with_instructions` and an :code:`e3nn` gate. Scalar nonlinearities go through :func:`enerzyme.models.activation.resolve_scalar_activation`, shared with NequIP. :meth:`BaseFFCore.adopt_radius_stack` places the radius layers. The Core emits :code:`atom_feature` and :code:`feature_irreps` for :code:`SimpleReadout`. :code:`conv_denominator` rescales the scatter (default 1; GGNN's OMat yaml uses the dataset neighbor count). Cuequivariance / FlashTP are not wired. Example: :code:`equflash_layers_example.yaml`. Tests: :code:`test/test_equflash_core.py`. Enerzymette: :code:`architecture: equflash` + resolved :code:`config.yaml`.

So3krates (:code:`so3krates`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frank et al. (NeurIPS 2022) lives under :code:`enerzyme/models/so3krates/` with shared :code:`RealSphericalHarmonics` and :code:`L0Contraction` (:code:`cgmatrix.npz`) in :code:`enerzyme/models/mp/equivariant/`. The Core emits invariant :code:`atom_feature` and SPHC :code:`atom_sphere_feature` (``[N, m_tot]``, not eSCN/EquiformerV2 channel layout). Compose :code:`SimpleReadout` + :code:`EnergyReduce` / :code:`Force` (and optional ZBL / electrostatics / dispersion) outside the Core. Offline parity vs So3krates-torch fixtures: :code:`test/test_so3krates_parity_ops.py`. Enerzymette only needs :code:`architecture: so3krates` plus a resolved :code:`config.yaml`.

SO3LR (:code:`so3lr`)
^^^^^^^^^^^^^^^^^^^^^

Kabylda et al. (JACS 2025) is registered as :code:`architecture: so3lr` but **reuses** :code:`So3kratesCore`. Defaults and layers live in :code:`enerzyme/models/so3krates/so3lr.py`. Physics uses shared modules with SO3LR options: :code:`ZBLRepulsionEnergy` (:code:`switch_off`), :code:`ElectrostaticEnergy` (:code:`flavor: SO3LR`), :code:`TSQDODispersionEnergy` under :code:`enerzyme/models/layers/dispersion/`, plus :code:`SimpleReadout(Qa)` / :code:`AtomicAffine` / :code:`HirshfeldReadout` / :code:`ChargeSpinEmbedding`. Grimme D3/D4 remain for PhysNet/SpookyNet stacks. Cutoff alias :code:`phys` → polynomial. Tests: :code:`test/test_so3lr.py`. Enerzymette: :code:`architecture: so3lr` + resolved :code:`config.yaml` (see :code:`enerzyme/config/so3lr_layers_example.yaml`).

EFA (:code:`efa` / :code:`so3lr_efa`) and SpookyNet :code:`use_efa`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frank et al. (arXiv:2412.08541) — shared package :code:`enerzyme/models/efa/` (ERoPE + Lebedev linear attention). Lebedev point tables live in :code:`enerzyme/models/mp/equivariant/data/lebedev_grids.npz` (e3x Apache-2.0) and are imported from :code:`enerzyme.models.mp.equivariant.s2.lebedev` (also re-exported on the :code:`efa` package).

* Register :code:`efa` / :code:`so3lr_efa` in :code:`get_ff_core` (both use :code:`So3kratesCore` with :code:`era_use_in_iterations`).
* SpookyNet: Core / :code:`InteractionModule` flag :code:`use_efa` swaps :code:`NonlocalInteraction` for :code:`EFABlock` (pass :code:`Ra`).
* Adding EFA to a **new** Core: (1) include :code:`Ra` and :code:`batch_seg` in Core :code:`input_fields`; (2) :code:`build_efa_blocks(...)` or construct :code:`EFABlock`; (3) after a local layer, :code:`x = x + apply_efa_if_configured(x, Ra, batch_seg, block)`. Keep EFA inside the Core, not as a post-core YAML physics layer.
* Tests: :code:`test/test_efa.py`. Examples: :code:`efa_layers_example.yaml`, :code:`so3lr_efa_layers_example.yaml`.

BOTNet (:code:`botnet`)
^^^^^^^^^^^^^^^^^^^^^^^^^

Batatia et al. (Nat. Mach. Intell., doi:10.1038/s42256-024-00956-x, arXiv:2205.06643) live under :code:`enerzyme/models/botnet/` (:code:`interaction.py`, :code:`readout.py`). Register via :code:`get_ff_core("botnet")`. Aliases are :code:`body_ordered` and :code:`body-ordered`. Do not import `gncs/botnet <https://github.com/gncs/botnet>`_; its checkpoints do not load. The interaction classes are the ones already in :mod:`enerzyme.models.mace.interaction` (``AgnosticNonlinearInteractionBlock``, ``AgnosticResidualNonlinearInteractionBlock``, ``ResidualElementDependentInteractionBlock``). MACE's ``RealAgnostic*`` blocks return a product-basis tensor and are rejected. Reuse :code:`BesselRBF(flavor="mace")` with ``polynomial_p: 6``, :code:`RandomAtomEmbedding`, :meth:`BaseFFCore.adopt_radius_stack`, and :class:`~enerzyme.models.mace.blocks.feature_spec`. The symmetric contraction stays in MACE. The Core emits :code:`atom_feature`. :code:`BOTNetEnergyReadout` is the linear-per-layer head with a scalar MLP on the last layer. :code:`AtomicAffine`, :code:`EnergyReduce`, and :code:`Force` stay outside. Published widths are :code:`PUBLISHED_CORE_PARAMS` and :code:`PUBLISHED_3BPA_HIDDEN_IRREPS`. Example: :code:`botnet_layers_example.yaml`. Tests: :code:`test/test_botnet_ops.py`, :code:`test/test_botnet_core.py`. Enerzymette: :code:`architecture: botnet` + resolved :code:`config.yaml`.

AllScAIP (:code:`AllScAIP`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enerzyme's AllScAIP is an **experimental, modified (魔改)** attention Core under :code:`enerzyme/models/allscaip/`. Do **not** treat it as the recommended production model; document regressions and keep example FF entries inactive unless deliberately testing. The Core returns :code:`atom_feature` only — YAML stacks must include :code:`SimpleReadout` / NSE heads (see :code:`DEFAULT_LAYER_PARAMS`).

Flow matching (:code:`uma_flow_qs`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Requires optional :code:`torchdiffeq` (:code:`pip install -e ".[flow]"`) for ODE integration in :code:`enerzyme/tasks/generator_ode.py`. Keep ODE utilities in tasks/, not inside Core modules.
