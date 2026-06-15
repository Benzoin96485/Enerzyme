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
- **Development** — :code:`requirements-dev.yaml` (conda env for day-to-day coding)
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
2. Ensure loader supports the source format (:code:`pickle`, :code:`npz`, or HDF5 cache)
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

:code:`FFSimulate` loads :code:`-cp` (calculator) and :code:`-pp` (PLUMED generator) via :code:`importlib` from user-supplied :code:`.py` files. Plugin modules must expose the factory functions expected by :code:`simulator.py` / Enerzymette — see :doc:`/user_guide/workflows/enhanced_sampling_plumed`.

Stable artifact paths
^^^^^^^^^^^^^^^^^^^^^

Avoid renaming without strong reason:

- :code:`out/config.yaml`
- :code:`out/processed_dataset_<hash>/`
- :code:`out/FF<id>-<arch>[-suffix]/best/` and :code:`last/`
- Simulation outputs such as :code:`md.traj.xyz`, :code:`plumed.traj.xyz`, :code:`neb.xyz`

Testing strategy
----------------

Test layout
^^^^^^^^^^^

Tests live in :code:`test/`:

- :code:`test_scatter.py` — :code:`torch_scatter` equivalence (CPU/GPU parametrized)
- :code:`test_spookynet.py` — SpookyNet layer and forward tests
- :code:`test_physnet.py` — PhysNet parity against reference TensorFlow implementation (heavy optional stack)
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
