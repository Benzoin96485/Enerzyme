Configuration System
====================

Enerzyme workflows are composed from YAML sections. Different commands read different subsets; some sections override training defaults at inference time.

Section index
-------------

+-------------------+------------------------------------------+---------------------------+
| Section           | Used by                                  | Reference config          |
+===================+==========================================+===========================+
| :code:`Datahub`   | train, predict, extract, collect         | :code:`train.yaml`        |
+-------------------+------------------------------------------+---------------------------+
| :code:`Modelhub`  | train; loaded from model dir at infer    | :code:`train.yaml`        |
+-------------------+------------------------------------------+---------------------------+
| :code:`Trainer`   | train; partial use in predict/extract    | :code:`train.yaml`        |
+-------------------+------------------------------------------+---------------------------+
| :code:`Metric`    | train, predict                           | :code:`predict.yaml`      |
+-------------------+------------------------------------------+---------------------------+
| :code:`Simulation`| simulate                                 | :code:`opt.yaml`, etc.    |
+-------------------+------------------------------------------+---------------------------+
| :code:`System`    | simulate                                 | all sim YAMLs             |
+-------------------+------------------------------------------+---------------------------+
| :code:`Extractor` | extract                                  | :code:`extract.yaml`      |
+-------------------+------------------------------------------+---------------------------+
| :code:`Supplier`  | annotate                                 | :code:`annotate.yaml`     |
+-------------------+------------------------------------------+---------------------------+
| :code:`QMDriver`  | annotate                                 | :code:`annotate.yaml`     |
+-------------------+------------------------------------------+---------------------------+

Training artifacts
------------------

After :code:`enerzyme train -c train.yaml -o out/`:

- :code:`out/config.yaml` — resolved configuration (use as :code:`-mc` for predict/simulate)
- :code:`out/FFxx/` or :code:`out/FFxx_suffix/` — model checkpoints (:code:`best/`, :code:`last/`)
- :code:`out/processed_dataset_<hash>/` — preprocessed HDF5 cache
- :code:`out/logs/` — training logs

Override rules at inference
---------------------------

**Predict** (:code:`enerzyme predict -c pred.yaml -m model_dir -mc train.yaml`):

- :code:`-c` overrides :code:`Datahub` (test path, features) and :code:`Metric`
- :code:`-mc` supplies :code:`Modelhub` and default transforms/neighbor list consistency
- Neighbor list and transform settings should match training unless you know the effect

**Simulate** (:code:`enerzyme simulate -c sim.yaml -m model_dir -mc train.yaml`):

- :code:`-c` supplies :code:`Simulation` and :code:`System`
- :code:`-mc` supplies model architecture and :code:`Datahub` transforms for the ASE calculator
- Optional :code:`-cp` (calculator patch) and :code:`-pp` (PLUMED patch) add external plugins

**Extract** — combines predict-style :code:`Datahub` with :code:`Extractor` radii and reference molecule.

**Annotate** — :code:`Supplier` + :code:`QMDriver` only; independent of Modelhub.

Legacy vs current schema
------------------------

Older configs use a flat :code:`Datahub` with :code:`compression:`; current configs use :code:`compressed:` and optional :code:`datasets:` for multi-dataset training. Both layouts are still accepted; new projects should follow :code:`enerzyme/config/train.yaml`.

Splitter configs similarly support:

- **Old:** :code:`parts: [training, validation]` + :code:`ratios: [...]`
- **New:** :code:`parts: [{name, dataset, ratio}, ...]` for multi-dataset splits

See :doc:`/user_guide/data/splitting`.
