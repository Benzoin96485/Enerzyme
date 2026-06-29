Active Learning Workflows
=========================

Two distinct mechanisms share the name "active learning" in Enerzyme documentation.

Decision guide
--------------

- **Enerzyme internal dataset AL** — All candidates already in one labeled dataset; resample via :code:`withheld` partition
- **Enerzymette-driven loop** — New geometries from MD/PLUMED; QM on fragments

Enerzyme internal dataset AL
----------------------------

Single command:

.. code-block:: bash

    enerzyme train -c active_learning_train.yaml -o al_out/

Key settings:

- :code:`Trainer.committee_size` > 1
- :code:`Trainer.active_learning_params.active: true`
- :code:`data_source: withheld`
- :code:`picking_method`: :code:`max_Fa_norm_std`, :code:`mean_Fa_std`, or :code:`random`
- :code:`picking_params` error bounds filter trivial or divergent samples
- :code:`Splitter` with small :code:`training` and large :code:`withheld`

Picking references:

- J. Chem. Inf. Model. **2024**, *64*, 6377-6387
- Comput. Phys. Commun. **2020**, *253*, 107206

Does **not** run simulate, extract, or annotate.

Enerzymette-driven external loop
--------------------------------

Managed by:

.. code-block:: bash

    enerzymette enerzyme_active_learning \
        -p parent_model_dir \
        -sc config/simulate.yaml \
        -ec config/extract.yaml \
        -ac config/annotate.yaml \
        -tc config/train.yaml \
        -n <iterations> -np <presim_steps> \
        -cp <calc_patch> -pp <plumed_plugin> \
        -rp <reference.pdb> -ts <template.sdf>

Per-iteration stages
^^^^^^^^^^^^^^^^^^^^

1. **MD** — :code:`FFxx_md/` (:code:`simulation.yaml`, :code:`plumed.traj.xyz`, :code:`plumed.traj.pkl`)
2. **Prediction** — uncertainty on trajectory frames
3. **Extraction** — :code:`FFxx_extraction/FFxx_fragments.sdf`
4. **Annotation** — QM on fragments → :code:`fragments.pkl`
5. **Training** — merge data, :code:`pretrain_path` from previous model, new :code:`suffix`
6. **Model** — :code:`FFxx/` checkpoints for next round

Task folder template
^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    task_root/
    |-- al.sh
    |-- cluster.xyz
    |-- cluster.mol
    |-- config/
    |   |-- simulate.yaml    # sampling policy; paths filled per round
    |   |-- extract.yaml     # data_path null in template
    |   |-- annotate.yaml    # Supplier.path empty in template
    |   `-- train.yaml       # architecture; paths/suffix filled per round
    |-- FF02-0 -> parent_pretrained_model
    |-- FF02-0_md/
    |-- FF02-0_prediction/
    |-- FF02-0_extraction/
    |-- FF02-0_fragments/
    |-- FF02-0_training/
    `-- FF02-1/

Template rewrite rules
^^^^^^^^^^^^^^^^^^^^^^

- :code:`simulate.yaml` — :code:`System.structure_file` → :code:`FFxx_md/initial_structure.xyz`
- :code:`extract.yaml` — :code:`Datahub.data_path` → trajectory pickle
- :code:`annotate.yaml` — :code:`Supplier.path` → fragment SDF
- :code:`train.yaml` — :code:`pretrain_path`, :code:`suffix`, dataset paths, UDD :code:`B` may update from validation error

Initial scans
^^^^^^^^^^^^^

Optional :code:`scan-*` directories from :code:`enerzymette enerzyme_scan` seed reaction coordinates before AL. Separate from both AL modes.

Failure recovery
^^^^^^^^^^^^^^^^

- Archive :code:`config.yaml` and checkpoints each iteration
- Resume training with :code:`Trainer.resume: 2` inside a round if needed
- Enerzymette :code:`-rm` policy controls cleanup of intermediate files

See also :doc:`/getting_started/active_learning` for a tutorial-style walkthrough.
