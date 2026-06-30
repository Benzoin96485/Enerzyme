Active Learning for Neural Network Potentials
==============================================

Enerzyme and Enerzymette use the phrase **active learning** for two related but different workflows:

- **Enerzyme internal dataset active learning** is a single :code:`enerzyme train` job. It starts from a labeled dataset, splits part of it into a :code:`withheld` pool, trains a committee, picks uncertain labeled structures from that pool, and retrains.
- **Enerzymette-driven active learning** is an outer workflow manager. It repeatedly runs :code:`simulate`, :code:`predict` / :code:`extract`, :code:`annotate`, and :code:`train`, creating new structures and new QM labels between model updates.

Do not mix the two concepts. The first is useful when all candidate structures are already labeled or at least already stored in one dataset. The second is the production loop for enzymatic reaction exploration, where each round creates new geometries by MD or scans, extracts uncertain fragments, runs QM, and trains the next model.

Enerzyme internal dataset AL
----------------------------

This mode is configured inside :code:`Trainer.active_learning_params` and launched with :code:`enerzyme train`. The reference config is :code:`enerzyme/config/active_learning_train.yaml`.

What it does
^^^^^^^^^^^^

1. Split one prepared dataset into :code:`training` and :code:`withheld`
2. Train a committee (:code:`committee_size` > 1)
3. Predict uncertainty on the :code:`withheld` partition
4. Pick structures by :code:`picking_method`
5. Move picked structures into training and continue

The key limitation is that the loop stays **inside the dataset**. It does not run MD, PLUMED, QM annotation, or fragment extraction by itself.

Configuration sketch
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    Trainer:
        committee_size: 4
        active_learning_params:
            active: true
            data_source: withheld
            picking_method: max_Fa_norm_std
            picking_params:
                relative_bound_on_validation: true
                relative_error_lower_bound: 0.5
                relative_error_upper_bound: 5
                error_lower_bound: 0.01
                error_upper_bound: 0.05
            sample_size: 10000
            max_epoch_per_iter: 10000
            max_iter: 100
            resume: true
        Splitter:
            method: random
            parts:
                - training
                - withheld
            ratios:
                - 0.01
                - 0.99

Run it with:

.. code-block:: bash

    enerzyme train -c active_learning_train.yaml -o al_results/

Key parameters
^^^^^^^^^^^^^^

:code:`committee_size`
    Number of independently initialized models trained in the committee.

:code:`data_source`
    Pool for picking; usually the Splitter partition named :code:`withheld`.

:code:`picking_method`
    :code:`random`, :code:`mean_Fa_std`, or :code:`max_Fa_norm_std`.

:code:`picking_params`
    Error thresholds used to avoid both uninformative low-error samples and out-of-domain high-error samples.

:code:`resume`
    Resume from the internal AL checkpoint :code:`al_ckp.data`.

Enerzymette-driven AL loop
--------------------------

Enerzymette's :code:`enerzyme_active_learning` command manages an **external** loop around Enerzyme. A typical round is:

.. code-block:: text

    current model
        -> simulate / presimulate with Enerzyme
        -> convert trajectory to candidate dataset
        -> predict uncertainty
        -> extract uncertain fragments
        -> annotate fragments with QM
        -> merge training / validation pickles
        -> train next model from previous checkpoint

This workflow is the right choice when active learning should expand chemical space, not merely resample an existing dataset.

Launcher command
^^^^^^^^^^^^^^^^

A representative launcher looks like:

.. code-block:: bash

    enerzymette enerzyme_active_learning \
        -p /path/to/pretrained_or_parent_model_dir \
        -cp uma \
        -pp sammt \
        -sc config/simulate.yaml \
        -ac config/annotate.yaml \
        -ec config/extract.yaml \
        -tc config/train.yaml \
        -n 20 \
        -np 1000 \
        -rp /path/to/reference_cluster.pdb \
        -ts /path/to/template_ligands.sdf \
        -rm hard

Important inputs:

- :code:`-p` — parent model directory or previous-stage model root. Iteration 0 links or copies from this source.
- :code:`-cp` — external calculator patch key, for example :code:`uma`.
- :code:`-pp` — PLUMED CV plugin key, for example :code:`sammt`.
- :code:`-sc` — simulation template for steered MD or biased MD.
- :code:`-ac` — QM annotation template.
- :code:`-ec` — fragment extraction template.
- :code:`-tc` — training template.
- :code:`-n` — number of AL iterations.
- :code:`-np` — presimulation MD steps before the steered / biased production segment.
- :code:`-rp` — reference PDB used by the PLUMED CV plugin and chemistry-specific atom selection.
- :code:`-ts` — template SDF used for bond orders / ligand connectivity.
- :code:`-rm` — cleanup or run-management policy.

Task folder layout
^^^^^^^^^^^^^^^^^^

A practical task directory should contain only stable, human-authored inputs at the top level. Generated files are written into numbered iteration folders.

.. code-block:: text

    my_al_task/
    |-- al.sh
    |-- cluster.xyz
    |-- cluster.mol
    |-- config/
    |   |-- simulate.yaml
    |   |-- annotate.yaml
    |   |-- extract.yaml
    |   `-- train.yaml
    |-- FF02-SpookyNet-0 -> /path/to/parent/FF02-SpookyNet
    |-- FF02-SpookyNet-0_md/
    |-- FF02-SpookyNet-0_prediction/
    |-- FF02-SpookyNet-0_extraction/
    |-- FF02-SpookyNet-0_fragments/
    |-- FF02-SpookyNet-0_training/
    |-- FF02-SpookyNet-1/
    `-- ...

The directory :code:`/home/gridsan/wlluo/multiscale/MLFF/enerzyme-new/spookynet/L3-COMT_2ZVJ-frag6-1` follows this pattern:

- :code:`al.sh` is the batch launcher.
- :code:`config/` stores the four templates passed to :code:`-sc`, :code:`-ac`, :code:`-ec`, and :code:`-tc`.
- :code:`cluster.xyz` is the initial structure for simulation.
- :code:`cluster.mol` is the reference molecule for fragment extraction.
- :code:`FF02-SpookyNet-<i>_md/` stores per-iteration MD input/output.
- :code:`FF02-SpookyNet-<i>_prediction/` stores the prediction config/artifacts for the MD trajectory.
- :code:`FF02-SpookyNet-<i>_extraction/` stores extraction config and the fragment SDF.
- :code:`FF02-SpookyNet-<i>_fragments/` stores QM fragment calculations and labeled fragments.
- :code:`FF02-SpookyNet-<i>_training/` stores generated training / validation sets and the resolved training config.
- :code:`FF02-SpookyNet-<i>/` stores the trained model for the next iteration.

Simulation template
^^^^^^^^^^^^^^^^^^^

The simulation template describes the **sampling policy**. Enerzymette rewrites paths such as :code:`System.structure_file` each round.

.. code-block:: yaml

    Simulation:
        task: plumed
        environment: ase
        external_calculator:
            name: uma_calculator
            weight: 1.0
        internal_calculator_weight: 0.0
        uncertainty_calculator:
            name: UDD
            params:
                A: 0.08
                B: 0.0001
        plumed_config_generator:
            name: SAMMTConfigGenerator
            method: standard_steered_md
        sampling:
            params:
                plumed_config:
                    lower_bound: -2
                    upper_bound: 2
                    reference_pdb_file: /path/to/reference_cluster.pdb
                    substrate: KOM
                    nucleophile: O9
        integrate:
            integrator: Langevin
            temperature_in_K: 500
            time_step: 0.5
            n_step: 10000
        constraint:
            fix_atom:
                indices: [80, 81, 82]
    System:
        structure_file: cluster.xyz
        charge: -1
        multiplicity: 1

In generated iteration folders this becomes, for example, :code:`FF02-SpookyNet-19_md/simulation.yaml`, where :code:`System.structure_file` points to :code:`FF02-SpookyNet-19_md/initial_structure.xyz`. The MD output trajectory is converted into a prediction/extraction dataset such as :code:`FF02-SpookyNet-19_md/plumed.traj.pkl`.

Extraction template
^^^^^^^^^^^^^^^^^^^

The extraction template should describe **how to pick fragments**, not a fixed candidate file. Leave :code:`Datahub.data_path` empty in the template; Enerzymette fills it with the current MD trajectory pickle.

.. code-block:: yaml

    Datahub:
        data_format: pickle
        data_path: null
        features:
            N:
            Q: Q
            Ra: Ra
            Za: Za
        neighbor_list: full
        preload: true
        targets: null
    Extractor:
        reference_mol_path: /path/to/cluster.mol
        fragment_per_frame: 6
        local_uncertainty_radius: 5
        fragment_radius: 4
        n_centers: 2
    Trainer:
        inference_batch_size: 4

In a generated round this becomes :code:`FF02-SpookyNet-19_extraction/extraction.yaml`, and it points to :code:`FF02-SpookyNet-19_md/plumed.traj.pkl`. The main output is an SDF such as :code:`FF02-SpookyNet-19_extraction/FF02-SpookyNet-19_fragments.sdf`.

Annotation template
^^^^^^^^^^^^^^^^^^^

The annotation template should define the QM method. Leave :code:`Supplier.path` empty; Enerzymette fills it with the fragment SDF for the current round.

.. code-block:: yaml

    Supplier:
        path:
    QMDriver:
        engine: TeraChem
        bs: 6-31gs
        xc: b3lyp
        pcm: cosmo
        dftd: d3
        pcm_radii_file: /path/to/pcm_radii
        epsilon: 10
        keep_molden: false
        keep_output: false
        clean_tmp: true
        pickle_name: fragments.pkl
        dump_single_run: false
        n_processes: 8

In a generated round :code:`Supplier.path` points to the extracted SDF. The labeled :code:`fragments.pkl` is then merged into the next training set.

Training template
^^^^^^^^^^^^^^^^^

The training template defines the architecture, losses, transforms, and optimizer. Enerzymette rewrites the dataset paths, suffix, and pretraining path for each round.

.. code-block:: yaml

    Datahub:
        datasets:
            training:
                data_path: training_set.pkl
                features:
                    Ra: coord
                    Za: atom_type
                    Q: total_chrg
                targets:
                    E: energy
                    Fa: grad
                    M2: dipole
                    Q: total_chrg
                transforms:
                    atomic_energy: /path/to/atomic_energy.csv
                    negative_gradient: true
            validation:
                data_path: validation_set.pkl
                # same mappings
    Modelhub:
        internal_FFs:
            FF02:
                architecture: SpookyNet
                active: true
                pretrain_path:
                suffix:
                layers:
                - name: ShallowEnsembleReduce
                  params:
                    var: [E, Fa]
                    eval_only: true
    Trainer:
        Splitter:
            method: random
            parts:
            - name: training
              dataset: training
            - name: validation
              dataset: validation
        resume: 0

In generated configs, round :code:`i` sets :code:`pretrain_path` to round :code:`i-1` model, writes :code:`suffix: '<i>'`, and fills absolute paths such as :code:`FF02-SpookyNet-19_training/training_set.pkl`.

Optional initial scans
^^^^^^^^^^^^^^^^^^^^^^

Enerzymette scan launchers can prepare reaction-coordinate structures before the MD loop. In the reference task, :code:`scan-10/`, :code:`scan-15/`, and :code:`scan-20/` are separate scan folders. Each contains:

- :code:`launch.sh` for :code:`enerzymette enerzyme_scan`
- :code:`scan.csv` with elementary reaction labels such as :code:`1a -> 2a`, :code:`1b -> 2b`
- One directory per elementary reaction, e.g. :code:`1a-2a/`
- Generated :code:`reactant_opt.yaml`, :code:`scan.yaml`, :code:`product_opt.yaml`
- :code:`local_minima/` and :code:`rate_determining_ts/` summaries

These scans are useful for seeding structures or checking reaction coordinates, but they are separate from Enerzyme's internal :code:`Trainer.active_learning_params`.

When to use which mode
----------------------

Use **Enerzyme internal dataset AL** when:

- all candidate structures are already in one dataset;
- you want a compact :code:`enerzyme train`-only experiment;
- no new QM calculations are required during the loop.

Use **Enerzymette-driven AL** when:

- new geometries must be generated by MD, PLUMED, scans, or presimulation;
- uncertain regions should be extracted into fragments;
- QM labels are produced on the fly;
- the next model should be trained from the previous model checkpoint.

.. note::
    Both modes need uncertainty-aware models. In the Enerzymette loop, uncertainty is usually exposed by shallow-ensemble layers (for example :code:`ShallowEnsembleReduce` with :code:`var: [E, Fa]`) and consumed by prediction/extraction.

References
----------

- J. Chem. Inf. Model. **2024**, *64*, 6377-6387 (mean force std / local uncertainty)
- Comput. Phys. Commun. **2020**, *253*, 107206 (max force norm std picking)
