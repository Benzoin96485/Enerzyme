Running Simulations with a Trained Model
=========================================

Enerzyme wraps trained models as ASE calculators for geometry optimization, scans, molecular dynamics, and nudged elastic band (NEB) calculations. Reference configs live in :code:`enerzyme/config/`.

Basic command
-------------

.. code-block:: bash

    enerzyme simulate -c opt.yaml -o sim_out/ -m model_dir/ -mc train.yaml

Arguments:

- :code:`-c` — simulation YAML (:code:`Simulation` + :code:`System`)
- :code:`-m` — trained model directory
- :code:`-mc` — training config (defaults to :code:`model_dir/config.yaml`)
- :code:`-o` — output directory for trajectories and logs
- :code:`-cp` — optional external calculator patch module (hybrid potentials)
- :code:`-pp` — optional PLUMED CV plugin module (see :doc:`enhanced_sampling`)

Shared configuration blocks
---------------------------

System
^^^^^^

.. code-block:: yaml

    System:
        structure_file: "initial.xyz"
        charge: -1
        multiplicity: 1

:code:`structure_file` is an ASE-readable path. For NEB it may contain multiple frames (see below).

Simulation
^^^^^^^^^^

Common keys:

- :code:`task` — :code:`sp`, :code:`opt`, :code:`scan`, :code:`md`, :code:`neb`, :code:`plumed`, or :code:`plumed_scan`
- :code:`environment` — :code:`ase` (currently the only backend)
- :code:`neighbor_list` — must match training (:code:`full` or on-the-fly)
- :code:`idx_start_from` — :code:`1` for 1-based or :code:`0` for 0-based atom indices
- :code:`dtype` — :code:`float64` recommended for optimization
- :code:`cuda` — GPU inference when available
- :code:`constraint` — :code:`fix_atom`, :code:`Hookean_allpairs`, etc.

.. caution::
    **Units.** Model :code:`Hartree_in_E` and :code:`Bohr_in_R` from training define energy and force units in ASE. MD configs may set :code:`fs_in_t` and :code:`Hartree_in_E` under :code:`Simulation` for time-step conversion.

Single-point energy (:code:`task: sp`)
--------------------------------------

Computes energy (and available properties) for each frame in :code:`structure_file`. Output: :code:`sp.xyz`.

Geometry optimization (:code:`task: opt`)
-----------------------------------------

See :code:`enerzyme/config/opt.yaml`.

.. code-block:: yaml

    Simulation:
        task: opt
        optimize:
            optimizer: LBFGS
        constraint:
            fix_atom:
                indices: [1, 2, 3, 4, 5]

Outputs: :code:`traj-opt.xyz` (optimization path), :code:`optim.xyz` (final structure).

Distance scan (:code:`task: scan`)
----------------------------------

See :code:`enerzyme/config/scan.yaml`. Scans the distance between two atoms while optimizing all other degrees of freedom at each point.

.. code-block:: yaml

    Simulation:
        task: scan
        sampling:
            cv: distance
            params:
                i0: 100
                i1: 101
                x0: 3.812
                x1: 1.492
                num: 25

Outputs: :code:`scan_optim.xyz`, per-point :code:`traj-<i>.xyz`.

.. note::
    For chemistry-specific collective variables and PLUMED-based scans, use :code:`task: plumed_scan` with Enerzymette CV plugins (:doc:`enhanced_sampling`).

NVT molecular dynamics (:code:`task: md`)
-----------------------------------------

See :code:`enerzyme/config/nvt_md.yaml`.

.. code-block:: yaml

    Simulation:
        task: md
        fs_in_t: 1
        integrate:
            integrator: Langevin
            time_step: 0.5
            temperature_in_K: 300
            friction: 0.01
            n_step: 100000

Output trajectory: :code:`md.traj.xyz`. Optional :code:`initialize` block sets Maxwell–Boltzmann velocities.

NEB and CI-NEB (:code:`task: neb`)
----------------------------------

See :code:`enerzyme/config/neb.yaml`.

.. code-block:: yaml

    Simulation:
        task: neb
        sampling:
            params:
                num_images: 25
                spring_constants: 0.1
                climb: true
                interpolation:
                    method: idpp
                    apply_constraints: true

**Structure file formats** (multi-frame XYZ):

- **2 frames** — reactant and product; Enerzyme interpolates :code:`num_images` with IDPP
- **3 frames** — reactant, TS guess, product; two-stage interpolation
- **num_images frames** — use pre-built path as-is

By default endpoints are relaxed before interpolation (:code:`relax_endpoints: true`). Outputs include :code:`neb.xyz`, :code:`ci-neb.xyz` when climbing is enabled, and per-image XYZ files.

.. note::
    To build an initial path externally, Enerzymette provides :code:`enerzymette idpp` (see :doc:`bond_and_utilities`).

Supported optimizers
--------------------

Geometry and NEB: :code:`BFGS`, :code:`LBFGS`, :code:`MDMin`, :code:`FIRE`, :code:`GPMin`, and line-search variants. NEB-specific: :code:`odesolver`, :code:`static`.

Next steps
----------

- PLUMED biased MD and hybrid calculators: :doc:`enhanced_sampling`
- Uncertainty-driven sampling loops: :doc:`active_learning`
