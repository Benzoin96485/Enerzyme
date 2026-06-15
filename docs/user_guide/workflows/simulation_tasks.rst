Simulation Tasks
================

:code:`enerzyme simulate` dispatches ASE workflows from :code:`Simulation.task` in :code:`enerzyme/tasks/simulator.py`.

Command
-------

.. code-block:: bash

    enerzyme simulate -c sim.yaml -o sim_out/ -m model_dir/ -mc config.yaml \
        [-cp calculator_patch.py] [-pp plumed_patch.py]

Task matrix
-----------

- :code:`sp` — (inline config) → :code:`sp.xyz`
- :code:`opt` — :code:`opt.yaml` → :code:`traj-opt.xyz`, :code:`optim.xyz`
- :code:`scan` — :code:`scan.yaml` → :code:`scan_optim.xyz`, :code:`traj-*`
- :code:`md` — :code:`nvt_md.yaml` → :code:`md.traj.xyz`
- :code:`neb` — :code:`neb.yaml` → :code:`neb.xyz`, :code:`ci-neb.xyz`
- :code:`plumed` — :code:`plumed.yaml` → :code:`plumed.traj.xyz`
- :code:`plumed_scan` — (plugin) → :code:`scan_optim.xyz`

Shared Simulation keys
----------------------

- :code:`environment: ase`
- :code:`dtype` — :code:`float64` recommended for optimization
- :code:`cuda`
- :code:`neighbor_list` — match training
- :code:`idx_start_from` — 1 (1-based) or 0 (0-based) for constraints and scans
- :code:`Hartree_in_E`, :code:`fs_in_t` — unit conversion for integrators

Constraints
-----------

.. code-block:: yaml

    constraint:
        fix_atom:
            indices: [80, 81, 82]
        Hookean_allpairs:
            indices: [10, 11, 12]
            k: 10.0

Optimizers
----------

:code:`opt`, :code:`scan`, :code:`plumed_scan`: :code:`BFGS`, :code:`LBFGS`, :code:`FIRE`, :code:`MDMin`, :code:`GPMin`, line-search variants.

NEB optimizers: :code:`odesolver`, :code:`static` (ASE NEBOptimizer).

Integrators
-----------

:code:`md` / :code:`plumed`: currently :code:`Langevin` with :code:`time_step`, :code:`temperature_in_K`, :code:`friction`, :code:`n_step`.

NEB inputs
----------

:code:`System.structure_file` as multi-frame XYZ:

- 2 frames — reactant + product (interpolate with IDPP)
- 3 frames — reactant + TS guess + product
- :code:`num_images` frames — use path as-is

Options: :code:`relax_endpoints`, :code:`climb`, :code:`spring_constants`, :code:`interpolation.method: idpp`.

Calculator integration
----------------------

The trained model is wrapped as :code:`ASECalculator` (:code:`enerzyme/tasks/calculator.py`). Hybrid and UDD setups use :code:`external_calculator` and :code:`uncertainty_calculator` — see :doc:`/user_guide/workflows/enhanced_sampling_plumed`.
