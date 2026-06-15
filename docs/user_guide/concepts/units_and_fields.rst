Units and Standard Fields
=========================

Enerzyme maps dataset attributes to **standard field names** so the same model code works across QM packages and naming conventions. Units must stay consistent across Datahub, Modelhub, and Simulation.

Standard fields
---------------

Defined in :code:`enerzyme/data/datatype.py`:

- :code:`N` (global) — Number of atoms per frame
- :code:`Ra` (atomic) — Cartesian coordinates
- :code:`Za` (atomic) — Atomic numbers
- :code:`E` (global) — Total energy
- :code:`Fa` (atomic) — Atomic forces (not raw gradients)
- :code:`Qa` (atomic) — Atomic partial charges
- :code:`Q` (global) — Total charge
- :code:`M2` (global) — Dipole moment (3-vector)
- :code:`S` (global) — Spin multiplicity (spin-aware models)

Custom fields can be registered under :code:`Datahub.fields` with :code:`is_atomic: true` (see :doc:`/user_guide/data/datahub_reference`).

Energy and length units
-----------------------

Model :code:`build_params` define conversion to internal atomic units:

- :code:`Hartree_in_E` — numeric value of one Hartree in your energy unit
- :code:`Bohr_in_R` — numeric value of one Bohr in your coordinate unit

Examples:

- Ha and Angstrom: :code:`Hartree_in_E: 1`, :code:`Bohr_in_R: 0.5291772108`
- eV and Angstrom: :code:`Hartree_in_E: 27.211386`, :code:`Bohr_in_R: 0.5291772108`

Simulation configs may repeat :code:`Hartree_in_E` under :code:`Simulation` for ASE integrators. PLUMED requires a matching :code:`UNITS` line in :code:`plumed.dat` (see :doc:`/user_guide/workflows/enhanced_sampling_plumed`).

Forces vs gradients
-------------------

Quantum chemistry outputs are often **energy gradients** :math:`\nabla E`, while forces are :math:`F = -\nabla E`. Set:

.. code-block:: yaml

    transforms:
        negative_gradient: true

when :code:`Fa` targets in the dataset are raw QC gradients. TeraChem parsers (e.g. :code:`scripts/picklizer.py`) may already convert gradient units from Ha/Bohr to Ha/Angstrom; still apply the sign flip if needed.

Loss and metric weights
-----------------------

Loss and :code:`Metric` weights are **not** dimensionless. Force terms are typically scaled to match energy units per length unit. A common pattern for Ha and Angstrom:

.. code-block:: yaml

    loss:
        rmse:
            E: 1
            Fa: 52.917721
            M2: 1.8897261

The force weight approximates :math:`\mathrm{Ha}/\mathrm{Bohr}` expressed in your target force unit (e.g. eV/Angstrom). Keep loss and validation :code:`Metric` weights aligned so early stopping reflects the same scale.

Time units in MD
----------------

:code:`Simulation.fs_in_t` sets how many femtoseconds correspond to one unit of :code:`integrate.time_step` and :code:`friction`. Default :code:`1` means :code:`time_step` is in fs.

Uncertainty fields
------------------

Committee or shallow-ensemble models may expose :code:`E_var`, :code:`Fa_var`, :code:`Qa_var`. Request them via :code:`Trainer.non_target_features` in predict configs. See :doc:`/user_guide/models/loss_metrics_uncertainty`.
