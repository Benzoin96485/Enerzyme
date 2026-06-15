Layer Stack
===========

Enerzyme builds models as ordered **layers** rather than monolithic classes. The :code:`architecture` name selects a :code:`Core`; pre- and post-core layers handle geometry, embeddings, physics, and output reductions.

Registered layers
-----------------

From :code:`enerzyme/models/layers/`:

**Geometry**
    :code:`DistanceLayer`, :code:`RangeSeparationLayer`, :code:`RadiusGraphLayer`

**Radial basis**
    :code:`GaussianSmearing`, :code:`ExponentialGaussianRBFLayer`, :code:`ExponentialBernsteinRBFLayer`, :code:`BesselRBFLayer`, :code:`BernsteinRBFLayer`, :code:`SincRBFLayer`

**Embeddings**
    :code:`RandomAtomEmbedding`, :code:`NuclearEmbedding`, :code:`ElectronicEmbedding`, :code:`ScalarDenseEmbedding`, :code:`GatherAtomEmbedding`

**Core**
    Architecture-specific message passing (:code:`Core` with :code:`architecture` in Modelhub)

**Physics / post-processing**
    :code:`AtomicAffine`, :code:`ChargeConservation`, :code:`ElectrostaticEnergy`, :code:`AtomicCharge2Dipole`, :code:`GrimmeD3Energy`, :code:`GrimmeD4Energy`, :code:`ZBLRepulsionEnergy`

**Output**
    :code:`EnergyReduce`, :code:`Force`, :code:`ShallowEnsembleReduce`

Typical charge-aware stack
--------------------------

.. code-block:: yaml

    layers:
      - name: RangeSeparation
      - name: ExponentialBernsteinRBF
      - name: NuclearEmbedding
      - name: ElectronicEmbedding
        params:
            attribute: charge
      - name: Core
        params:
            num_modules: 6
            shallow_ensemble_size: 10
      - name: AtomicAffine
      - name: ChargeConservation
      - name: ElectrostaticEnergy
        params:
            flavor: SpookyNet
            dielectric_constant: 10.0
      - name: AtomicCharge2Dipole
      - name: EnergyReduce
      - name: ShallowEnsembleReduce
        params:
            var: [E]
            train_only: true
      - name: Force
      - name: ShallowEnsembleReduce
        params:
            var: [E, Fa]
            eval_only: true

Layer ordering matters
----------------------

1. Build geometric features (range separation, RBFs)
2. Embed atoms and optional scalar features
3. Message passing (:code:`Core`)
4. Normalize atomic outputs (:code:`AtomicAffine`)
5. Enforce physics (charge conservation, electrostatics, dispersion)
6. Reduce to molecular properties (:code:`EnergyReduce`)
7. Optional ensemble statistics
8. Analytic forces via autograd (:code:`Force`)

Shared :code:`build_params`
---------------------------

Common keys in :code:`build_params`:

- :code:`cutoff_sr`, :code:`cutoff_lr`, :code:`cutoff_fn`
- :code:`dim_embedding`, :code:`num_rbf`, :code:`max_Za`
- :code:`Hartree_in_E`, :code:`Bohr_in_R`

Layers inherit these unless :code:`params` overrides them.

Monitoring energy terms
-----------------------

Optional :code:`Trainer.Monitor` lists terms such as :code:`E_ele` (electrostatic), :code:`E_disp` (D3/D4), :code:`E_zbl` for debugging layer contributions during training.
