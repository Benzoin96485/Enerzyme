# V100 eager performance

The Phase 0 reference is `1aa0f8b`. The main-branch comparison uses trained checkpoints from `b05ea1a`. Both measure energy and forces during inference, a backward pass through the force loss during training, and peak allocation for each. The three systems contain 32 molecules of 20 atoms, a 256-atom periodic cell, and a 2,000-atom cluster.

Each revised measurement uses the paired run for that case, with its own candidate revision. The JSON archives alongside this report contain the measured times, memory, exceptions, and run order.

| Reference | Within +3% | Above +3% | Unavailable | Total |
| --- | ---: | ---: | ---: | ---: |
| Phase 0 defaults | 1180 | 0 | 212 | 1392 |
| Main checkpoints | 96 | 0 | 0 | 96 |
| XequiNet xPaiNN | 24 | 0 | 0 | 24 |

The original xPaiNN wrapper omitted periodic image offsets. The periodic timing compares the two wrappers. Independent energy, force and weight-gradient checks use the upstream XequiNet model with the correct periodic offsets on CPU and V100.

48 of 48 main-checkpoint timing medians were as fast as or faster than main.

## Missing paired measurements

The Phase 0 model ran out of memory on the 32 GB V100 for 164 missing timing or memory comparisons.

A separate frozen-weight AllScAIP run (6317546) recorded the exception for each system. On the molecular batch and the 2,000-atom cluster, the old model modified a tensor that its force calculation needed. On the periodic system its graph builder received an empty cell argument. Both attention backends reproduced these failures.

The old E2Former-V2 raised a module-call error in FP32 and rejected the FP64 dtype. Neither old run returned timing or memory measurements.

Each model and precision has 12 possible comparisons. Individual exceptions are preserved in the baseline and candidate archives.

| Preset | float32 | float64 |
| --- | ---: | ---: |
| alignn | 6 | 8 |
| allscaip | 12 | 12 |
| alphanet | 0 | 2 |
| dimenet | 4 | 4 |
| e2former_lsr | 0 | 2 |
| e2former_v2 | 12 | 12 |
| equflash | 8 | 10 |
| equiformer_v3 | 0 | 2 |
| escn | 2 | 2 |
| freecg | 10 | 12 |
| gemnet | 0 | 2 |
| gemnet-oc | 12 | 12 |
| gemnet-q | 12 | 12 |
| geodite | 2 | 4 |
| leftnet | 0 | 2 |
| mace_polar | 4 | 4 |
| quinnet | 4 | 4 |
| spinconv | 4 | 4 |
| tece | 0 | 2 |
| visnet | 4 | 4 |
