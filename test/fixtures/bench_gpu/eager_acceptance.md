# V100 eager performance

The Phase 0 reference is `1aa0f8b`. The main-branch comparison uses trained checkpoints from `b05ea1a`. Both measure energy and forces during inference, a backward pass through the force loss during training, and peak allocation for each. The three systems contain 32 molecules of 20 atoms, a 256-atom periodic cell, and a 2,000-atom cluster.

Each revised measurement uses the paired run for that case, with its own candidate revision. The JSON archives alongside this report contain the measured times, memory, exceptions, and run order.

| Reference | Within +3% | Above +3% | Unavailable | Total |
| --- | ---: | ---: | ---: | ---: |
| Phase 0 defaults | 1180 | 0 | 212 | 1392 |
| Main checkpoints | 96 | 0 | 0 | 96 |

Of the 48 directly comparable main-checkpoint timings, 43 were as fast or faster. The following medians were slower.

- physnet_ff01 float32, bulk_256, infer_ms +2.80% (run 6317461)
- physnet_ff01 float32, bulk_256, train_ms +0.26% (run 6317461)
- physnet_ff01 float32, mol_batch_32x20, train_ms +1.00% (run 6317461)
- physnet_ff01 float64, bulk_256, infer_ms +0.41% (run 6317461)
- spookynet_ff02 float64, bulk_256, infer_ms +2.18% (run 6317461)

## Missing paired measurements

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
