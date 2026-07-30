# Equiformer upstream fixture (MIT)

Vendored from https://github.com/atomicarchitects/equiformer for offline numerical parity tests.

- Pin commit: see `COMMIT_SHA`
- Branch: `master` at vendoring time
- Local patches for offline CI (documented, minimal):
  - Optional `ocpmodels` / Bessel import stubbed (parity uses `exp` RBF only)
  - `torch_cluster.radius_graph` may be provided by `test/equiformer_parity_utils.py` if the package is missing
  - `NodeEmbeddingNetwork` one-hot cast uses LinearRS weight dtype (float64-safe)

Refresh (enerzyme-dev + gh):

```bash
SHA=$(gh api repos/atomicarchitects/equiformer/commits/master --jq .sha)
# re-fetch nets/* via gh api as in the Equiformer parity plan, then re-apply local patches
```
