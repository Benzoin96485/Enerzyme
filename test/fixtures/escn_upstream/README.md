# eSCN upstream fixture (MIT)

Vendored from https://github.com/facebookresearch/fairchem for offline numerical
parity tests of Enerzyme's native paper eSCN / SO(2)–SO(3) primitives.

- Pin tag: `fairchem_core-1.10.0` (see `COMMIT_SHA`)
- Source tree: `src/fairchem/core/models/escn/`
- Local adaptations for offline CI:
  - `escn_blocks.py` extracts `SO2Conv` / `SO2Block` / `EdgeBlock` / `MessageBlock` /
    `LayerBlock` without `fairchem.core` registry / GraphModelMixin imports
  - `EdgeBlock` accepts a precomputed `distance_features` tensor (parity harness)
    instead of importing fairchem SCN smearing modules
  - `so3.py` loads `Jd.pt` from this directory (same bytes as Enerzyme `models/so3/Jd.pt`)
  - dtype-safe casts for Wigner / grid / `to_m` when running float64 parity tests
  - absolute `from so3 import …` (fixture dir on `sys.path`)

Do **not** use this fixture as a production dependency. Refresh:

```bash
TAG=fairchem_core-1.10.0
curl -sL -o so3.py "https://raw.githubusercontent.com/facebookresearch/fairchem/${TAG}/src/fairchem/core/models/escn/so3.py"
curl -sL -o Jd.pt "https://raw.githubusercontent.com/facebookresearch/fairchem/${TAG}/src/fairchem/core/models/escn/Jd.pt"
# re-sync escn_blocks.py from escn.py Message/SO2/Layer/Edge blocks
```
