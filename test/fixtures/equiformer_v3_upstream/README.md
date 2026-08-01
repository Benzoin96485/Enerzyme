# EquiformerV3 upstream fixture (MIT)

Vendored from https://github.com/atomicarchitects/equiformer_v3 for offline numerical
parity tests of Enerzyme's EquiformerV3 Core / SO(2) attention blocks.

- Pin: see `COMMIT_SHA` (main at vendor time)
- Source tree: `experimental/models/equiformer_v3/`
- Local adaptations: `so3.py` renamed to `eqv3_so3.py` so pytest can load this
  fixture without colliding with `escn_upstream/so3.py` / `equiformer_v2_upstream`
  on `sys.path`.

Do **not** use this fixture as a production dependency. Refresh:

```bash
SHA=$(cat COMMIT_SHA)  # or update to a new commit
for f in activation.py drop.py input_block.py layer_norm.py radial_function.py \
  so2_ops.py transformer_block.py so3.py wigner.py edge_rot_mat.py \
  envelope.py softmax.py utils.py; do
  curl -sL -o "$f" \
    "https://raw.githubusercontent.com/atomicarchitects/equiformer_v3/${SHA}/experimental/models/equiformer_v3/$f"
done
mv so3.py eqv3_so3.py
# Fix relative imports that pointed at .so3
sed -i 's/from \.so3 import/from .eqv3_so3 import/g' activation.py transformer_block.py
```
