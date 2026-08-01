# EquiformerV2 upstream fixture (MIT)

Vendored from https://github.com/atomicarchitects/equiformer_v2 for offline numerical
parity tests of Enerzyme's EquiformerV2 Core / SO(2) attention blocks.

- Pin: see `COMMIT_SHA` (main at vendor time)
- Source tree: `nets/equiformer_v2/`
- Local adaptations: `so3.py` renamed to `eqv2_so3.py` so pytest can load this
  fixture without colliding with `escn_upstream/so3.py` on `sys.path`.

Do **not** use this fixture as a production dependency. Refresh:

```bash
SHA=$(cat COMMIT_SHA)  # or update to a new commit
for f in activation.py drop.py input_block.py layer_norm.py radial_function.py so2_ops.py transformer_block.py so3.py wigner.py edge_rot_mat.py; do
  curl -sL -o "$f" "https://raw.githubusercontent.com/atomicarchitects/equiformer_v2/${SHA}/nets/equiformer_v2/$f"
done
```
