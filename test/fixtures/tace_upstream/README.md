# TACE upstream fixture (MIT)

Vendored from https://github.com/xvzemin/tace tag `v0.1.0` for offline numerical
parity of Enerzyme's Cartesian TACE backend (`cartnn`).

- Pin: see `COMMIT_SHA`
- Tree: `cartnn/` (self-contained Cartesian-3j helpers)

Spherical TACE ops are checked against an e3nn FullyConnectedTensorProduct
reference in `test_tace_parity_ops.py` (Enerzyme helpers live in `e3nn_nn`).

Do **not** use this fixture as a production dependency.
