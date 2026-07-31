So3krates-torch parity fixtures
================================

Vendored (MIT) snippets from https://github.com/TCPUniLU/So3krates-torch for
numerical parity tests against Enerzyme ``so3`` / ``so3krates`` modules.

- ``spherical_harmonics.py`` — RealSphericalHarmonics
- ``so3_conv_invariants.py`` — L0Contraction (+ local ``cgmatrix.npz``)
- ``upstream_blocks.py`` — FilterNet, EuclideanAttentionBlock, InteractionBlock
  (scatter via ``torch_scatter``; cutoff broadcasting fixed to ``[:, None, None]``)

No JAX / mlff dependency.
