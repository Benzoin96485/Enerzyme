"""Unified message-passing toolkit.

Every Core is assembled from the pieces in this package: ``ops`` (scatter,
softmax, safe norms), ``basis`` (radial / angular bases and envelopes), ``nn``
(MLPs, norms, attention), ``equivariant`` (layouts, rotations, tensor products,
SO(2) convolutions, S2 grids, Cartesian tensors, body-order engines), ``graph``
(``GraphContext`` / ``GraphRequires``), and ``core`` (``ComposableCore``).
"""
