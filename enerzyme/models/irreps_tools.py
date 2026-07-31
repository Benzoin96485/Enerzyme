"""Backward-compatible re-exports; prefer ``enerzyme.models.e3nn_nn``. """

from .e3nn_nn.tools import (  # noqa: F401
    U_matrix_real,
    extract_scalar_0e,
    linear_out_irreps,
    reshape_irreps,
    scalar_0e_dim,
    tp_out_irreps_with_instructions,
)
