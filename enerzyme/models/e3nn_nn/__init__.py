"""e3nn Irreps-stack helpers (Equiformer V1, MACE, related readouts).

Distinct from ``enerzyme.models.so3``, which owns SH-array / SPHC layouts.
Both packages may use e3nn as a math library; this package's *API* is flat
``Irreps`` tensors and e3nn tensor products.
"""

from .activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout
from .norm import EquivariantLayerNormV2
from .tensor_product import (
    FullyConnectedTensorProductRescale,
    FullyConnectedTensorProductRescaleSwishGate,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
    sort_irreps_even_first,
)
from .tools import (
    U_matrix_real,
    extract_scalar_0e,
    linear_out_irreps,
    reshape_irreps,
    scalar_0e_dim,
    tp_out_irreps_with_instructions,
)

__all__ = [
    "Activation",
    "EquivariantDropout",
    "EquivariantLayerNormV2",
    "EquivariantScalarsDropout",
    "FullyConnectedTensorProductRescale",
    "FullyConnectedTensorProductRescaleSwishGate",
    "Gate",
    "LinearRS",
    "TensorProductRescale",
    "U_matrix_real",
    "extract_scalar_0e",
    "irreps2gate",
    "linear_out_irreps",
    "reshape_irreps",
    "scalar_0e_dim",
    "sort_irreps_even_first",
    "tp_out_irreps_with_instructions",
]
