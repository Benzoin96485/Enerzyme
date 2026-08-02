"""e3nn Irreps-stack helpers (Equiformer V1, MACE, TACE, related readouts).

Distinct from ``enerzyme.models.so3``, which owns SH-array / SPHC layouts.
Both packages may use e3nn as a math library; this package's *API* is flat
``Irreps`` tensors and e3nn tensor products.
"""

from .activation import Activation, Gate, get_gated_nonlinear
from .drop import EquivariantDropout, EquivariantScalarsDropout
from .linear import (
    ElementIrrepsLinear,
    IrrepsLinear,
    SkipIdentity,
    get_resnet_layer,
)
from .norm import EquivariantLayerNormV2
from .tensor_product import (
    FullyConnectedTensorProductRescale,
    FullyConnectedTensorProductRescaleSwishGate,
    LinearRS,
    O3ScatterTensorProduct,
    TensorProductRescale,
    UUUTensorProduct,
    irreps2gate,
    sort_irreps_even_first,
)
from .tools import (
    U_matrix_real,
    extract_scalar_0e,
    generate_paths,
    linear_out_irreps,
    reshape_irreps,
    satisfy,
    scalar_0e_dim,
    to_possible_tp_irreps,
    tp_out_irreps_with_instructions,
)

__all__ = [
    "Activation",
    "ElementIrrepsLinear",
    "EquivariantDropout",
    "EquivariantLayerNormV2",
    "EquivariantScalarsDropout",
    "FullyConnectedTensorProductRescale",
    "FullyConnectedTensorProductRescaleSwishGate",
    "Gate",
    "IrrepsLinear",
    "LinearRS",
    "O3ScatterTensorProduct",
    "SkipIdentity",
    "TensorProductRescale",
    "UUUTensorProduct",
    "U_matrix_real",
    "extract_scalar_0e",
    "generate_paths",
    "get_gated_nonlinear",
    "get_resnet_layer",
    "irreps2gate",
    "linear_out_irreps",
    "reshape_irreps",
    "satisfy",
    "scalar_0e_dim",
    "sort_irreps_even_first",
    "to_possible_tp_irreps",
    "tp_out_irreps_with_instructions",
]
