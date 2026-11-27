"""Representation metadata shared by message-passing presets and readouts."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from e3nn.o3 import Irreps


class FeatureLayout(str, Enum):
    SCALAR = "scalar"
    CART_VECTOR = "cartesian_n3f"
    CART_RANK2 = "cartesian_rank2"
    CART_POLYADIC = "cartesian_polyadic"
    IRREPS_FLAT = "mul_ir"
    SPH_ARRAY = "spherical_array"
    S2_GRID = "s2_grid"


@dataclass(frozen=True)
class IrrepsSpec:
    """The emitted atom_feature and its invariant width available to scalar readouts."""

    irreps: Optional[str]
    scalar_width: int
    layout: FeatureLayout
    num_taps: int = 1

    @classmethod
    def atom_feature(cls, width, irreps=None, num_taps=1):
        irreps = str(irreps) if irreps is not None else None
        scalar_only = irreps is None or all(ir.l == 0 and ir.p == 1 for _, ir in Irreps(irreps))
        layout = FeatureLayout.SCALAR if scalar_only else FeatureLayout.IRREPS_FLAT
        return cls(irreps, int(width), layout, int(num_taps))

    @classmethod
    def from_layers(cls, layers, default_width=None):
        """Find the latent before a property readout, including standalone YAML layers."""
        for layer in reversed(layers):
            spec = getattr(layer, "feature_spec", None)
            if isinstance(spec, cls):
                return spec
            fields = getattr(layer, "_output_fields", set())
            width = getattr(layer, "dim_feature_out", None)
            if width and "atom_feature" in fields and not {"Ea", "Qa"}.intersection(fields):
                return cls.atom_feature(width, getattr(layer, "feature_irreps", None),
                                        getattr(layer, "num_output_blocks", 1))
        # External stacks can provide a feature width without declaring output fields.
        width = getattr(layers[-1], "dim_feature_out", default_width) if layers else default_width
        if width is None:
            raise ValueError("dim_embedding or dim_feature_out from the last layer must be provided")
        irreps = next((layer.feature_irreps for layer in reversed(layers)
                       if getattr(layer, "feature_irreps", None) is not None), None)
        return cls.atom_feature(width, irreps)
