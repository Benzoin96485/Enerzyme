"""Nequix Core.

Koker and Smidt, arXiv:2508.16067. A short stack of separable NequIP
convolutions with a species-independent residual, equivariant RMSNorm, and
a scalar final layer. One-hot species, the Bessel basis, the polynomial
cutoff, per-element energy shifts, and forces stay outside this module.

Reference code: https://github.com/atomicarchitects/nequix (MIT). The
published ``nequix-mp-1`` widths live in :data:`PUBLISHED_MP_CORE_PARAMS`.
The defaults below are a small stack with the same block, so importing the
architecture does not compile the 700k-parameter tensor products.
"""

from __future__ import annotations

from typing import Dict, List

from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..e3nn_nn.tools import scalar_0e_dim
from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from .interaction import NequixConvolution
from .species import MPTRJ_ATOMIC_NUMBERS


# Table A.1 and configs/nequix-mp-1.yml. avg_num_neighbors is the MPtrj mean.
PUBLISHED_MP_CORE_PARAMS = {
    "num_layers": 4,
    "lmax": 3,
    "hidden_irreps": "128x0e + 64x1o + 32x2e + 32x3o",
    "radial_mlp_size": 64,
    "radial_mlp_layers": 2,
    "mlp_init_scale": 4.0,
    "avg_num_neighbors": 57.413687022442645,
    "layer_norm": True,
}
PUBLISHED_MP_SCALE = 0.8066479563713074
PUBLISHED_MP_SHIFT = 0.16502578765761478


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": len(MPTRJ_ATOMIC_NUMBERS),
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "nequip",
    "atomic_numbers": list(MPTRJ_ATOMIC_NUMBERS),
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "equflash",
            "trainable": False,
            "apply_cutoff_fn": True,
        },
    },
    {"name": "OneHotAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 2,
            "hidden_irreps": "8x0e + 4x1o + 4x2e",
            "radial_mlp_size": 32,
            "radial_mlp_layers": 2,
            "mlp_init_scale": 4.0,
            "avg_num_neighbors": 1.0,
            "layer_norm": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "order": "scale_then_shift",
            "scales": {"Ea": {"values": 1.0, "learnable": False}},
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class NequixCore(BaseFFCore):
    """Map a scalar embedding and a radial basis to scalar ``atom_feature``.

    Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
    ``vij_sr = R_j - R_i``, the same orientation as the reference
    ``positions[senders] - positions[receivers]``. The last block keeps only
    even scalars. ``dim_feature_out`` is that channel count.
    """

    def __str__(self) -> str:
        return f"""
################################################################################
# Nequix — separable NequIP conv, species-free residual, RMSNorm, scalar out  #
# lmax={self.lmax} layers={self.num_layers} hidden={self.hidden_irreps} -> {self.feature_irreps}
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 89,
        num_rbf: int = 8,
        num_layers: int = 2,
        lmax: int = 2,
        hidden_irreps: str = "8x0e + 4x1o + 4x2e",
        radial_mlp_size: int = 32,
        radial_mlp_layers: int = 2,
        mlp_init_scale: float = 4.0,
        avg_num_neighbors: float = 1.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
            },
            output_fields={"atom_feature"},
        )
        if dim_embedding < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        if radial_mlp_size < 1:
            raise ValueError(f"radial_mlp_size must be >= 1, got {radial_mlp_size}")
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.num_layers = int(num_layers)
        self.lmax = int(lmax)
        self.hidden_irreps = o3.Irreps(hidden_irreps).regroup()
        if len(self.hidden_irreps) == 0:
            raise ValueError("hidden_irreps is empty")
        if self.hidden_irreps.lmax > self.lmax:
            raise ValueError(
                f"hidden irreps reach l={self.hidden_irreps.lmax}, "
                f"but the edge spherical harmonics stop at lmax={self.lmax}"
            )
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)
        self.sph = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )
        scalar = o3.Irreps(f"{self.dim_embedding}x0e")
        scalar_out = _keep_even_scalars(self.hidden_irreps)
        blocks = []
        for layer in range(self.num_layers):
            irreps_in = scalar if layer == 0 else self.hidden_irreps
            irreps_out = (
                self.hidden_irreps if layer < self.num_layers - 1 else scalar_out
            )
            blocks.append(
                NequixConvolution(
                    irreps_in,
                    irreps_out,
                    self.irreps_sh,
                    num_rbf=self.num_rbf,
                    radial_mlp_size=int(radial_mlp_size),
                    radial_mlp_layers=int(radial_mlp_layers),
                    mlp_init_scale=float(mlp_init_scale),
                    avg_num_neighbors=float(avg_num_neighbors),
                    layer_norm=bool(layer_norm),
                )
            )
        self.layers = ModuleList(blocks)
        self.feature_irreps = str(self.layers[-1].irreps_out)
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)

    def build(self, built_layers: List[Module]) -> None:
        self.calculate_distance = DistanceLayer()
        self.calculate_distance.with_vector_on("vij_lr")
        self.calculate_distance.reset_field_name(Dij="Dij_lr")
        self.pre_sequence.append(self.calculate_distance)

        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                if isinstance(layer, DistanceLayer):
                    continue
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, (BaseAtomEmbedding, BaseRBF)):
                    pass
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must be [n_edges, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        if idx_i_sr.shape != idx_j_sr.shape or vij_sr.shape[:1] != idx_i_sr.shape:
            raise ValueError("idx_i_sr, idx_j_sr, and vij_sr must share the edge axis")
        if vij_sr.numel():
            edge_sh = self.sph(vij_sr)
        else:
            edge_sh = vij_sr.new_zeros((0, self.irreps_sh.dim))
        features = atom_embedding
        for layer in self.layers:
            features = layer(features, idx_i_sr, idx_j_sr, rbf, edge_sh)
        return {"atom_feature": features}


def _keep_even_scalars(irreps: o3.Irreps) -> o3.Irreps:
    kept = [(mul, ir) for mul, ir in irreps if ir.l == 0 and ir.p == 1]
    if not kept:
        raise ValueError(f"{irreps} has no even scalars for the final Nequix layer")
    return o3.Irreps(kept)
