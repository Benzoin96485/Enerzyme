"""EQNorm Core.

Chen, Fu, Yang, and Chen, https://github.com/yzchen08/eqnorm (MIT). A
dual-branch E(3) network: one branch keeps high-order tensor features, and
the other accumulates even scalars read out of that tensor state. The
scalar branch is the latent ``atom_feature``. One-hot species, the Bessel
basis, the polynomial window, the energy MLP, per-element shifts, and
``Fa = -dE/dR`` stay outside this module.

The checked-in widths are a small stack. Published MPtrj / OMat / max
settings are :data:`PUBLISHED_MPTRJ_CORE_PARAMS`,
:data:`PUBLISHED_OMAT_CORE_PARAMS`, and :data:`PUBLISHED_MAX_CORE_PARAMS`.
Reference checkpoints are not loaded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..cutoff import polynomial_envelope
from ..layers import BaseFFCore
from ..nequip.interaction import tp_path_exists
from .interaction import EqnormInteraction
from .layernorm import one_irrep_per_degree


# eqnorm/model_settings/eqnorm-mptrj.yaml. avg_num_neighbors is MPtrj at 6 Å.
PUBLISHED_MPTRJ_CORE_PARAMS = {
    "num_layers": 4,
    "lmax": 3,
    "hidden_irreps": "128x0e+64x1o+32x2e+32x3o",
    "num_features": 128,
    "invariant_layers": 2,
    "invariant_neurons": 64,
    "avg_num_neighbors": 61.84,
    "normalization": "constant",
    "message_envelope": False,
    "polynomial_p": 6,
}
# eqnorm-omat.yaml: per-node degree normalization, no dataset neighbor mean.
PUBLISHED_OMAT_CORE_PARAMS = {
    **PUBLISHED_MPTRJ_CORE_PARAMS,
    "avg_num_neighbors": None,
    "normalization": "degree",
}
# eqnorm-max-mptrj.yaml: the same MPtrj stack with the envelope on the message.
PUBLISHED_MAX_CORE_PARAMS = {
    **PUBLISHED_MPTRJ_CORE_PARAMS,
    "message_envelope": True,
}


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 4,
    "num_rbf": 8,
    "max_Za": 8,
    "cutoff_sr": 6.0,
    "cutoff_fn": "nequip",
    "atomic_numbers": [1, 6, 7, 8],
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "equflash",
            "trainable": True,
            "apply_cutoff_fn": True,
        },
    },
    {"name": "OneHotAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 1,
            "hidden_irreps": "8x0e+4x1o",
            "num_features": 8,
            "invariant_layers": 2,
            "invariant_neurons": 16,
            "avg_num_neighbors": PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"],
            "normalization": "constant",
            "message_envelope": False,
            "polynomial_p": 6,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "variance_preserving",
            "use_bias": False,
            "nonlinearity": "silu",
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


class EqnormCore(BaseFFCore):
    """Map a scalar embedding and a radial basis to scalar ``atom_feature``.

    The species linear is the start of both branches: the tensor state and
    the scalar accumulator begin as the same ``{num_features}x0e`` vector.
    Each layer adds ``SiLU`` of the scalar convolution onto the accumulator
    and replaces the tensor state with the gated tensor convolution. The
    energy MLP reads the accumulator, so it stays in :class:`SimpleReadout`.
    ``readout_hidden_width`` is ``num_features // 2``, the published head's
    hidden width.
    """

    def __str__(self) -> str:
        return f"""
################################################################################
# EQNorm — tensor branch + scalar accumulator (yzchen08/eqnorm)               #
# {self.num_layers} layers, lmax={self.lmax}, hidden {self.hidden_irreps}, scalar {self.feature_irreps}
# normalization={self.normalization}, message_envelope={self.message_envelope}
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 4,
        num_rbf: int = 8,
        cutoff_sr: float = 6.0,
        num_layers: int = 2,
        lmax: int = 1,
        hidden_irreps: str = "8x0e+4x1o",
        num_features: int = 8,
        invariant_layers: int = 2,
        invariant_neurons: int = 16,
        avg_num_neighbors: Optional[float] = PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"],
        normalization: str = "constant",
        message_envelope: bool = False,
        polynomial_p: int = 6,
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
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if int(lmax) < 0:
            raise ValueError(f"lmax must be >= 0, got {lmax}")
        if int(num_features) < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if int(polynomial_p) < 2:
            raise ValueError(f"polynomial_p must be >= 2, got {polynomial_p}")
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_layers = int(num_layers)
        self.lmax = int(lmax)
        self.num_features = int(num_features)
        self.normalization = str(normalization)
        self.message_envelope = bool(message_envelope)
        self.polynomial_p = int(polynomial_p)
        self.avg_num_neighbors = (
            None if avg_num_neighbors is None else float(avg_num_neighbors)
        )
        if self.normalization == "constant" and (
            self.avg_num_neighbors is None or self.avg_num_neighbors <= 0
        ):
            raise ValueError(
                "constant neighbor normalization needs a positive "
                f"avg_num_neighbors, got {avg_num_neighbors}"
            )
        if self.normalization not in ("constant", "degree"):
            raise ValueError(
                "normalization must be 'constant' or 'degree', "
                f"got {normalization!r}"
            )
        self.hidden_irreps = one_irrep_per_degree(hidden_irreps)
        if self.hidden_irreps.lmax > self.lmax:
            raise ValueError(
                f"hidden irreps reach l={self.hidden_irreps.lmax}, "
                f"but spherical harmonics stop at lmax={self.lmax}"
            )
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)
        scalar_in = o3.Irreps(f"{self.num_features}x0e")
        missing = [
            f"{mul}x{ir}"
            for mul, ir in self.hidden_irreps
            if not tp_path_exists(scalar_in, self.irreps_sh, ir)
        ]
        if missing:
            raise ValueError(
                "hidden irreps "
                f"{missing} are not reachable from {scalar_in} and "
                f"spherical harmonics up to lmax={self.lmax}"
            )
        self.sph = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )
        self.species_linear = o3.Linear(
            o3.Irreps(f"{self.dim_embedding}x0e"),
            scalar_in,
            biases=False,
        )
        blocks = []
        irreps_in = scalar_in
        for _ in range(self.num_layers):
            block = EqnormInteraction(
                irreps_in,
                self.hidden_irreps,
                self.irreps_sh,
                scalar_in,
                num_rbf=self.num_rbf,
                invariant_layers=int(invariant_layers),
                invariant_neurons=int(invariant_neurons),
                normalization=self.normalization,
                avg_num_neighbors=self.avg_num_neighbors,
            )
            blocks.append(block)
            irreps_in = block.irreps_tensor
        self.layers = ModuleList(blocks)
        self.feature_irreps = str(scalar_in)
        self.dim_feature_out = self.num_features
        self.readout_hidden_width = max(self.num_features // 2, 1)

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

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
        if rbf.shape[:1] != idx_i_sr.shape:
            raise ValueError(
                f"rbf has {rbf.shape[0]} edges but idx_i_sr has {idx_i_sr.shape[0]}"
            )
        if vij_sr.numel():
            edge_sh = self.sph(vij_sr)
        else:
            edge_sh = vij_sr.new_zeros((0, self.irreps_sh.dim))
        envelope = None
        if self.message_envelope:
            if vij_sr.numel() == 0:
                distance = vij_sr.new_zeros((0,))
            else:
                distance = torch.linalg.vector_norm(vij_sr, dim=-1)
            envelope = polynomial_envelope(distance, self.cutoff_sr, self.polynomial_p)
        tensor = self.species_linear(atom_embedding)
        scalar = tensor
        for layer in self.layers:
            tensor, delta = layer(
                tensor, idx_i_sr, idx_j_sr, rbf, edge_sh, envelope
            )
            scalar = scalar + F.silu(delta)
        return {"atom_feature": scalar}
