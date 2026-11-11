"""EquFlashV2 Core.

Channelwise Clebsch-Gordan message passing with equivariant RMSNorm and a
bilinear (or gated) node update. Species embedding, the Bessel / XPLOR
radial window, species-wise energy calibration, and force heads stay outside
the Core.

The block layout follows Samsung Electronics' EquFlashV2
(https://github.com/SamsungDS/GGNN, CC BY-NC-SA 4.0). This module is an
Enerzyme reimplementation on e3nn. It does not depend on cuequivariance or
FlashTP. Published OMat / OAM checkpoints are not loaded here.

``conv_type`` ``full`` and ``efficient`` share one channelwise ``uvu``
convolution. Selecting ``efficient`` warns, because this port has no fused
kernel. The cuequivariance kernels differ in layout and fusion, not in the
Enerzyme data flow.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

from e3nn import o3
from torch import Tensor
from torch.nn import Identity, Module, ModuleList

from ..e3nn_nn import scalar_0e_dim
from ..layers import BaseFFCore
from .interaction_v2 import INTERACTION_TYPES, FeedForwardBlock
from .irreps import equflash_schedule, parse_schedule, spherical_harmonics_irreps


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cuton": 5.5,
    "cutoff_fn": "xplor",
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
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 2,
            "is_parity": True,
            "interaction_type": "equflash",
            "conv_type": "full",
            "sc_type": "identity",
            "nonlinear_type": "bilinear_gate",
            "nonlinear_scale": 1.0,
            "after_nonlinear": "linear",
            "feedforward_type": "feedforward",
            "ff_nonlinear_type": "bilinear_gate",
            "ff_nonlinear_scale": 1.0,
            "ff_sc_type": "identity",
            "radial_hidden": [32, 32],
            "denominator": 1.0,
            "train_denominator": False,
            "normalize_sph": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activate_hidden": False,
            "use_bias": False,
            "keep_feature": False,
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


class EquFlashV2Core(BaseFFCore):
    """Map a scalar atom embedding and a radial basis to ``atom_feature``.

    Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
    ``vij_sr = R_j - R_i``. The last layer's flat irreps are ``atom_feature``;
    ``feature_irreps`` describes them and ``dim_feature_out`` counts even
    scalars for :class:`~enerzyme.models.layers.readout.SimpleReadout`.
    """

    def __str__(self) -> str:
        return f"""
################################################################################
# EquFlashV2 — channelwise CG convolution, RMSNorm, bilinear gate             #
# lmax={self.lmax} layers={self.num_layers} parity={self.is_parity} -> {self.feature_irreps}
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 32,
        num_rbf: int = 8,
        num_layers: int = 2,
        lmax: int = 2,
        is_parity: bool = True,
        irreps_manual: Optional[Sequence[str]] = None,
        interaction_type: str = "equflash",
        conv_type: str = "full",
        sc_type: str = "identity",
        nonlinear_type: str = "bilinear_gate",
        nonlinear_scale: float = 1.0,
        nonlinear_kwargs: Optional[dict] = None,
        after_nonlinear: str = "linear",
        feedforward_type: str = "feedforward",
        ff_nonlinear_type: str = "bilinear_gate",
        ff_nonlinear_scale: float = 1.0,
        ff_nonlinear_kwargs: Optional[dict] = None,
        ff_sc_type: str = "identity",
        radial_hidden: Optional[Sequence[int]] = None,
        denominator: float = 1.0,
        train_denominator: bool = False,
        normalize_sph: bool = True,
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
        if interaction_type not in INTERACTION_TYPES:
            raise ValueError(
                f"Unknown interaction_type {interaction_type!r}; "
                f"expected {sorted(INTERACTION_TYPES)}"
            )
        if conv_type not in ("full", "efficient"):
            raise ValueError(
                f"Unknown conv_type {conv_type!r}; expected 'full' or 'efficient'"
            )
        if conv_type == "efficient":
            warnings.warn(
                "EquFlashV2 conv_type='efficient' runs the same e3nn "
                "channelwise convolution as 'full'. This port has no separate "
                "fused kernel.",
                UserWarning,
                stacklevel=2,
            )
        if feedforward_type not in ("feedforward", "identity"):
            raise ValueError(
                f"Unknown feedforward_type {feedforward_type!r}; "
                "expected 'feedforward' or 'identity'"
            )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.lmax = int(lmax)
        self.is_parity = bool(is_parity)
        self.interaction_type = interaction_type
        self.conv_type = conv_type
        self.normalize_sph = bool(normalize_sph)
        if irreps_manual:
            schedule = parse_schedule(irreps_manual)
        else:
            schedule = equflash_schedule(
                self.dim_embedding, self.lmax, int(num_layers), self.is_parity
            )
        scalar = o3.Irreps(f"{self.dim_embedding}x0e")
        if schedule[0] != scalar:
            raise ValueError(
                f"EquFlashV2 starts from the scalar embedding {scalar}, "
                f"got {schedule[0]}. Change dim_embedding or irreps_manual[0]."
            )
        self.schedule = schedule
        self.num_layers = len(schedule) - 1
        edge_l = max([self.lmax] + [block.lmax for block in schedule])
        self.irreps_filter = spherical_harmonics_irreps(edge_l, self.is_parity)
        self.sph = o3.SphericalHarmonics(
            self.irreps_filter,
            normalize=self.normalize_sph,
            normalization="component",
        )
        hidden = [32, 32] if radial_hidden is None else list(radial_hidden)
        interaction_cls = INTERACTION_TYPES[interaction_type]
        interactions = []
        feedforwards = []
        for layer in range(self.num_layers):
            irreps_in = schedule[layer]
            irreps_out = schedule[layer + 1]
            interactions.append(
                interaction_cls(
                    irreps_in,
                    self.irreps_filter,
                    irreps_out,
                    num_rbf=self.num_rbf,
                    radial_hidden=hidden,
                    denominator=denominator,
                    train_denominator=train_denominator,
                    sc_type=sc_type,
                    nonlinear_type=nonlinear_type,
                    nonlinear_scale=nonlinear_scale,
                    nonlinear_kwargs=nonlinear_kwargs,
                    after_nonlinear=after_nonlinear,
                )
            )
            if feedforward_type == "identity":
                feedforwards.append(Identity())
            else:
                feedforwards.append(
                    FeedForwardBlock(
                        irreps_out,
                        irreps_out,
                        sc_type=ff_sc_type,
                        nonlinear_type=ff_nonlinear_type,
                        nonlinear_scale=ff_nonlinear_scale,
                        nonlinear_kwargs=ff_nonlinear_kwargs,
                    )
                )
        self.interactions = ModuleList(interactions)
        self.feedforwards = ModuleList(feedforwards)
        self.feature_irreps = str(schedule[-1])
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)

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
        features = atom_embedding
        edge_sh = self.sph(vij_sr) if vij_sr.numel() else vij_sr.new_zeros(
            (0, self.irreps_filter.dim)
        )
        for interaction, feedforward in zip(self.interactions, self.feedforwards):
            features = interaction(features, idx_i_sr, idx_j_sr, rbf, edge_sh)
            features = feedforward(features)
        return {"atom_feature": features}
