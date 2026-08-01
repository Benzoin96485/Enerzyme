"""So3krates Core (Frank et al., NeurIPS 2022).

Message-passing Core: dual-stream invariant features ``x`` and SPHC ``χ``.
Emits ``atom_feature`` (``x``) and ``atom_sphere_feature`` (``χ``, shape
``[N, m_tot]`` — not EquiformerV2's ``[N, (lmax+1)^2, C]``). Embedding / RBF
stay in pre-core layers; energy / force / charge / physics heads stay in
post-core layers.

Optional Euclidean Fast Attention (EFA) on the invariant stream via
``era_use_in_iterations`` (Frank et al., arXiv:2412.08541). Architectures
``efa`` / ``so3lr_efa`` enable this by default.

Algorithm follows So3krates-torch EuclideanTransformer
(https://github.com/TCPUniLU/So3krates-torch, MIT), matching mlff FeatureBlock
+ GeometricBlock + InteractionBlock.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Identity, Module, ModuleList, SiLU

from ..cutoff import CUTOFF_REGISTER, CUTOFF_KEY_TYPE
from ..efa import apply_efa_if_configured, build_efa_blocks, parse_era_iterations
from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from ..so3 import RealSphericalHarmonics
from .interaction import EuclideanTransformer

# Typical MD17 / small-molecule average degree (shared ballpark with Equiformer).
_AVG_NUM_NEIGHBORS = 15.57930850982666

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 132,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BernsteinRBF",
        "params": {"cutoff_fn": "cosine"},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "degrees": [1, 2, 3],
            "num_features": 132,
            "num_heads": 4,
            "num_layers": 3,
            "message_normalization": "avg_num_neighbors",
            "initialize_ev_to_zeros": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class So3kratesCore(BaseFFCore):
    """So3krates equivariant attention Core (invariant + SPHC streams)."""

    def __str__(self) -> str:
        return """
#################################################################################
# So3krates Core (Frank et al., NeurIPS 2022)                                  #
# Dual-stream Euclidean transformer; atom_feature = invariants x                #
# Optional EFA nonlocal on invariant stream (era_use_in_iterations)             #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        degrees: Optional[Sequence[int]] = None,
        num_features: int = 132,
        num_heads: int = 4,
        num_layers: int = 3,
        cutoff_sr: float = 5.0,
        cutoff_fn: CUTOFF_KEY_TYPE = "cosine",
        message_normalization: str = "avg_num_neighbors",
        avg_num_neighbors: Optional[float] = None,
        initialize_ev_to_zeros: bool = True,
        interaction_bias: bool = True,
        layer_normalization_1: bool = False,
        layer_normalization_2: bool = False,
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        qk_non_linearity: str = "identity",
        # EFA (Euclidean Fast Attention) — optional nonlocal on invariants
        era_use_in_iterations: Optional[object] = None,
        era_lebedev_num: int = 146,
        era_max_frequency: Optional[float] = None,
        era_max_length: float = 10.0,
        era_qk_num_features: Optional[int] = None,
        era_v_num_features: Optional[int] = None,
        # Accept build_params alias: dim_embedding may equal num_features when
        # RandomAtomEmbedding width matches Core feature dim.
        **kwargs,
    ) -> None:
        era_iters = parse_era_iterations(era_use_in_iterations)
        self.era_enabled = era_iters is not None
        input_fields = {
            "atom_embedding",
            "rbf",
            "idx_i_sr",
            "idx_j_sr",
            "vij_sr",
            "Dij_sr",
        }
        if self.era_enabled:
            input_fields |= {"Ra", "batch_seg"}
        super().__init__(
            input_fields=input_fields,
            output_fields={"atom_feature", "atom_sphere_feature"},
        )
        del kwargs  # absorb unused build_params (e.g. max_Za passed twice)
        degrees = list(degrees) if degrees is not None else [1, 2, 3]
        if num_features % 4 != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by 4")
        if num_features % num_heads != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by num_heads ({num_heads})"
            )
        if num_features % len(degrees) != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by len(degrees) ({len(degrees)})"
            )
        if dim_embedding != num_features:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must equal num_features ({num_features}); "
                "use RandomAtomEmbedding with matching width"
            )

        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.degrees = degrees
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cutoff_sr = cutoff_sr
        self.cutoff_fn_name = cutoff_fn
        self.message_normalization = message_normalization
        self.initialize_ev_to_zeros = initialize_ev_to_zeros
        self.m_tot = sum(2 * l + 1 for l in degrees)

        self.dim_feature_out = num_features
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.avg_num_neighbors = (
            _AVG_NUM_NEIGHBORS if avg_num_neighbors is None else float(avg_num_neighbors)
        )
        self._cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        self.spherical_harmonics = RealSphericalHarmonics(degrees=degrees)

        if qk_non_linearity.lower() in {"identity", "none", "linear"}:
            qk_act = Identity
        elif qk_non_linearity.lower() == "silu":
            qk_act = SiLU
        else:
            raise ValueError(f"Unsupported qk_non_linearity: {qk_non_linearity}")

        self.layers = ModuleList(
            [
                EuclideanTransformer(
                    degrees=degrees,
                    num_heads=num_heads,
                    num_features=num_features,
                    num_radial_basis_fn=num_rbf,
                    activation_fn=SiLU,
                    interaction_bias=interaction_bias,
                    message_normalization=message_normalization,
                    avg_num_neighbors=self.avg_num_neighbors,
                    layer_normalization_1=layer_normalization_1,
                    layer_normalization_2=layer_normalization_2,
                    residual_mlp_1=residual_mlp_1,
                    residual_mlp_2=residual_mlp_2,
                    qk_non_linearity=qk_act,
                )
                for _ in range(num_layers)
            ]
        )

        qk = era_qk_num_features if era_qk_num_features is not None else min(32, num_features)
        if qk % 2 != 0:
            qk += 1
        self.efa_blocks = build_efa_blocks(
            num_layers,
            num_features,
            era_use_in_iterations=era_iters,
            num_features_qk=qk,
            num_features_v=era_v_num_features if era_v_num_features is not None else min(16, num_features),
            lebedev_num=era_lebedev_num,
            max_frequency=era_max_frequency,
            max_length=era_max_length,
            as_delta=True,
        )

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
                elif isinstance(layer, BaseAtomEmbedding):
                    pass
                elif isinstance(layer, BaseRBF):
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
        Dij_sr: Tensor,
        Ra: Optional[Tensor] = None,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run Euclidean transformer stack (+ optional EFA on invariants).

        Enerzyme ``vij_sr = R_j - R_i``. So3krates-torch evaluates SH on the
        negated displacements; we match that convention with ``-vij_sr``.
        Receivers = ``idx_i_sr``, senders = ``idx_j_sr``.
        """
        num_atoms = atom_embedding.shape[0]
        dtype = atom_embedding.dtype
        receivers = idx_i_sr.long()
        senders = idx_j_sr.long()
        rbf = rbf.to(dtype=dtype)
        vij_sr = vij_sr.to(dtype=dtype)
        Dij_sr = Dij_sr.to(dtype=dtype)

        # Match So3krates-torch: SH on -edge_vectors when vectors are R_j - R_i.
        sh_vectors = self.spherical_harmonics(-vij_sr)
        cutoffs = self._cutoff_fn(Dij_sr, self.cutoff_sr).to(dtype=dtype)

        inv_features = atom_embedding
        if self.initialize_ev_to_zeros:
            ev_features = atom_embedding.new_zeros(num_atoms, self.m_tot)
        else:
            # Scatter cutoff-weighted SH to receivers (optional non-default path).
            from torch_scatter import scatter_sum

            ev_features = scatter_sum(
                cutoffs[:, None] * sh_vectors,
                receivers,
                dim=0,
                dim_size=num_atoms,
            )

        if self.era_enabled:
            if Ra is None:
                raise ValueError("Ra is required when era_use_in_iterations is set")
            if batch_seg is None:
                batch_seg = torch.zeros(
                    num_atoms, dtype=torch.long, device=atom_embedding.device
                )
            Ra = Ra.to(dtype=dtype)

        for i, layer in enumerate(self.layers):
            inv_features, ev_features = layer(
                inv_features,
                ev_features,
                rbf=rbf,
                senders=senders,
                receivers=receivers,
                sh_vectors=sh_vectors,
                cutoffs=cutoffs,
            )
            if self.era_enabled:
                delta = apply_efa_if_configured(
                    inv_features,
                    Ra,
                    batch_seg,
                    self.efa_blocks[i],
                    as_delta=True,
                )
                inv_features = inv_features + delta

        return {
            "atom_feature": inv_features,
            "atom_sphere_feature": ev_features,
        }
