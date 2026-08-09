"""EquiformerV3 Core (Liao et al., 2026, arXiv:2604.09130).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs). Embedding / RBF stay in pre-core
layers; energy / force / charge heads stay in post-core layers.

Adapted from https://github.com/atomicarchitects/equiformer_v3 (MIT license).
Distinct from EquiformerV2 (``enerzyme.models.equiformer_v2``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from ..so3 import (
    PolynomialEnvelope,
    SO3RotationFused,
    get_normalization_layer,
    init_edge_rot_mat,
)
from .input_block import EdgeDegreeEmbedding
from .interaction import TransBlockV3

# Typical MD17 / small-molecule average degree (same ballpark as Equiformer V2).
_AVG_DEGREE = 15.57930850982666

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "polynomial",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "GaussianRBF", "params": {"flavor": "SchNet", "apply_cutoff_fn": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 64,
            "attn_hidden_channels": 64,
            "num_heads": 4,
            "attn_alpha_channels": 32,
            "attn_value_channels": 16,
            "ffn_hidden_channels": 256,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "edge_channels": 64,
            "norm_type": "merge_layer_norm",
            "use_atom_edge_embedding": True,
            "use_envelope": True,
            "attn_activation": "sep-merge_gates2_swiglu",
            "ffn_activation": "sep-merge_gates2_swiglu",
            "use_grid_mlp": True,
            "use_attn_renorm": True,
            "attn_grid_resolution": [8, 8],
            "ffn_grid_resolution": [8, 8],
            "alpha_drop": 0.0,
            "drop_path_rate": 0.0,
            "proj_drop": 0.0,
            "attn_weights_drop": 0.0,
            "ffn_drop": 0.0,
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


def _as_resolution_list(value: Optional[Sequence[int]], default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    value = list(value)
    if len(value) == 1:
        return [value[0], value[0]]
    if len(value) != 2:
        raise ValueError(f"grid resolution must be length 1 or 2, got {value}")
    return value


class EquiformerV3Core(BaseFFCore):
    """EquiformerV3 SO(2) graph-attention transformer Core."""

    def __str__(self) -> str:
        return """
#################################################################################
# EquiformerV3 Core (Liao et al., 2026, arXiv:2604.09130)                       #
# Merged LN + SwiGLU-S2 + smooth-cutoff attention; atom_feature = l=0 channels  #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        sphere_channels: int = 64,
        attn_hidden_channels: int = 64,
        num_heads: int = 4,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 256,
        lmax: int = 2,
        mmax: int = 2,
        num_layers: int = 2,
        edge_channels: int = 64,
        norm_type: str = "merge_layer_norm",
        use_atom_edge_embedding: bool = True,
        use_envelope: bool = True,
        envelope_exponent: int = 5,
        attn_activation: str = "sep-merge_gates2_swiglu",
        ffn_activation: str = "sep-merge_gates2_swiglu",
        use_grid_mlp: bool = True,
        use_attn_renorm: bool = True,
        use_add_merge: bool = False,
        use_rad_l_parametrization: bool = True,
        softcap: Optional[float] = None,
        attn_eps: float = 1e-16,
        attn_grid_resolution: Optional[Sequence[int]] = None,
        ffn_grid_resolution: Optional[Sequence[int]] = None,
        alpha_drop: float = 0.0,
        attn_mask_rate: float = 0.0,
        attn_weights_drop: float = 0.0,
        value_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        avg_degree: float = _AVG_DEGREE,
        cutoff_sr: float = 5.0,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "batch_seg",
            },
            output_fields={"atom_feature", "atom_sphere_feature"},
        )
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")

        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.max_num_elements = max_Za + 1
        self.sphere_channels = sphere_channels
        self.num_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.num_layers = num_layers
        self.edge_channels = edge_channels
        self.norm_type = norm_type
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_envelope = use_envelope
        self.attn_activation = attn_activation
        self.ffn_activation = ffn_activation
        self.use_grid_mlp = use_grid_mlp
        self.use_attn_renorm = use_attn_renorm
        self.use_add_merge = use_add_merge
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.softcap = softcap
        self.attn_eps = attn_eps
        self.avg_degree = avg_degree
        self.cutoff_sr = cutoff_sr

        self.attn_grid_resolution_list = _as_resolution_list(
            attn_grid_resolution, [2 * (lmax + 1), 2 * (mmax + 1) + 1]
        )
        self.ffn_grid_resolution_list = _as_resolution_list(
            ffn_grid_resolution, [2 * (lmax + 1), 2 * (lmax + 1)]
        )

        self.dim_feature_out = self.sphere_channels
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.sphere_proj = torch.nn.Linear(dim_embedding, self.sphere_channels)

        self.so3_rotation = SO3RotationFused(
            self.lmax, self.mmax, use_rotation_mask=False
        )

        self.edge_channels_list = [num_rbf] + [edge_channels] * 2

        self.envelope_func = (
            PolynomialEnvelope(cutoff=self.cutoff_sr, exponent=envelope_exponent)
            if self.use_envelope
            else None
        )

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        self.blocks = ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                TransBlockV3(
                    num_in_channels=self.sphere_channels,
                    attn_hidden_channels=self.attn_hidden_channels,
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels,
                    attn_value_channels=self.attn_value_channels,
                    ffn_hidden_channels=self.ffn_hidden_channels,
                    num_out_channels=self.sphere_channels,
                    lmax=self.lmax,
                    mmax=self.mmax,
                    so3_rotation=self.so3_rotation,
                    attn_grid_resolution_list=self.attn_grid_resolution_list,
                    ffn_grid_resolution_list=self.ffn_grid_resolution_list,
                    max_num_elements=self.max_num_elements,
                    edge_channels_list=self.edge_channels_list,
                    use_atom_edge_embedding=self.use_atom_edge_embedding,
                    attn_activation=self.attn_activation,
                    use_attn_renorm=self.use_attn_renorm,
                    use_add_merge=self.use_add_merge,
                    use_rad_l_parametrization=self.use_rad_l_parametrization,
                    softcap=self.softcap,
                    attn_eps=self.attn_eps,
                    ffn_activation=self.ffn_activation,
                    use_grid_mlp=self.use_grid_mlp,
                    norm_type=self.norm_type,
                    alpha_drop=alpha_drop,
                    attn_mask_rate=attn_mask_rate,
                    attn_weights_drop=attn_weights_drop,
                    value_drop=value_drop,
                    drop_path_rate=drop_path_rate,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                )
            )

        self.norm = get_normalization_layer(
            self.norm_type, lmax=self.lmax, num_channels=self.sphere_channels
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

    def encode_sphere(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
        edge_rot_mat: Optional[Tensor] = None,
        edge_envelope_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Return full spherical node coeffs after TransBlocks + final norm."""
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.long()

        if batch_seg is None:
            batch_seg = torch.zeros(num_atoms, dtype=torch.long, device=device)

        edge_index = torch.stack([idx_i_sr.long(), idx_j_sr.long()], dim=0)
        edge_distance_vec = vij_sr.to(dtype=dtype)
        edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1)

        if edge_rot_mat is None:
            edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        self.so3_rotation.set_wigner(edge_rot_mat)

        if edge_envelope_weight is None and self.envelope_func is not None:
            edge_envelope_weight = self.envelope_func(edge_distance)

        x = torch.zeros(
            (num_atoms, (self.lmax + 1) ** 2, self.sphere_channels),
            device=device,
            dtype=dtype,
        )
        x[:, 0, :] = self.sphere_proj(atom_embedding)

        edge_degree = self.edge_degree_embedding(
            Za, rbf, edge_index, edge_envelope_weight
        )
        x = x + edge_degree

        source_atomic_numbers = Za[edge_index[0]]
        target_atomic_numbers = Za[edge_index[1]]
        for blk in self.blocks:
            x = blk(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                rbf,
                edge_index,
                edge_envelope_weight,
                batch_seg,
            )

        x = self.norm(x)
        return x

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        atom_sphere_feature = self.encode_sphere(
            atom_embedding,
            Za,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            batch_seg=batch_seg,
        )
        atom_feature = atom_sphere_feature[:, 0, :]
        return {
            "atom_feature": atom_feature,
            "atom_sphere_feature": atom_sphere_feature,
        }
