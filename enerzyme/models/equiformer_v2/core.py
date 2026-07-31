"""EquiformerV2 Core (Liao et al., ICLR 2024).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs). Embedding / RBF stay in pre-core
layers; energy / force / charge heads stay in post-core layers.

Adapted from https://github.com/atomicarchitects/equiformer_v2 (MIT license).
Distinct from Equiformer V1 (``enerzyme.models.equiformer``) and paper eSCN.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    init_edge_rot_mat,
    get_normalization_layer,
)
from .input_block import EdgeDegreeEmbedding
from .interaction import TransBlockV2

# Typical MD17 / small-molecule average degree (same ballpark as Equiformer V1).
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
    {"name": "GaussianSmearing"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 64,
            "attn_hidden_channels": 64,
            "num_heads": 4,
            "attn_alpha_channels": 32,
            "attn_value_channels": 16,
            "ffn_hidden_channels": 128,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "edge_channels": 64,
            "norm_type": "rms_norm_sh",
            "use_atom_edge_embedding": True,
            "use_gate_act": False,
            "use_grid_mlp": False,
            "use_sep_s2_act": True,
            "use_attn_renorm": True,
            "alpha_drop": 0.0,
            "drop_path_rate": 0.0,
            "proj_drop": 0.0,
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


class EquiformerV2Core(BaseFFCore):
    """EquiformerV2 SO(2) graph-attention transformer Core."""

    def __str__(self) -> str:
        return """
#################################################################################
# EquiformerV2 Core (Liao et al., ICLR 2024, arXiv:2306.12059)                 #
# SO(2) equivariant graph attention + S2/gate FFN; atom_feature = l=0 channels  #
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
        ffn_hidden_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        num_layers: int = 2,
        edge_channels: int = 64,
        norm_type: str = "rms_norm_sh",
        grid_resolution: Optional[int] = None,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        avg_degree: float = _AVG_DEGREE,
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
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.lmax_list = [lmax]
        self.mmax_list = [mmax]
        self.num_layers = num_layers
        self.edge_channels = edge_channels
        self.norm_type = norm_type
        self.grid_resolution = grid_resolution
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.avg_degree = avg_degree
        self.num_resolutions = len(self.lmax_list)

        self.dim_feature_out = self.num_resolutions * self.sphere_channels
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.sphere_proj = torch.nn.Linear(dim_embedding, self.dim_feature_out)

        # Mutable list shared with EdgeDegreeEmbedding / TransBlockV2 (updated each forward).
        self.SO3_rotation: List[Optional[SO3_Rotation]] = [None] * self.num_resolutions

        # Mapping lives on CPU at init; tensors move with module / are .to()'d in ops.
        self.mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, torch.device("cpu")
        )

        self.SO3_grid = ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = ModuleList()
            for mval in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        mval,
                        resolution=grid_resolution,
                        normalization="component",
                        rescale_by_mmax=True,
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # edge_channels_list: [num_rbf, edge_channels, edge_channels]
        self.edge_channels_list = [num_rbf] + [edge_channels] * 2

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        self.blocks = ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                TransBlockV2(
                    self.sphere_channels,
                    self.attn_hidden_channels,
                    self.num_heads,
                    self.attn_alpha_channels,
                    self.attn_value_channels,
                    self.ffn_hidden_channels,
                    self.sphere_channels,
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_rotation,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.max_num_elements,
                    self.edge_channels_list,
                    self.use_atom_edge_embedding,
                    self.use_m_share_rad,
                    attn_activation,
                    use_s2_act_attn,
                    use_attn_renorm,
                    ffn_activation,
                    use_gate_act,
                    use_grid_mlp,
                    use_sep_s2_act,
                    norm_type,
                    alpha_drop,
                    drop_path_rate,
                    proj_drop,
                )
            )

        self.norm = get_normalization_layer(
            norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels
        )

        # Expose FFN hyperparameters for optional EquiformerV2FeedForwardReadout.
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

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
    ) -> Tensor:
        """Return full spherical node coeffs after TransBlocks + final norm.

        When ``edge_rot_mat`` is provided it is used as-is (parity harness).
        Edge convention matches eSCN: ``edge_index = stack(idx_i, idx_j)``.
        """
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.long()

        if batch_seg is None:
            batch_seg = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # Move mapping buffers to the compute device (created on CPU at init).
        if self.mappingReduced.device != device:
            self.mappingReduced.device = device
            self.mappingReduced.l_harmonic = self.mappingReduced.l_harmonic.to(device)
            self.mappingReduced.m_harmonic = self.mappingReduced.m_harmonic.to(device)
            self.mappingReduced.m_complex = self.mappingReduced.m_complex.to(device)
            self.mappingReduced.res_size = self.mappingReduced.res_size.to(device)
            self.mappingReduced.to_m = self.mappingReduced.to_m.to(device)
            self.mappingReduced.m_size = self.mappingReduced.m_size.to(device)
            self.mappingReduced._mask_indices_cache = None
            self.mappingReduced._rotate_inv_rescale_cache = None

        edge_index = torch.stack([idx_i_sr.long(), idx_j_sr.long()], dim=0)
        if edge_rot_mat is None:
            edge_rot_mat = init_edge_rot_mat(vij_sr.to(dtype=dtype))

        for i in range(self.num_resolutions):
            self.SO3_rotation[i] = SO3_Rotation(
                edge_rot_mat,
                self.lmax_list[i],
                apply_rotate_inv_rescale=True,
            )

        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            dtype,
        )
        sphere0 = self.sphere_proj(atom_embedding)
        offset = 0
        offset_res = 0
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = sphere0[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        edge_degree = self.edge_degree_embedding(Za, rbf, edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        for blk in self.blocks:
            x = blk(x, Za, rbf, edge_index, batch=batch_seg)

        x.embedding = self.norm(x.embedding)
        return x.embedding

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
        features = []
        offset_res = 0
        for i in range(self.num_resolutions):
            features.append(atom_sphere_feature[:, offset_res, :])
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        atom_feature = torch.cat(features, dim=-1)
        return {
            "atom_feature": atom_feature,
            "atom_sphere_feature": atom_sphere_feature,
        }
