"""Native eSEN Core (Fu et al., 2025, arXiv:2502.12147).

Equivariant Smooth Energy Network. The Core maps an external scalar atom
embedding and radial basis to spherical latents. Nodewise layers stay in
the spherical-harmonic basis (SiLU gate), and a polynomial envelope makes
edge messages vanish smoothly at the cutoff. Forces stay outside the Core
(``EnergyReduce`` + ``Force``), so they are energy-conserving.

Distinct from ``uma_qs`` / ``UMAWrapperQS`` in the parent package, which
wrap a fairchem UMA checkpoint. This module does not import fairchem and
does not load those checkpoints.

Reuses :class:`~enerzyme.models.so3.so2_ops.SO2_Convolution`,
:class:`~enerzyme.models.so3.linear.SO3Linear`,
:class:`~enerzyme.models.so3.activation.GateActivation`,
:func:`~enerzyme.models.so3.layer_norm.get_normalization_layer`,
:class:`~enerzyme.models.blocks.radial_mlp.RadialFunction`, and
:func:`~enerzyme.models.cutoff.polynomial_envelope`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ...layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from ...so3 import (
    CoefficientMapping,
    SO3_Rotation,
    build_so3_grid_table,
    get_normalization_layer,
    init_edge_rot_mat,
)
from .interaction import EdgeDegreeEmbedding, eSENBlock

# fairchem eSEN divides aggregated messages by 5 (about sqrt of a typical
# neighbor count). It is a constant, not a dataset statistic.
_RESCALE = 5.0

# Released widths. The checked-in stack stays small; swap these onto the Core
# when reproducing a published model. Radial basis and cutoff stay in
# build_params / the RBF layer (Gaussian, 10 functions, 6 Å on MPTrj;
# 64 Gaussians for the 30M OAM model; Bessel, 10 functions, 5 Å on SPICE).
PUBLISHED_MP_3M_CORE_PARAMS = {
    "sphere_channels": 128,
    "hidden_channels": 128,
    "edge_channels": 128,
    "lmax": 2,
    "mmax": 2,
    "num_layers": 2,
    "norm_type": "rms_norm_sh",
    "act_type": "gate",
    "mlp_type": "spectral",
    "use_envelope": True,
    "rescale_factor": _RESCALE,
}
PUBLISHED_SPICE_6M_CORE_PARAMS = {
    **PUBLISHED_MP_3M_CORE_PARAMS,
    "num_layers": 4,
}
PUBLISHED_30M_CORE_PARAMS = {
    **PUBLISHED_MP_3M_CORE_PARAMS,
    "lmax": 3,
    "mmax": 2,
    "num_layers": 10,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 16,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": None,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "GaussianRBF",
        "params": {"flavor": "esen", "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 16,
            "hidden_channels": 16,
            "edge_channels": 16,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "norm_type": "rms_norm_sh",
            "act_type": "gate",
            "mlp_type": "spectral",
            "use_envelope": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "num_hidden_layers": 2,
            "use_bias_out": True,
            "activation_fn": "swish",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class eSENCore(BaseFFCore):
    """eSEN message-passing Core. Emits ``atom_feature`` and ``atom_sphere_feature``."""

    def __str__(self) -> str:
        return """
#################################################################################
# eSEN Core (Fu et al., 2025, arXiv:2502.12147)                                #
# Smooth SO(2) edge convolution + spectral SiLU gate; atom_feature = l = 0     #
# UMA checkpoints stay on architecture uma_qs                                  #
#################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        max_Za: int = 94,
        sphere_channels: int = 16,
        hidden_channels: int = 16,
        edge_channels: int = 16,
        lmax: int = 2,
        mmax: int = 2,
        num_layers: int = 2,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        mlp_type: str = "spectral",
        use_envelope: bool = True,
        rescale_factor: float = _RESCALE,
        cutoff_sr: float = 6.0,
        grid_resolution: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
            },
            output_fields={"atom_feature", "atom_sphere_feature"},
        )
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")
        if lmax < 1:
            raise ValueError(f"lmax must be >= 1, got {lmax}")
        if rescale_factor <= 0:
            raise ValueError(f"rescale_factor must be positive, got {rescale_factor}")
        if act_type not in ("gate", "s2"):
            raise ValueError(f"Unknown act_type {act_type!r}")
        if mlp_type not in ("spectral", "grid"):
            raise ValueError(f"Unknown mlp_type {mlp_type!r}")

        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.max_Za = int(max_Za)
        self.sphere_channels = int(sphere_channels)
        self.hidden_channels = int(hidden_channels)
        self.edge_channels = int(edge_channels)
        self.lmax_list = [int(lmax)]
        self.mmax_list = [int(mmax)]
        self.num_layers = int(num_layers)
        self.norm_type = norm_type
        self.act_type = act_type
        self.mlp_type = mlp_type
        self.use_envelope = bool(use_envelope)
        self.rescale_factor = float(rescale_factor)
        self.cutoff_sr = float(cutoff_sr)
        self.num_resolutions = 1
        self.dim_feature_out = self.sphere_channels
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.sphere_proj = nn.Linear(self.dim_embedding, self.sphere_channels)
        self.source_embedding = nn.Embedding(self.max_Za + 1, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_Za + 1, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.mapping = CoefficientMapping(
            self.lmax_list, self.mmax_list, torch.device("cpu")
        )
        self.edge_channels_list = [
            self.num_rbf + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.mapping,
            self.edge_channels_list,
            self.rescale_factor,
            self.cutoff_sr,
            self.use_envelope,
        )

        self.SO3_grid = None
        if act_type == "s2" or mlp_type == "grid":
            self.SO3_grid = build_so3_grid_table(
                lmax,
                normalization="component",
                resolution=grid_resolution,
                rescale_by_mmax=True,
            )

        self.blocks = nn.ModuleList(
            [
                eSENBlock(
                    self.sphere_channels,
                    self.hidden_channels,
                    lmax,
                    mmax,
                    self.mapping,
                    self.edge_channels_list,
                    self.cutoff_sr,
                    norm_type,
                    act_type,
                    mlp_type,
                    self.use_envelope,
                    SO3_grid=self.SO3_grid,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = get_normalization_layer(
            norm_type, lmax=lmax, num_channels=self.sphere_channels
        )

    def build(self, built_layers: List[nn.Module]) -> None:
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

    def encode_sphere(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        edge_rot_mat: Optional[Tensor] = None,
    ) -> Tensor:
        """Full SH node features after the eSEN blocks and the final norm.

        Enerzyme edges use ``idx_i`` as the receiver and ``idx_j`` as the
        sender, with ``vij = R_j - R_i``. That is fairchem's
        ``(edge_index[0], edge_index[1]) = (sender, receiver)`` and
        ``pos[sender] - pos[receiver]``.

        Blocks always run, including when a system has no neighbors inside
        the cutoff. A vanished envelope message then matches the no-edge
        evaluation, which is the continuous limit as the last neighbor
        leaves the cutoff.
        """
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.to(device=device).long()
        rbf = rbf.to(device=device, dtype=dtype)
        vij = vij_sr.to(device=device, dtype=dtype)
        self.mapping.to(device)

        # fairchem: source = edge_index[0], receiver = edge_index[1]
        edge_index = torch.stack([idx_j_sr.to(device).long(), idx_i_sr.to(device).long()], dim=0)
        rotation = None
        x_edge = None
        edge_distance = None
        if edge_index.shape[1] > 0:
            if edge_rot_mat is None:
                edge_rot_mat = init_edge_rot_mat(vij)
            else:
                edge_rot_mat = edge_rot_mat.to(device=device, dtype=dtype)
            rotation = [
                SO3_Rotation(
                    edge_rot_mat,
                    self.lmax_list[0],
                    apply_rotate_inv_rescale=False,
                )
            ]
            edge_distance = torch.linalg.norm(vij, dim=-1)
            x_edge = torch.cat(
                (
                    rbf,
                    self.source_embedding(Za[edge_index[0]]),
                    self.target_embedding(Za[edge_index[1]]),
                ),
                dim=1,
            )

        node = atom_embedding.new_zeros(
            (num_atoms, (self.lmax_list[0] + 1) ** 2, self.sphere_channels)
        )
        node[:, 0, :] = self.sphere_proj(atom_embedding)
        if edge_index.shape[1] > 0:
            node = node + self.edge_degree(
                x_edge, edge_distance, edge_index, rotation, num_atoms
            )

        for block in self.blocks:
            node = block(node, x_edge, edge_distance, edge_index, rotation)
        return self.norm(node)

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        atom_sphere_feature = self.encode_sphere(
            atom_embedding, Za, rbf, idx_i_sr, idx_j_sr, vij_sr
        )
        return {
            "atom_feature": atom_sphere_feature[:, 0, :],
            "atom_sphere_feature": atom_sphere_feature,
        }
