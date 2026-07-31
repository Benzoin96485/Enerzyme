"""DPA4 Core for Enerzyme (arXiv:2606.02419).

Native PyTorch reimplementation of the DPA4 EMFA SO(2) descriptor.
Subclasses BaseFFCore; mirrors EquiformerV3Core registration style.

v1 limitations:
- lmax <= 2 (Wigner-D uses direct quaternion formulas)
- message_node_so3=False (SO3GridNet message path not implemented)
- n_focus=1 primary path; n_focus>1 API supported but less tested
- No l_schedule, kmax>1, grid_mlp branches, InnerClamp/BridgingSwitch
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import (
    BaseAtomEmbedding,
    BaseFFCore,
    BaseRBF,
    DistanceLayer,
    RangeSeparationLayer,
)
from .edge_cache import EdgeCache, build_edge_cache
from .embedding import EnvironmentInitialEmbedding, GeometricInitialEmbedding
from .interaction import SeZMInteractionBlock
from .norm import EquivariantRMSNorm
from .radial import C3CutoffEnvelope, RadialBasis, RadialMLP
from .wignerd import WignerDCalculator


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "channels": 32,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "mixing_layers": 3,
            "n_focus": 1,
            "focus_dim": 0,
            "n_atten_head": 1,
            "radial_so2_rank": 1,
            "radial_so2_mode": "degree_channel",
            "ffn_so3_grid": True,
            "message_node_so3": False,
            "use_env_seed": True,
            "n_radial": 16,
            "basis_type": "bessel",
            "sandwich_norm": [False, True, True, False],
            "ffn_blocks": 1,
            "ffn_neurons": 96,
            "glu_activation": True,
            "lebedev_quadrature": True,
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


class DPA4Core(BaseFFCore):
    """DPA4 EMFA SO(2) equivariant graph transformer Core."""

    def __str__(self) -> str:
        return f"""
###############################################################################
# DPA4 Core (arXiv:2606.02419)                                                #
# EMFA SO(2) convolution; channels={self.channels}, lmax={self.lmax}, mmax={self.mmax}  #
###############################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        max_Za: int = 94,
        channels: int = 32,
        lmax: int = 2,
        mmax: int = 1,
        n_blocks: int = 2,
        mixing_layers: int = 3,
        n_focus: int = 1,
        focus_dim: int = 0,
        n_atten_head: int = 1,
        radial_so2_rank: int = 1,
        radial_so2_mode: str = "degree_channel",
        ffn_so3_grid: bool = True,
        message_node_so3: bool = False,
        use_env_seed: bool = True,
        n_radial: int = 16,
        basis_type: str = "bessel",
        cutoff_sr: float = 6.0,
        sandwich_norm: Optional[list] = None,
        ffn_blocks: int = 1,
        ffn_neurons: int = 96,
        glu_activation: bool = True,
        lebedev_quadrature: bool = True,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
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
        self.max_Za = max_Za
        self.channels = channels
        self.lmax = lmax
        self.mmax = mmax
        self.n_blocks = n_blocks
        self.cutoff_sr = cutoff_sr
        self.n_radial = n_radial
        self.use_env_seed = use_env_seed

        self.dim_feature_out = channels
        self.feature_irreps = f"{channels}x0e"

        # Internal radial basis and envelope (no external RBF needed)
        self.radial_basis = RadialBasis(
            rcut=cutoff_sr, n_radial=n_radial,
            basis_type=basis_type, exponent=envelope_exponent + 2,
        )
        self.envelope = C3CutoffEnvelope(rcut=cutoff_sr, exponent=envelope_exponent)

        # Wigner-D calculator
        self.wigner_calc = WignerDCalculator(lmax=lmax)

        # Project atom embedding to sphere channels
        self.sphere_proj = nn.Linear(dim_embedding, channels)

        # Radial feature MLP: n_radial -> (lmax+1) * channels
        self.radial_feat_mlp = RadialMLP(
            n_radial,
            (lmax + 1) * channels,
            hidden=[64],
        )

        # Geometric initial embedding
        self.gie = GeometricInitialEmbedding(lmax=lmax, channels=channels)

        # Environment initial embedding (FiLM on l=0)
        if use_env_seed:
            self.env_embed = EnvironmentInitialEmbedding(
                ntypes=max_Za + 1,
                n_radial=n_radial,
                channels=channels,
            )
        else:
            self.env_embed = None

        # Interaction blocks
        if sandwich_norm is None:
            sandwich_norm = [False, True, True, False]
        self.blocks = ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(SeZMInteractionBlock(
                lmax=lmax,
                mmax=mmax,
                channels=channels,
                n_focus=n_focus,
                focus_dim=focus_dim,
                mixing_layers=mixing_layers,
                n_atten_head=n_atten_head,
                radial_so2_mode=radial_so2_mode,
                radial_so2_rank=radial_so2_rank,
                n_radial=n_radial,
                ffn_neurons=ffn_neurons,
                ffn_blocks=ffn_blocks,
                ffn_so3_grid=ffn_so3_grid,
                lebedev_quadrature=lebedev_quadrature,
                glu_activation=glu_activation,
                sandwich_norm=sandwich_norm,
                message_node_so3=message_node_so3,
            ))

        # Final norm
        self.norm = EquivariantRMSNorm(lmax, channels)

    def build(self, built_layers: List[Module]) -> None:
        """Wire pre/post layers following Enerzyme convention."""
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
                    continue  # We use internal radial basis
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        device = atom_embedding.device
        dtype = atom_embedding.dtype
        num_atoms = atom_embedding.shape[0]
        Za = Za.long()

        if batch_seg is None:
            batch_seg = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # Build edge cache with internal radial basis and envelope
        edge_cache = build_edge_cache(
            idx_i=idx_i_sr.long(),
            idx_j=idx_j_sr.long(),
            vij=vij_sr.to(dtype=dtype),
            n_nodes=num_atoms,
            radial_basis=self.radial_basis,
            envelope=self.envelope,
            wigner_calc=self.wigner_calc,
            random_gamma=self.training,
        )

        # Radial features: (E, lmax+1, C)
        edge_rbf = edge_cache.edge_rbf  # (E, n_radial)
        radial_feat = self.radial_feat_mlp(edge_rbf)  # (E, (lmax+1)*C)
        radial_feat = radial_feat.reshape(-1, self.lmax + 1, self.channels)

        # Initialize node features
        ebed_dim = (self.lmax + 1) ** 2
        scalar_feat = self.sphere_proj(atom_embedding)  # (N, C)

        # Geometric initial embedding (l >= 1)
        gie_feat = self.gie(num_atoms, edge_cache, radial_feat[:, 1:, :])
        # gie_feat: (N, D, C) — l=0 row is zero

        # Combine scalar + GIE
        x = gie_feat.clone()
        # Add scalar to l=0 via out-of-place op
        scalar_component = scalar_feat.unsqueeze(1)  # (N, 1, C)
        l0_mask = torch.zeros(1, ebed_dim, 1, device=device, dtype=dtype)
        l0_mask[0, 0, 0] = 1.0
        x = x + scalar_component * l0_mask

        # Environment seed FiLM (l = 0)
        if self.env_embed is not None:
            film_logits = self.env_embed(edge_cache, Za, num_atoms)  # (N, 2*C)
            scale = film_logits[:, :self.channels]
            shift = film_logits[:, self.channels:]
            # Apply FiLM to l=0 only, out-of-place
            x_l0 = x[:, 0, :] * (1.0 + torch.tanh(scale)) + shift  # (N, C)
            film_delta = x_l0 - x[:, 0, :]  # (N, C)
            x = x + film_delta.unsqueeze(1) * l0_mask

        # Add singleton focus dim: (N, D, C) -> (N, D, 1, C)
        x = x.unsqueeze(2)

        # Interaction blocks
        for blk in self.blocks:
            x = blk(x, edge_cache, radial_feat)

        # Final norm
        x = self.norm(x)

        # Remove focus dim
        atom_sphere_feature = x.squeeze(2)  # (N, D, C)

        # Expand to full (N, (lmax+1)^2, C) if needed
        atom_feature = atom_sphere_feature[:, 0, :]  # (N, C)

        return {
            "atom_feature": atom_feature,
            "atom_sphere_feature": atom_sphere_feature,
        }
