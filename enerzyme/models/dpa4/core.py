"""DPA4 Core for Enerzyme (arXiv:2606.02419).

Native PyTorch reimplementation of the DPA4 EMFA SO(2) descriptor.
Subclasses BaseFFCore; mirrors EquiformerV3Core registration style.

v1 limitations:
- Wigner-D via shared e3nn/Jd backend (``lmax`` up to packaged ``Jd.pt`` max)
- message_node_so3=False (SO3GridNet message path not implemented)
- n_focus=1 primary path; n_focus>1 API supported but less tested
- No l_schedule, kmax>1, grid_mlp branches, InnerClamp/BridgingSwitch
"""

from __future__ import annotations

import math
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
from ..so3 import C3CutoffEnvelope
from ..so3.indexing import build_gie_zonal_index, get_so3_dim
from ..so3.layer_norm import EquivariantDegreeRMSNorm
from ..so3.radial import BesselC3RadialBasis, RadialMLP
from ..so3.wigner_quaternion import WignerDCalculator
from .interaction import SeZMInteractionBlock
from .so2 import EdgeCache, build_edge_cache


class GeometricInitialEmbedding(nn.Module):
    """Geometric initial embedding: zonal (m=0) rotated radial features.

    For l>=1, rotates radial features using Dt_full zonal coupling.
    l=0 component is zero (comes from type embedding).
    """

    def __init__(self, lmax: int, channels: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.ebed_dim = get_so3_dim(lmax)

        row, m0_col, rad_l = build_gie_zonal_index(lmax)
        self.register_buffer("non_scalar_row", torch.from_numpy(row).long())
        self.register_buffer("zonal_m0_col", torch.from_numpy(m0_col).long())
        self.register_buffer("radial_slot", torch.from_numpy(rad_l).long())

    def forward(self, n_nodes: int, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            n_nodes: N
            edge_cache: EdgeCache with Dt_full, dst, inv_sqrt_deg
            radial_feat: (E, lmax, C) for l=1..lmax

        Returns:
            (N, D, C) initial features (l=0 row is zero)
        """
        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype

        if self.lmax == 0:
            return torch.zeros(n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype)

        Dt = edge_cache.Dt_full  # (E, D, D)
        n_edge = Dt.shape[0]
        dim_full = Dt.shape[-1]

        # Gather zonal coupling: Dt[e, row, m0_col] for each non-scalar position
        flat_idx = self.non_scalar_row * dim_full + self.zonal_m0_col
        zonal = Dt.reshape(n_edge, dim_full * dim_full)[:, flat_idx]  # (E, D-1)

        # Broadcast radial features
        rad_val = radial_feat[:, self.radial_slot, :]  # (E, D-1, C)
        message = zonal.unsqueeze(-1) * rad_val  # (E, D-1, C)

        # Scatter to nodes
        non_scalar_out = torch.zeros(n_nodes, self.ebed_dim - 1, self.channels,
                                      device=device, dtype=dtype)
        dst_expand = edge_cache.dst.unsqueeze(-1).unsqueeze(-1).expand_as(message)
        non_scalar_out.scatter_add_(0, dst_expand, message)

        # Prepend zero l=0 row
        zero_l0 = torch.zeros(n_nodes, 1, self.channels, device=device, dtype=dtype)
        out = torch.cat([zero_l0, non_scalar_out], dim=1)

        # Normalize by smooth degree
        out = out * edge_cache.inv_sqrt_deg.to(out.dtype)
        return out


class EnvironmentInitialEmbedding(nn.Module):
    """Environment matrix initial embedding for l=0 (FiLM conditioning).

    Builds r_tilde = [s, s*r_hat], computes G network, aggregates outer
    product by destination, constructs D matrix, projects to FiLM logits.
    """

    def __init__(
        self,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.channels = channels
        self.embed_dim = embed_dim
        self.axis_dim = axis_dim
        self.eps = eps
        self.coord_dim = 4

        rbf_out_dim = max(32, embed_dim - 2 * type_dim)
        self.rbf_proj = nn.Sequential(
            nn.Linear(n_radial, rbf_out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(rbf_out_dim, rbf_out_dim, bias=False),
        )

        self.env_type_embed = nn.Embedding(ntypes + 1, type_dim, padding_idx=ntypes)
        nn.init.normal_(self.env_type_embed.weight[:ntypes], std=1.0 / math.sqrt(ntypes + type_dim))
        self.env_type_embed.weight.data[ntypes] = 0.0

        g_in = rbf_out_dim + 2 * type_dim
        self.g_net = nn.Sequential(
            nn.Linear(g_in, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )

        self.output_proj = nn.Linear(embed_dim * axis_dim, 2 * channels, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, edge_cache: EdgeCache, atype: Tensor, n_nodes: int) -> Tensor:
        """Returns FiLM logits (N, 2*C)."""
        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec
        edge_rbf = edge_cache.edge_rbf
        edge_env = edge_cache.edge_env
        device = edge_vec.device

        r_sq = (edge_vec * edge_vec).sum(-1, keepdim=True)
        inv_r = torch.rsqrt(r_sq + self.eps ** 2)
        s = edge_env * inv_r
        r_hat = edge_vec * inv_r
        r_tilde = torch.cat([s, s * r_hat], dim=-1)  # (E, 4)

        atype_src = atype[src]
        atype_dst = atype[dst]
        type_src = self.env_type_embed(atype_src)
        type_dst = self.env_type_embed(atype_dst)

        rbf_proj = self.rbf_proj(edge_rbf)
        g_input = torch.cat([rbf_proj, type_src, type_dst], dim=-1)
        g = self.g_net(g_input)  # (E, embed_dim)

        outer = r_tilde.unsqueeze(-1) * g.unsqueeze(-2)  # (E, 4, embed_dim)
        outer_flat = outer.reshape(-1, self.coord_dim * self.embed_dim)

        env_agg = torch.zeros(n_nodes, self.coord_dim * self.embed_dim,
                              device=device, dtype=outer_flat.dtype)
        env_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(outer_flat), outer_flat)
        env_agg = env_agg.reshape(n_nodes, self.coord_dim, self.embed_dim)

        env_agg = env_agg * edge_cache.inv_sqrt_deg.to(env_agg.dtype)

        env_agg_t = env_agg.permute(0, 2, 1)  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, :self.axis_dim]  # (N, 4, axis_dim)
        D = torch.bmm(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        D_flat = D.reshape(n_nodes, self.embed_dim * self.axis_dim)
        return self.output_proj(D_flat)


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
        self.radial_basis = BesselC3RadialBasis(
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
        self.norm = EquivariantDegreeRMSNorm(lmax, channels)

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
