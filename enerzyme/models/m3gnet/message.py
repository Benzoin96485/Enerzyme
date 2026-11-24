"""Shared bond and three-body geometry for M3GNet and MatterSim."""

from torch import nn

from ..mp.block import BlockState
from ..mp.core import Embed
from ..mp.basis.envelope import polynomial_envelope
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.rbf import BaseRBF, GaussianRBFLayer
from .basis import joint_zonal_basis
from .geometry import bond_angle
from .interaction import M3GNetBlock


class MaterialEmbed(Embed):
    input_fields = ("atom_embedding",)

    def __init__(self, max_l, num_rbf, dim_edge, threebody_cutoff, envelope_exponent):
        super().__init__()
        self.max_l = max_l
        self.num_rbf = num_rbf
        self.degree = max_l * num_rbf
        self.threebody_cutoff = threebody_cutoff
        self.envelope_exponent = envelope_exponent
        self.edge_embed = nn.Sequential(nn.Linear(self.degree, dim_edge), nn.SiLU())
        self.centrality = None
        self.attention_cutoff = None
        self.attention_width = None

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        if node.ndim != 2:
            raise ValueError(f"atom_embedding must be (N, F), got {tuple(node.shape)}")
        if ctx.rbf.shape != (ctx.num_edges, self.degree):
            raise ValueError(f"rbf must be (E, {self.degree}), got {tuple(ctx.rbf.shape)}")
        if ctx.dist.shape != (ctx.num_edges,) or ctx.sender.shape != ctx.receiver.shape:
            raise ValueError("Dij_sr, rbf, idx_i_sr, and idx_j_sr must share the edge count")
        if ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError(f"vij_sr must be (E, 3), got {tuple(ctx.vec.shape)}")
        first, second = ctx.triplets("receiver", cutoff=self.threebody_cutoff)
        angle = bond_angle(ctx.vec, first, second)
        joint = joint_zonal_basis(ctx.rbf, angle, second, self.max_l, self.num_rbf)
        envelope = polynomial_envelope(ctx.dist, self.threebody_cutoff, self.envelope_exponent)
        aux = {"threebody": (first, second, joint, envelope)}
        if self.attention_width is not None:
            radial = inputs.get("graphormer_rbf")
            if radial is None:
                raise ValueError("graphormer_rbf is required when centrality or attention is enabled")
            if radial.shape != (ctx.num_edges, self.attention_width):
                raise ValueError(f"graphormer_rbf must be (E, {self.attention_width}), got {tuple(radial.shape)}")
            if self.centrality is not None:
                node = node + self.centrality(radial, ctx.dist, ctx.receiver, ctx.num_nodes, self.attention_cutoff)
            aux["attention_mask"] = polynomial_envelope(ctx.dist, self.attention_cutoff, 3)
        return BlockState(scalar=node, edge=self.edge_embed(ctx.rbf), aux=aux)


class MaterialBlock(M3GNetBlock):
    def forward(self, state, ctx):
        node, edge = super().forward(
            state.scalar, state.edge, ctx.rbf, ctx.receiver, ctx.sender, *state.aux["threebody"]
        )
        return BlockState(scalar=node, edge=edge, aux=state.aux)


def build_material_stack(core, built_layers):
    core.adopt_radius_stack(built_layers)
    radial = [
        layer for layer in built_layers if isinstance(layer, BaseRBF)
    ]
    spherical = [
        layer for layer in radial if getattr(layer, "flavor", None) == "m3gnet"
    ]
    extra = [layer for layer in radial if layer not in spherical]
    if len(spherical) != 1:
        if len(radial) == 1:
            flavor = getattr(radial[0], "flavor", None)
            raise ValueError(
                "M3GNetCore requires SphericalBesselRBF (flavor 'm3gnet'), "
                f"found {flavor!r}"
            )
        raise ValueError(
            "M3GNetCore requires exactly one SphericalBesselRBF layer "
            f"before the Core, found {len(spherical)}"
        )
    if extra and not core.uses_graphormer_basis:
        raise ValueError(
            "M3GNetCore requires exactly one SphericalBesselRBF layer "
            f"before the Core, found {len(radial)}"
        )
    layer = spherical[0]
    if int(layer.max_l) != core.max_l or int(layer.num_rbf) != core.num_rbf:
        raise ValueError(
            "SphericalBesselRBF "
            f"(max_l={layer.max_l}, num_rbf={layer.num_rbf}) must match "
            f"the Core (max_l={core.max_l}, num_rbf={core.num_rbf})"
        )
    if abs(float(layer.cutoff_sr) - core.cutoff_sr) > 1e-4:
        raise ValueError(
            "SphericalBesselRBF cutoff_sr "
            f"({layer.cutoff_sr}) must equal M3GNet cutoff_sr "
            f"({core.cutoff_sr})"
        )
    core.radial_basis_function = layer
    for candidate in built_layers:
        if isinstance(candidate, BaseAtomEmbedding):
            core.atom_embedding = candidate
    if not core.uses_graphormer_basis:
        return
    extra = [
        layer
        for layer in built_layers
        if isinstance(layer, BaseRBF) and getattr(layer, "flavor", None) != "m3gnet"
    ]
    if len(extra) != 1 or not isinstance(extra[0], GaussianRBFLayer):
        raise ValueError(
            "MatterSimCore requires exactly one GaussianRBF next to "
            "SphericalBesselRBF when centrality or attention is enabled, "
            f"found {[type(layer).__name__ for layer in extra]}"
        )
    layer = extra[0]
    if layer.apply_cutoff_fn:
        raise ValueError(
            "GaussianRBF for MatterSim must set apply_cutoff_fn: false. "
            "The Core multiplies the polynomial cutoff after the softmax "
            "and inside the centrality sum."
        )
    if int(layer.num_rbf) != core.graphormer_num_basis:
        raise ValueError(
            "GaussianRBF num_rbf "
            f"({layer.num_rbf}) must equal graphormer_num_basis "
            f"({core.graphormer_num_basis})"
        )
    if abs(float(layer.cutoff_sr) - core.cutoff_sr) > 1e-4:
        raise ValueError(
            "GaussianRBF cutoff_sr "
            f"({layer.cutoff_sr}) must equal MatterSim cutoff_sr "
            f"({core.cutoff_sr})"
        )
    layer.reset_field_name(rbf="graphormer_rbf")
    core.graphormer_basis = layer
