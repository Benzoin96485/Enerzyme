"""Common operations for Cartesian-vector message-passing presets."""

from torch import nn

from ...layers.atom_embedding import RandomAtomEmbedding
from ..basis.angular import angular_dim, polynomial_spherical_harmonics
from ..ops.self_pairs import insert_missing_self_pairs, unit_direction
from ..basis.envelope import cosine_transition
from ..block import Block, BlockState
from ..core import Embed
from ..head import Head


def vector_metadata(width):
    return dict(dim_embedding=width, dim_feature_out=width, feature_irreps=f"{width}x0e",
                vector_feature_layout="cartesian_n3f")


class ScalarNormBlock(Block):
    """Normalize the scalar stream while passing the equivariant features through."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, state, ctx):
        return state._replace(scalar=self.norm(state.scalar))


def build_attention_stack(core, built_layers):
    core.adopt_radius_stack(built_layers)
    embed = core.embed
    radial = None
    for layer in built_layers:
        if layer is core:
            break
        fields = getattr(layer, "_output_fields", ())
        if "rbf" in fields and callable(getattr(layer, "get_rbf", None)):
            radial = layer
        if isinstance(layer, RandomAtomEmbedding):
            if int(layer.dim_embedding) != embed.dim_embedding:
                raise ValueError(
                    "RandomAtomEmbedding width "
                    f"({layer.dim_embedding}) != dim_embedding ({embed.dim_embedding})"
                )
            if int(layer.max_Za) != embed.max_Za:
                raise ValueError(
                    f"RandomAtomEmbedding max_Za ({layer.max_Za}) != "
                    f"Core max_Za ({embed.max_Za})"
                )
    if radial is not None:
        radial_cutoff = getattr(radial, "cutoff_sr", None)
        if (
            radial_cutoff is not None
            and abs(float(radial_cutoff) - embed.cutoff_sr) > 1e-5
        ):
            raise ValueError(
                f"radial cutoff_sr ({radial_cutoff}) != Core cutoff_sr ({embed.cutoff_sr})"
            )
        radial_width = getattr(radial, "num_rbf", None)
        if radial_width is not None and int(radial_width) != embed.num_rbf:
            raise ValueError(
                f"radial num_rbf ({radial_width}) != Core num_rbf ({embed.num_rbf})"
            )
    embed._radial_holder = [] if radial is None else [radial]


class VectorAttentionEmbed(Embed):
    """Initialize scalar, angular and optional edge states on a graph with self pairs."""

    input_fields = ("atom_embedding", "Za")

    def __init__(self, width, num_rbf, max_Za, cutoff, include_self, lmax,
                 neighborhood=None, edge_embedding=None):
        super().__init__()
        self.dim_embedding = width
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.cutoff_sr = cutoff
        self.include_self = include_self
        self.lmax = lmax
        self.angular_dim = angular_dim(lmax)
        self.neighborhood = neighborhood
        self.edge_embedding = edge_embedding
        self._radial_holder = []

    def forward(self, inputs, ctx):
        x = inputs["atom_embedding"]
        n = ctx.num_nodes
        if x.ndim != 2 or x.shape[-1] != self.dim_embedding:
            raise ValueError(f"atom_embedding must have shape [N, {self.dim_embedding}], got {tuple(x.shape)}")
        if ctx.rbf.ndim != 2 or ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf must have shape [E, {self.num_rbf}], got {tuple(ctx.rbf.shape)}")
        if ctx.receiver.ndim != 1 or ctx.receiver.shape != ctx.sender.shape:
            raise ValueError("idx_i_sr and idx_j_sr must be 1D edge indices of the same length")
        if ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError(f"vij_sr must have shape [{ctx.num_edges}, 3], got {tuple(ctx.vec.shape)}")
        dist = ctx.dist.reshape(-1)
        if ctx.rbf.shape[0] != ctx.num_edges or dist.shape[0] != ctx.num_edges:
            raise ValueError("rbf, Dij_sr and edge indices must have one row per edge")
        window = ctx.cutoff_values
        if window is None:
            window = cosine_transition(dist, self.cutoff_sr)
        elif window.numel() != ctx.num_edges:
            raise ValueError("cutoff_values_sr and Dij_sr must describe the same edges")
        receiver, sender, vec, dist, basis, window = insert_missing_self_pairs(
            n, ctx.receiver, ctx.sender, ctx.vec, dist, ctx.rbf, window.reshape(-1),
            enabled=self.include_self, radial_basis=self._radial_holder[0] if self._radial_holder else None,
            num_rbf=self.num_rbf,
        )
        direction = unit_direction(vec, dist)
        if self.edge_embedding is not None:
            direction = polynomial_spherical_harmonics(direction, self.lmax).to(x.dtype)
        if self.neighborhood is not None:
            if inputs.get("Za") is None:
                raise ValueError("neighbor_embedding requires atomic numbers Za")
            x = self.neighborhood(inputs["Za"].long(), x, receiver, sender, basis, window)
        vectors = x.new_zeros(n, self.angular_dim, self.dim_embedding)
        edges = None if self.edge_embedding is None else self.edge_embedding(receiver, sender, basis, x)
        ctx.cache["attention_edges"] = receiver, sender, basis, window, direction
        return BlockState(x, vectors, edges)


def vector_attention_step(layer, state, ctx):
    receiver, sender, basis, window, direction = ctx.cache["attention_edges"]
    dx, dv = layer(state.scalar, state.equi, receiver, sender, basis, window, direction)
    return BlockState(state.scalar + dx, state.equi + dv)


def edge_attention_step(layer, state, ctx):
    receiver, sender, _, window, direction = ctx.cache["attention_edges"]
    dx, dv, de = layer(state.scalar, state.equi, state.edge, receiver, sender, window, direction)
    edges = state.edge if de is None else state.edge + de
    return BlockState(state.scalar + dx, state.equi + dv, edges)


class AngularFeatureHead(Head):
    """Apply final normalization and separate Cartesian l=1 and optional l=2 features."""

    def __init__(self, width, lmax, vec_out_norm=None):
        super().__init__()
        self.out_norm = nn.LayerNorm(width)
        self.vec_out_norm = vec_out_norm
        self.lmax = lmax
        self.output_fields = ("atom_feature", "atom_vector_feature")
        if lmax == 2:
            self.output_fields += ("atom_l2_feature",)

    def forward(self, state, ctx, taps, inputs):
        scalar = self.out_norm(state.scalar)
        vector = state.equi if self.vec_out_norm is None else self.vec_out_norm(state.equi)
        out = {"atom_feature": scalar, "atom_vector_feature": vector if self.lmax == 1 else vector[:, :3].contiguous()}
        if self.lmax == 2:
            out["atom_l2_feature"] = vector[:, 3:].contiguous()
        return out
