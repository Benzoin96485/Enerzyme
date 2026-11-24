"""Cartesian-vector message-passing presets."""

from __future__ import annotations

from torch import nn

from ..mp.block import FnBlock
from ..mp.core import ComposableCore
from ..mp.graph import GraphRequires
from ..mp.head import FeatureHead
from ..mp.message.vector import ScalarNormBlock, vector_metadata


def painn(
    dim_embedding: int = 128,
    num_rbf: int = 20,
    cutoff_sr: float = 5.0,
    num_interactions: int = 3,
    shared_interactions: bool = False,
    shared_filters: bool = False,
    propagate_vectors: bool = True,
    use_vector_scalar_product: bool = True,
    epsilon: float = 1e-8,
    ) -> ComposableCore:
    if num_interactions < 1:
        raise ValueError(f"num_interactions must be >= 1, got {num_interactions}")
    if dim_embedding < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")

    from ..painn.interaction import PaiNNEmbed, PaiNNInteraction, PaiNNMixing, PaiNNBlock

    embed = PaiNNEmbed(dim_embedding, num_rbf, cutoff_sr, num_interactions, shared_filters)
    n_blocks = 1 if shared_interactions else num_interactions
    messages = [PaiNNInteraction(dim_embedding, propagate_vectors) for _ in range(n_blocks)]
    updates = [PaiNNMixing(dim_embedding, epsilon, use_vector_scalar_product) for _ in range(n_blocks)]
    blocks = [PaiNNBlock(message, update) for message, update in zip(messages, updates)]
    return ComposableCore(
        embed, blocks, FeatureHead(equi_field="atom_vector_feature"), GraphRequires(),
        schedule=[0] * num_interactions if shared_interactions else None,
        positional_fields=_VECTOR_INPUTS + ("cutoff_values_sr",), architecture="painn", num_interactions=num_interactions, shared_filters=shared_filters,
        shared_interactions=shared_interactions, **vector_metadata(dim_embedding),
    )


def newtonnet(
    dim_embedding: int = 128,
    num_rbf: int = 20,
    cutoff_sr: float = 5.0,
    num_interactions: int = 3,
    variant: str = "paper",
    envelope_p: int = 9,
    layer_norm: bool = False,
    shared_interactions: bool = False,
    displacement_init_gain: float = 1e-3,
    ) -> ComposableCore:
    if int(num_interactions) < 1:
        raise ValueError(f"num_interactions must be >= 1, got {num_interactions}")
    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if float(cutoff_sr) <= 0.0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if int(envelope_p) < 2:
        raise ValueError(
            f"envelope_p must be >= 2 so the force is continuous at the cutoff, got {envelope_p}"
        )
    variant_key = str(variant).lower()
    if variant_key not in {"paper", "repo"}:
        raise ValueError(f"variant must be 'paper' or 'repo', got {variant!r}")

    from ..newtonnet.interaction import (
        NewtonEmbed, NewtonHead, NewtonNetPaperInteraction, NewtonNetRepoInteraction,
        build_newtonnet, newton_paper_step, newton_repo_step,
    )

    n_blocks = 1 if shared_interactions else num_interactions
    if variant_key == "paper":
        blocks = [FnBlock(NewtonNetPaperInteraction(dim_embedding, num_rbf, displacement_init_gain),
                          newton_paper_step) for _ in range(n_blocks)]
    else:
        blocks = [FnBlock(NewtonNetRepoInteraction(dim_embedding, num_rbf), newton_repo_step)
                  for _ in range(n_blocks)]
    schedule = [0] * num_interactions if shared_interactions else list(range(n_blocks))
    if layer_norm:
        blocks.extend(ScalarNormBlock(nn.LayerNorm(dim_embedding)) for _ in range(num_interactions))
        schedule = [index for step, interaction in enumerate(schedule)
                    for index in (interaction, n_blocks + step)]
    return ComposableCore(
        NewtonEmbed(dim_embedding, num_rbf, cutoff_sr, envelope_p, variant_key), blocks,
        NewtonHead(variant_key), GraphRequires(optional_cutoff_values=False), schedule=schedule,
        positional_fields=_VECTOR_INPUTS, builder=build_newtonnet, architecture="newtonnet", variant=variant_key,
        num_interactions=num_interactions, **vector_metadata(dim_embedding),
    )


_VECTOR_INPUTS = ("atom_embedding", "rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr")
_ATTENTION_INPUTS = _VECTOR_INPUTS + ("cutoff_values_sr", "Za")


def _attention_embed(width, num_rbf, max_Za, cutoff, include_self, lmax, neighbors, edges):
    from ..et.interaction import ETNeighborEmbedding
    from ..mp.message.vector import VectorAttentionEmbed
    from ..visnet.interaction import EdgeEmbedding

    if width < 1 or num_rbf < 1 or max_Za < 1 or cutoff <= 0:
        raise ValueError("dim_embedding, num_rbf, max_Za and cutoff_sr must be positive")
    neighborhood = ETNeighborEmbedding(width, num_rbf, max_Za) if neighbors else None
    edge_embedding = EdgeEmbedding(num_rbf, width) if edges else None
    return VectorAttentionEmbed(width, num_rbf, max_Za, cutoff, include_self, lmax, neighborhood, edge_embedding)


def _attention_core(arch, embed, layers, head, num_heads, **attributes):
    from ..mp.message.vector import build_attention_stack, edge_attention_step, vector_attention_step

    step = vector_attention_step if embed.edge_embedding is None else edge_attention_step
    blocks = [FnBlock(layer, step) for layer in layers]
    return ComposableCore(
        embed, blocks, head, GraphRequires(), builder=build_attention_stack,
        positional_fields=_ATTENTION_INPUTS, architecture=arch, lmax=embed.lmax, angular_dim=embed.angular_dim,
        num_heads=num_heads, num_layers=len(blocks), **vector_metadata(embed.dim_embedding), **attributes,
    )


def et(
    dim_embedding: int = 128,
    num_rbf: int = 32,
    max_Za: int = 94,
    cutoff_sr: float = 5.0,
    num_layers: int = 6,
    num_heads: int = 8,
    distance_influence: str = "both",
    neighbor_embedding: bool = True,
    vector_cutoff: bool = True,
    include_self: bool = True,
    ) -> ComposableCore:
    from ..et.interaction import EquivariantMultiHeadAttention, normalize_distance_influence
    from ..mp.message.vector import AngularFeatureHead

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_heads < 1 or dim_embedding % num_heads:
        raise ValueError(f"dim_embedding ({dim_embedding}) must be divisible by num_heads ({num_heads})")

    influence = normalize_distance_influence(distance_influence)
    embed = _attention_embed(dim_embedding, num_rbf, max_Za, cutoff_sr, include_self, 1, neighbor_embedding, False)
    layers = [EquivariantMultiHeadAttention(dim_embedding, num_rbf, num_heads, influence, vector_cutoff)
              for _ in range(num_layers)]
    return _attention_core("et", embed, layers, AngularFeatureHead(dim_embedding, 1), num_heads)


def quinnet(
    dim_embedding: int = 256,
    num_rbf: int = 32,
    max_Za: int = 100,
    cutoff_sr: float = 5.0,
    num_layers: int = 5,
    num_heads: int = 8,
    distance_influence: str = "both",
    neighbor_embedding: bool = True,
    include_self: bool = True,
    activation: object = "silu",
    attn_activation: object = "silu",
    ) -> ComposableCore:
    from ..et.interaction import normalize_distance_influence
    from ..mp.message.vector import AngularFeatureHead
    from ..quinnet.interaction import QuinNetInteraction
    from ..visnet.interaction import normalize_activation

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_heads < 1 or dim_embedding % num_heads:
        raise ValueError(f"dim_embedding ({dim_embedding}) must be divisible by num_heads ({num_heads})")

    influence = normalize_distance_influence(distance_influence)
    normalize_activation(activation)
    normalize_activation(attn_activation)
    embed = _attention_embed(dim_embedding, num_rbf, max_Za, cutoff_sr, include_self, 2, neighbor_embedding, True)
    layers = [QuinNetInteraction(dim_embedding, num_heads, influence, last_layer=(i == num_layers - 1),
                                 activation=activation, attn_activation=attn_activation)
              for i in range(num_layers)]
    return _attention_core("quinnet", embed, layers, AngularFeatureHead(dim_embedding, 2), num_heads)


def _vis_parts(dim_embedding, num_rbf, max_Za, cutoff_sr, num_layers, num_heads,
               lmax, vertex_type, vecnorm_type, trainable_vecnorm, neighbor_embedding,
               include_self, activation, attn_activation, layer_type, **layer_params):
    from ..mp.message.vector import AngularFeatureHead
    from ..visnet.interaction import VecLayerNorm, normalize_activation, normalize_vecnorm_type, normalize_vertex_type

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_heads < 1 or dim_embedding % num_heads:
        raise ValueError(f"dim_embedding ({dim_embedding}) must be divisible by num_heads ({num_heads})")

    if lmax not in (1, 2):
        raise ValueError(f"lmax must be 1 or 2, got {lmax}")
    vertex = normalize_vertex_type(vertex_type)
    vecnorm = normalize_vecnorm_type(vecnorm_type)
    normalize_activation(activation)
    normalize_activation(attn_activation)
    embed = _attention_embed(dim_embedding, num_rbf, max_Za, cutoff_sr, include_self, lmax, neighbor_embedding, True)
    layers = [layer_type(dim_embedding, num_heads, vertex, vecnorm, trainable_vecnorm,
                         last_layer=(i == num_layers - 1), **layer_params) for i in range(num_layers)]
    head = AngularFeatureHead(dim_embedding, lmax, VecLayerNorm(dim_embedding, trainable=trainable_vecnorm,
                                                              norm_type=vecnorm))
    return embed, layers, head


def visnet(
    dim_embedding: int = 256,
    num_rbf: int = 32,
    max_Za: int = 100,
    cutoff_sr: float = 5.0,
    num_layers: int = 9,
    num_heads: int = 8,
    lmax: int = 2,
    vertex_type: object = "none",
    vecnorm_type: object = "none",
    trainable_vecnorm: bool = False,
    neighbor_embedding: bool = True,
    include_self: bool = True,
    activation: object = "silu",
    attn_activation: object = "silu",
    ) -> ComposableCore:
    from ..visnet.interaction import ViSInteraction

    embed, layers, head = _vis_parts(
        dim_embedding, num_rbf, max_Za, cutoff_sr, num_layers, num_heads, lmax,
        vertex_type, vecnorm_type, trainable_vecnorm, neighbor_embedding, include_self, activation,
        attn_activation, ViSInteraction,
    )
    return _attention_core("visnet", embed, layers, head, num_heads)


def freecg(
    dim_embedding: int = 256,
    num_rbf: int = 32,
    max_Za: int = 100,
    cutoff_sr: float = 5.0,
    num_layers: int = 9,
    num_heads: int = 8,
    lmax: int = 2,
    vertex_type: object = "none",
    vecnorm_type: object = "none",
    trainable_vecnorm: bool = False,
    neighbor_embedding: bool = True,
    include_self: bool = True,
    activation: object = "silu",
    attn_activation: object = "silu",
    num_groups: int = 8,
    shuffle_scale: float = 1.5,
    attention_enhancer: object = "linear",
    ) -> ComposableCore:
    from ..freecg.interaction import FreeCGInteraction, normalize_attention_enhancer

    if int(lmax) != 2:
        raise ValueError(
            "FreeCG's sparse paths are 1o and 2e, so lmax must be 2, "
            f"got {lmax}"
        )
    groups = int(num_groups)
    if groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if int(dim_embedding) % groups != 0:
        raise ValueError(
            f"dim_embedding ({dim_embedding}) must be divisible by "
            f"num_groups ({groups})"
        )
    if float(shuffle_scale) < 0:
        raise ValueError(f"shuffle_scale must be >= 0, got {shuffle_scale}")
    enhancer = normalize_attention_enhancer(attention_enhancer)

    embed, layers, head = _vis_parts(
        dim_embedding, num_rbf, max_Za, cutoff_sr, num_layers, num_heads, lmax,
        vertex_type, vecnorm_type, trainable_vecnorm, neighbor_embedding, include_self, activation,
        attn_activation, FreeCGInteraction, num_groups=groups, shuffle_scale=shuffle_scale, attention_enhancer=enhancer,
    )
    return _attention_core("freecg", embed, layers, head, num_heads, num_groups=groups,
                           channels_per_group=dim_embedding // groups)


def visnet_lsrm(
    dim_embedding: int = 128,
    num_rbf: int = 50,
    max_Za: int = 100,
    cutoff_sr: float = 4.0,
    num_layers: int = 6,
    long_layers: int = 2,
    num_heads: int = 8,
    lmax: int = 1,
    vertex_type: object = "none",
    vecnorm_type: object = "max_min",
    trainable_vecnorm: bool = False,
    neighbor_embedding: bool = True,
    include_self: bool = False,
    activation: object = "silu",
    attn_activation: object = "silu",
    cutoff_lr: float = 9.0,
    fragment_mode: object = "kmeans",
    center: object = "atomic_number",
    min_nodes_per_group: int = 24,
    kmeans_random_state: int = 0,
    dropout: float = 0.0,
    trainable_rbf_lr: bool = True,
    ) -> ComposableCore:
    from ..mp.message.vector import edge_attention_step, build_attention_stack
    from ..visnet.interaction import ViSInteraction, _xavier_linear
    from ..visnet.lsr import (
        FragmentBipartiteBlock, FragmentEdges, FragmentFusionHead, fragment_step,
        normalize_center, normalize_fragment_mode,
    )

    if int(long_layers) < 0:
        raise ValueError(f"long_layers must be >= 0, got {long_layers}")
    if float(cutoff_lr) <= 0.0:
        raise ValueError(f"cutoff_lr must be positive, got {cutoff_lr}")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
    mode = normalize_fragment_mode(fragment_mode)
    center_name = normalize_center(center)

    embed, layers, head = _vis_parts(
        dim_embedding, num_rbf, max_Za, cutoff_sr, num_layers, num_heads, lmax,
        vertex_type, vecnorm_type, trainable_vecnorm, neighbor_embedding, include_self, activation,
        attn_activation, ViSInteraction,
    )
    graph = FragmentEdges(num_rbf, dim_embedding, cutoff_lr, trainable_rbf_lr, mode, center_name,
                          min_nodes_per_group, kmeans_random_state, long_layers > 0)
    blocks = [FnBlock(layer, edge_attention_step) for layer in layers] + [graph]
    blocks.extend(FnBlock(FragmentBipartiteBlock(dim_embedding, num_heads, dropout), fragment_step)
                  for _ in range(long_layers))
    head = FragmentFusionHead(head, dim_embedding)
    _xavier_linear(graph.long_rbf_proj)
    return ComposableCore(
        embed, blocks, head, GraphRequires(), builder=build_attention_stack,
        positional_fields=_ATTENTION_INPUTS + ("Ra", "batch_seg", "cluster_ids"),
        architecture="visnet_lsrm", lmax=lmax, angular_dim=embed.angular_dim, num_heads=num_heads,
        num_layers=num_layers, dim_embedding=dim_embedding, dim_feature_out=2*dim_embedding,
        feature_irreps=f"{2*dim_embedding}x0e", vector_feature_layout="cartesian_n3f",
    )
