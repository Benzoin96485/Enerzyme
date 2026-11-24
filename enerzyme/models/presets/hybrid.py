"""Dual-stream local and nonlocal message-passing presets."""

from __future__ import annotations

from typing import List, Optional, Sequence

from torch.nn import Identity, SiLU

from ..allscaip.equivariant_input import EquivariantInputBlock
from ..allscaip.graph_blocks import GraphAttentionBlock, InputBlock
from ..allscaip.graph_preprocess import AllScAIPGraphHParams, _default_freequency_list, build_graph_attention_data
from ..mp.basis.angular import RealSphericalHarmonics
from ..mp.basis.envelope import CUTOFF_REGISTER, CUTOFF_KEY_TYPE
from ..mp.block import BlockState
from ..mp.core import ComposableCore, Embed
from ..mp.graph import GraphRequires
from ..mp.head import FeatureHead, Head
from ..mp.nonlocal_ import build_efa_blocks, parse_era_iterations
from ..mp.ops import scatter_sum
from ..so3krates.defaults import _AVG_NUM_NEIGHBORS
from ..so3krates.interaction import EuclideanTransformer


class SphericalAttentionEmbed(Embed):
    input_fields = ("atom_embedding",)

    def __init__(self, degrees, cutoff_fn, radius, zero_init, needs_positions):
        super().__init__()
        self.harmonics = RealSphericalHarmonics(degrees)
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
        self.radius = radius
        self.zero_init = zero_init
        self.needs_positions = needs_positions

    def forward(self, inputs, ctx):
        scalar = inputs["atom_embedding"]
        harmonics = self.harmonics(-ctx.vec)
        cutoff = self.cutoff_fn(ctx.dist, self.radius).to(scalar.dtype)
        if self.zero_init:
            sphere = scalar.new_zeros(ctx.num_nodes, self.harmonics.m_tot)
        else:
            sphere = scatter_sum(cutoff[:, None] * harmonics, ctx.receiver, ctx.num_nodes)
        aux = {"harmonics": harmonics, "cutoff": cutoff}
        if self.needs_positions:
            pos = inputs.get("Ra")
            if pos is None:
                raise ValueError("Ra is required when era_use_in_iterations is set")
            aux["positions"] = pos.to(scalar.dtype)
            aux["batch"] = ctx.batch if ctx.batch is not None else ctx.receiver.new_zeros(ctx.num_nodes)
        return BlockState(scalar=scalar, equi=sphere, aux=aux)


class SphericalAttentionBlock(EuclideanTransformer):
    def forward(self, state, ctx):
        scalar, sphere = super().forward(
            state.scalar, state.equi, rbf=ctx.rbf, senders=ctx.sender, receivers=ctx.receiver,
            sh_vectors=state.aux["harmonics"], cutoffs=state.aux["cutoff"],
        )
        if self.nonlocal_update is not None:
            scalar = scalar + self.nonlocal_update(scalar, state.aux["positions"], state.aux["batch"])
        return BlockState(scalar=scalar, equi=sphere, aux=state.aux)


def so3krates(
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
) -> ComposableCore:
    era_iters = parse_era_iterations(era_use_in_iterations)
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

    avg_num_neighbors = _AVG_NUM_NEIGHBORS if avg_num_neighbors is None else float(avg_num_neighbors)
    if qk_non_linearity.lower() in {"identity", "none", "linear"}:
        qk_act = Identity
    elif qk_non_linearity.lower() == "silu":
        qk_act = SiLU
    else:
        raise ValueError(f"Unsupported qk_non_linearity: {qk_non_linearity}")

    layers = [
        SphericalAttentionBlock(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            num_radial_basis_fn=num_rbf,
            activation_fn=SiLU,
            interaction_bias=interaction_bias,
            message_normalization=message_normalization,
            avg_num_neighbors=avg_num_neighbors,
            layer_normalization_1=layer_normalization_1,
            layer_normalization_2=layer_normalization_2,
            residual_mlp_1=residual_mlp_1,
            residual_mlp_2=residual_mlp_2,
            qk_non_linearity=qk_act,
        )
        for _ in range(num_layers)
    ]

    qk = era_qk_num_features if era_qk_num_features is not None else min(32, num_features)
    if qk % 2 != 0:
        qk += 1
    nonlocal_blocks = build_efa_blocks(
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
    active = set(era_iters or ())
    for index, layer in enumerate(layers):
        layer.nonlocal_update = nonlocal_blocks[index] if index in active else None
    embed = SphericalAttentionEmbed(degrees, cutoff_fn, cutoff_sr, initialize_ev_to_zeros, era_iters is not None)
    return ComposableCore(
        embed, layers, FeatureHead(equi_field="atom_sphere_feature"),
        GraphRequires(optional_cutoff_values=False, fields=("Ra", "batch_seg") if era_iters is not None else ()),
        positional_fields=("atom_embedding", "rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "Ra", "batch_seg"),
        architecture="so3krates", dim_embedding=dim_embedding, num_rbf=num_rbf, max_Za=max_Za,
        dim_feature_out=num_features, feature_irreps=f"{num_features}x0e", degrees=degrees,
        num_features=num_features, num_heads=num_heads, num_layers=num_layers, cutoff_sr=cutoff_sr,
        cutoff_fn_name=cutoff_fn, message_normalization=message_normalization, avg_num_neighbors=avg_num_neighbors,
        era_enabled=era_iters is not None, initialize_ev_to_zeros=initialize_ev_to_zeros, m_tot=embed.harmonics.m_tot,
    )


class NeighborEmbed(Embed):
    input_fields = ("atom_embedding", "charge_embedding", "spin_embedding", "Za", "Ra", "batch_seg")

    def __init__(self, input_block, graph_params):
        super().__init__()
        self.input_block = input_block
        self.graph_params = graph_params

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        for field in ("charge_embedding", "spin_embedding"):
            if inputs.get(field) is not None:
                node = node + inputs[field]
        graph = build_graph_attention_data(
            inputs["Ra"], inputs["Za"], ctx.batch, hparams=self.graph_params,
            charge=None, spin=None, cell=None,
        )
        return BlockState(edge=self.input_block(graph, node), aux={"graph": graph})


class NeighborAttentionBlock(GraphAttentionBlock):
    def forward(self, state, ctx):
        edge = super().forward(state.aux["graph"], state.edge)
        return BlockState(edge=edge, aux=state.aux)


class NeighborHead(Head):
    def forward(self, state, ctx, taps, inputs):
        node = state.edge[:, 0]
        mask = state.aux["graph"].node_padding_mask.to(node.dtype).unsqueeze(-1)
        return {"atom_feature": node * mask}


def allscaip(
    dim_embedding: int,
    num_layers: int = 4,
    atten_num_heads: int = 4,
    ffn_hidden_layer_multiplier: int = 2,
    mlp_dropout: float = 0.0,
    atten_dropout: float = 0.0,
    use_node_path: bool = True,
    use_residual_scaling: bool = True,
    activation: str = "silu",
    normalization: str = "layer_norm",
    cutoff_sr: float = 5.0,
    knn_k: int = 32,
    knn_pad_size: Optional[int] = None,
    use_envelope: bool = True,
    distance_function: str = "gaussian",
    edge_distance_expansion_size: int = 8,
    edge_direction_expansion_size: int = 3,
    node_direction_expansion_size: int = 3,
    use_freq_mask: bool = True,
    freequency_list: Optional[List[int]] = None,
    use_sincx_mask: bool = True,
    attn_num_freq: int = 32,
    preprocess_on_cpu: bool = False,
    pbc: bool = False,
    use_padding: bool = False,
    max_atoms: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    single_system_no_padding: bool = False,
    knn_soft: bool = False,
    knn_sigmoid_scale: float = 0.2,
    knn_lse_scale: float = 0.1,
    knn_use_low_mem: bool = False,
    equivariant_mode: bool = False,
    equivariant_tp_hidden: int = 32,
    equivariant_tp_mul: int = 8,
) -> ComposableCore:
    lmax_edge = edge_direction_expansion_size - 1
    if equivariant_mode:
        input_block = EquivariantInputBlock(
            hidden_size=dim_embedding,
            tp_hidden=equivariant_tp_hidden,
            lmax_edge=lmax_edge,
            edge_distance_expansion_size=edge_distance_expansion_size,
            tp_mul_per_l=equivariant_tp_mul,
            activation=activation,
            normalization=normalization,
        )
    else:
        edge_dir_sh_dim = edge_direction_expansion_size * edge_direction_expansion_size
        input_block = InputBlock(
            hidden_size=dim_embedding,
            node_direction_expansion_size=node_direction_expansion_size,
            edge_distance_expansion_size=edge_distance_expansion_size,
            edge_direction_sh_dim=edge_dir_sh_dim,
            activation=activation,
            normalization=normalization,
        )
    head_dim = dim_embedding // atten_num_heads
    frequencies: List[int] = (
        list(freequency_list)
        if freequency_list is not None
        else _default_freequency_list(head_dim, lmax_edge)
    )
    frequency_mask = bool(use_freq_mask) and not equivariant_mode

    blocks = [
        NeighborAttentionBlock(
            hidden_size=dim_embedding,
            num_layers=num_layers,
            use_node_path=use_node_path,
            attn_num_heads=atten_num_heads,
            attn_num_freq=attn_num_freq,
            atten_dropout=atten_dropout,
            mlp_dropout=mlp_dropout,
            activation=activation,
            ffn_hidden_layer_multiplier=ffn_hidden_layer_multiplier,
            use_freq_mask=frequency_mask,
            freequency_list=list(frequencies),
            use_sincx_mask=use_sincx_mask,
            use_residual_scaling=use_residual_scaling,
            normalization=normalization,
        )
        for _ in range(num_layers)
    ]
    params = AllScAIPGraphHParams(
        hidden_size=dim_embedding,
        max_radius=cutoff_sr,
        knn_k=knn_k,
        knn_soft=knn_soft,
        knn_sigmoid_scale=knn_sigmoid_scale,
        knn_lse_scale=knn_lse_scale,
        knn_use_low_mem=knn_use_low_mem,
        knn_pad_size=knn_pad_size,
        use_envelope=use_envelope,
        distance_function=distance_function,
        edge_distance_expansion_size=edge_distance_expansion_size,
        edge_direction_expansion_size=edge_direction_expansion_size,
        node_direction_expansion_size=node_direction_expansion_size,
        atten_num_heads=atten_num_heads,
        freequency_list=frequencies,
        use_freq_mask=frequency_mask,
        use_node_path=use_node_path,
        use_sincx_mask=use_sincx_mask,
        attn_num_freq=attn_num_freq,
        single_system_no_padding=single_system_no_padding,
        use_padding=use_padding,
        max_atoms=max_atoms,
        max_batch_size=max_batch_size,
        preprocess_on_cpu=preprocess_on_cpu,
        pbc=pbc,
    )
    return ComposableCore(
        NeighborEmbed(input_block, params), blocks, NeighborHead(),
        GraphRequires(edge_set="internal", vectors=False, distances=False, rbf=False, optional_cutoff_values=False),
        positional_fields=("Ra", "Za", "batch_seg", "atom_embedding", "charge_embedding", "spin_embedding"),
        architecture="allscaip", dim_embedding=dim_embedding, dim_feature_out=dim_embedding,
        num_layers=num_layers, atten_num_heads=atten_num_heads, equivariant_mode=equivariant_mode,
        cutoff_sr=cutoff_sr, knn_k=knn_k,
    )
