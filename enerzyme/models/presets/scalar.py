"""Invariant message-passing presets: SchNet, PhysNet, SpookyNet.

These three load ``main``-branch checkpoints through
:mod:`enerzyme.models.compat.main_branch`, so their Core parameter names and
numerics follow ``main`` exactly.
"""

from __future__ import annotations

from typing import List, Literal

from torch import nn
from torch.nn import Linear, init

from ..mp.basis.envelope import Envelope
from ..mp.block import InteractionBlock
from ..mp.core import ComposableCore, FeatureHead, FieldEmbed, ReadoutHead
from ..mp.graph import GraphRequires
from ..mp.message import FilterMessage
from ..mp.nn.activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, get_activation_fn
from ..mp.nn.mlp import DenseLayer
from ..mp.update import ResidualUpdate

SCHNET_BANNER = """
######################################################
# Augmented SchNet (NeurIPS 2017, arXiv: 1706.08566) #
######################################################
"""

PHYSNET_BANNER = """
###################################################################################
# Pytorch Implementation of PhysNet (J. Chem. Theory Comput. 2019, 15, 3678−3693) #
###################################################################################
"""

SPOOKYNET_BANNER = """
###############################################
# SpookyNet (Nat. Commun., 2021, 12(1): 7273) #
###############################################
"""


def _xavier(*layers: Linear) -> None:
    for layer in layers:
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0)


def schnet_block(hidden_channels: int, num_rbf: int, num_filters: int, cutoff: float) -> InteractionBlock:
    """CFConv with a cosine-cutoff filter, then ``x + lin(ssp(lin2(·)))``."""
    filter_net = nn.Sequential(
        Linear(num_rbf, num_filters), get_activation_fn("shifted_softplus", {}), Linear(num_filters, num_filters),
    )
    pre = Linear(hidden_channels, num_filters, bias=False)
    post = nn.Sequential(
        Linear(num_filters, hidden_channels), get_activation_fn("shifted_softplus", {}),
        Linear(hidden_channels, hidden_channels),
    )
    _xavier(filter_net[0], filter_net[2], pre, post[0], post[2])
    return InteractionBlock(
        message=FilterMessage(filter_net, pre=pre, envelope=Envelope("cosine", cutoff)),
        update=ResidualUpdate(post),
    )


def schnet(
    hidden_channels: int = 128,
    dim_embedding: int = 128,
    num_interactions: int = 4,
    num_rbf: int = 128,
    cutoff_sr: float = 5.0,
    activation_fn: ACTIVATION_KEY_TYPE = "shifted_softplus",
    activation_params: ACTIVATION_PARAM_TYPE = dict(),
    shallow_ensemble_size: int = 1,
    output_mode: Literal["direct", "feature"] = "direct",
) -> ComposableCore:
    """SchNet: ``num_interactions`` CFConv blocks; messages flow ``idx_i → idx_j`` as in PyG."""
    blocks = [schnet_block(hidden_channels, num_rbf, dim_embedding, cutoff_sr) for _ in range(num_interactions)]
    if output_mode == "direct":
        lin1 = Linear(hidden_channels, hidden_channels // 2)
        _xavier(lin1)
        readout = nn.Sequential(
            lin1, get_activation_fn(activation_fn, activation_params),
            DenseLayer(hidden_channels // 2, 2, initial_weight="xavier_uniform", initial_bias="zero",
                       shallow_ensemble_size=shallow_ensemble_size),
        )
        head = ReadoutHead(readout, mode="final", ensemble_size=shallow_ensemble_size)
    else:
        head = FeatureHead("feature")
    requires = GraphRequires(vectors=False, edge_convention="swap", optional_cutoff_values=False)
    return ComposableCore(
        FieldEmbed("atom_embedding"), blocks, head, requires,
        banner=SCHNET_BANNER, architecture="schnet", output_mode=output_mode,
        shallow_ensemble_size=shallow_ensemble_size, hidden_channels=hidden_channels,
        dim_feature_out=hidden_channels,
    )


def physnet(
    dim_embedding: int,
    num_rbf: int,
    num_blocks: int = 3,
    num_residual_atomic: int = 2,
    num_residual_interaction: int = 2,
    num_residual_output: int = 1,
    activation_fn: ACTIVATION_KEY_TYPE = "shifted_softplus",
    activation_params: ACTIVATION_PARAM_TYPE = dict(),
    dropout_rate: float = 0.0,
    shallow_ensemble_size: int = 1,
    output_mode: Literal["direct", "feature"] = "direct",
) -> ComposableCore:
    """PhysNet: gated interaction + residual stacks, one output block per module, NH loss."""
    from ..physnet.interaction import InteractionBlock as PhysNetBlock
    from ..physnet.interaction import OutputBlock

    blocks = [
        PhysNetBlock(num_rbf, dim_embedding, num_residual_atomic, num_residual_interaction,
                     activation_fn=activation_fn, activation_params=activation_params, dropout_rate=dropout_rate)
        for _ in range(num_blocks)
    ]
    if output_mode == "direct":
        readouts = [
            OutputBlock(dim_embedding, num_residual_output, activation_fn=activation_fn,
                        activation_params=activation_params, dropout_rate=dropout_rate,
                        shallow_ensemble_size=shallow_ensemble_size)
            for _ in range(num_blocks)
        ]
        head = ReadoutHead(readouts=readouts, mode="per_block", ensemble_size=shallow_ensemble_size,
                           nh_loss=True, l2_penalty=True)
    else:
        head = FeatureHead("layers")
    requires = GraphRequires(vectors=False, distances=False, optional_cutoff_values=False)
    return ComposableCore(
        FieldEmbed("atom_embedding"), blocks, head, requires,
        banner=PHYSNET_BANNER, architecture="physnet", output_mode=output_mode,
        num_blocks=num_blocks, shallow_ensemble_size=shallow_ensemble_size,
        dim_feature_out=dim_embedding, num_output_blocks=num_blocks,
    )


def _spookynet_builder(core: ComposableCore, built_layers: List[nn.Module]) -> None:
    from ..layers import GatherAtomEmbedding

    core.adopt_radius_stack(built_layers)
    core.pre_sequence.append(GatherAtomEmbedding())


def spookynet(
    dim_embedding: int, num_rbf: int, num_modules: int, num_residual_pre: int,
    num_residual_local_x: int, num_residual_local_s: int, num_residual_local_p: int,
    num_residual_local_d: int, num_residual_local: int,
    num_residual_nonlocal_q: int, num_residual_nonlocal_k: int, num_residual_nonlocal_v: int,
    num_residual_post: int, num_residual_output: int, activation_fn: ACTIVATION_KEY_TYPE,
    use_irreps: bool, dropout_rate: float = 0.0,
    shallow_ensemble_size: int = 1,
    output_mode: Literal["direct", "feature"] = "direct",
    use_efa: bool = False,
    efa_lebedev_num: int = 146,
    efa_max_frequency: float = None,
    efa_max_length: float = 10.0,
    efa_num_features_qk: int = None,
    efa_num_features_v: int = None,
) -> ComposableCore:
    """SpookyNet: local s/p/d message passing plus batch-masked attention (or EFA) per module."""
    from ..spookynet.interaction import InteractionModule

    blocks = [
        InteractionModule(
            dim_embedding=dim_embedding, num_rbf=num_rbf, num_residual_pre=num_residual_pre,
            num_residual_local_x=num_residual_local_x, num_residual_local_s=num_residual_local_s,
            num_residual_local_p=num_residual_local_p, num_residual_local_d=num_residual_local_d,
            num_residual_local=num_residual_local, num_residual_nonlocal_q=num_residual_nonlocal_q,
            num_residual_nonlocal_k=num_residual_nonlocal_k, num_residual_nonlocal_v=num_residual_nonlocal_v,
            num_residual_post=num_residual_post, num_residual_output=num_residual_output,
            activation_fn=activation_fn, use_irreps=use_irreps, use_efa=use_efa,
            efa_lebedev_num=efa_lebedev_num, efa_max_frequency=efa_max_frequency,
            efa_max_length=efa_max_length, efa_num_features_qk=efa_num_features_qk,
            efa_num_features_v=efa_num_features_v,
        )
        for _ in range(num_modules)
    ]
    keep_prob = 1 - dropout_rate
    if output_mode == "direct":
        if shallow_ensemble_size > 1:
            readout = DenseLayer(dim_embedding, 2, use_bias=False, shallow_ensemble_size=shallow_ensemble_size)
        else:
            readout = Linear(dim_embedding, 2, bias=False)
        head = ReadoutHead(readout, mode="sum_taps", source="y", ensemble_size=shallow_ensemble_size,
                           block_keep_prob=keep_prob)
    else:
        head = FeatureHead("sum", source="y")
    requires = GraphRequires(optional_cutoff_values=False, optional_fields=("batch_seg",) + (("Ra",) if use_efa else ()))
    return ComposableCore(
        FieldEmbed("atom_embedding"), blocks, head, requires,
        builder=_spookynet_builder, banner=SPOOKYNET_BANNER, architecture="spookynet",
        output_mode=output_mode, use_irreps=use_irreps, use_efa=bool(use_efa),
        module_keep_prob=keep_prob, shallow_ensemble_size=shallow_ensemble_size, dim_feature_out=dim_embedding,
    )
