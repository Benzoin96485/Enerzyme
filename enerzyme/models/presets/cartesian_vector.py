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
        architecture="painn", num_interactions=num_interactions, shared_filters=shared_filters,
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
        builder=build_newtonnet, architecture="newtonnet", variant=variant_key,
        num_interactions=num_interactions, **vector_metadata(dim_embedding),
    )
