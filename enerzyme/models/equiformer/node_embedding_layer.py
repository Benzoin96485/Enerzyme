# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License).
"""Pre-core Equiformer node embedding as an Enerzyme layer."""

from typing import Optional

from torch import Tensor

from ..layers._base_layer import BaseFFLayer
from .embedding import NodeEmbeddingNetwork


class EquiformerNodeEmbedding(BaseFFLayer):
    """Map atomic numbers ``Za`` to Equiformer irreps node embeddings.

    Unlike scalar :class:`~enerzyme.models.layers.atom_embedding.NuclearEmbedding`,
    this produces an e3nn irreps feature used by :class:`EquiformerCore`.
    """

    def __init__(
        self,
        max_Za: int,
        irreps_node_embedding: str = "128x0e+64x1e+32x2e",
        bias: bool = True,
    ) -> None:
        super().__init__(input_fields={"Za"}, output_fields={"atom_embedding"})
        self.max_Za = max_Za
        self.irreps_node_embedding = irreps_node_embedding
        self.embed = NodeEmbeddingNetwork(
            irreps_node_embedding,
            max_atom_type=max_Za + 1,
            bias=bias,
        )

    def get_atom_embedding(self, Za: Tensor) -> Tensor:
        embedding, _, _ = self.embed(Za.long())
        return embedding
