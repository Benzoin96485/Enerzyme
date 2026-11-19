"""MatterSim Core: M3GNet extended by the materials Graphormer encoder.

Yang et al., arXiv:2405.04967. The released `microsoft/mattersim` package
implements M3GNet only; its Graphormer path raises ``NotImplementedError``.
This Core keeps Chen and Ong's three-body convolution and adds the
structural encoder from supplementary section S1.3: centrality on the
species embedding, then cutoff-masked attention with a Gaussian distance
bias. The Gaussian basis and the spherical Bessel basis stay outside.
The gated energy head, elemental references, and ``Fa = -∇E`` stay outside.

The paper's GeoMFormer decoder predicts forces from an equivariant stream,
and a separate head predicts stress from the lattice. Those heads are not
part of this conservative potential.
"""

from __future__ import annotations

from typing import List, Optional

from torch import Tensor
from torch.nn import Module, ModuleList

from ..cutoff import polynomial_envelope
from ..layers.rbf import BaseRBF, GaussianRBFLayer
from ..m3gnet.core import M3GNetCore
from .attention import CentralityEncoding, GraphormerBlock


class MatterSimCore(M3GNetCore):
    """M3GNet blocks with a Graphormer centrality encoding and attention.

    Emits ``atom_feature`` ``(N, dim_embedding)`` and
    ``feature_irreps = "{F}x0e"``. With ``num_attn_layers = 0`` and
    ``use_centrality = false`` the forward pass is M3GNet. Otherwise each
    local block is followed by one attention block, and any extra attention
    blocks run after the local stack. An atom with no neighbors still passes
    through the attention feed-forward, which depends only on its species
    embedding.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        max_l: int = 3,
        num_blocks: int = 3,
        threebody_cutoff: float = 4.0,
        units: Optional[int] = None,
        n_conv_hidden: int = 2,
        dim_edge_embedding: Optional[int] = None,
        dropout: float = 0.0,
        envelope_exponent: int = 3,
        num_attn_layers: Optional[int] = None,
        num_heads: int = 4,
        graphormer_num_basis: int = 16,
        attention_cutoff: Optional[float] = None,
        ffn_dim: Optional[int] = None,
        use_centrality: bool = True,
    ) -> None:
        super().__init__(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            cutoff_sr=cutoff_sr,
            max_l=max_l,
            num_blocks=num_blocks,
            threebody_cutoff=threebody_cutoff,
            units=units,
            n_conv_hidden=n_conv_hidden,
            dim_edge_embedding=dim_edge_embedding,
            dropout=dropout,
            envelope_exponent=envelope_exponent,
        )
        if num_attn_layers is None:
            num_attn_layers = self.num_blocks
        if int(num_attn_layers) < 0:
            raise ValueError(
                f"num_attn_layers must be >= 0, got {num_attn_layers}"
            )
        if int(num_heads) < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if self.dim_embedding % int(num_heads) != 0:
            raise ValueError(
                f"dim_embedding ({self.dim_embedding}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if int(graphormer_num_basis) < 2:
            raise ValueError(
                "graphormer_num_basis must be >= 2, "
                f"got {graphormer_num_basis}"
            )
        resolved_cutoff = (
            self.cutoff_sr if attention_cutoff is None else float(attention_cutoff)
        )
        if resolved_cutoff <= 0:
            raise ValueError(
                f"attention_cutoff must be positive, got {resolved_cutoff}"
            )
        if resolved_cutoff > self.cutoff_sr + 1e-6:
            raise ValueError(
                "attention_cutoff "
                f"({resolved_cutoff}) cannot exceed cutoff_sr ({self.cutoff_sr}): "
                "edges outside the neighbor list are not available"
            )
        resolved_ffn = self.dim_embedding if ffn_dim is None else int(ffn_dim)
        if resolved_ffn < 1:
            raise ValueError(f"ffn_dim must be >= 1, got {ffn_dim}")
        self.num_attn_layers = int(num_attn_layers)
        self.num_heads = int(num_heads)
        self.graphormer_num_basis = int(graphormer_num_basis)
        self.attention_cutoff = float(resolved_cutoff)
        self.ffn_dim = int(resolved_ffn)
        self.use_centrality = bool(use_centrality)
        self.uses_graphormer_basis = self.use_centrality or self.num_attn_layers > 0
        if self.use_centrality:
            self.centrality = CentralityEncoding(
                self.graphormer_num_basis, self.dim_embedding
            )
        self.attention = ModuleList(
            [
                GraphormerBlock(
                    dim=self.dim_embedding,
                    num_heads=self.num_heads,
                    num_basis=self.graphormer_num_basis,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_attn_layers)
            ]
        )
        if self.uses_graphormer_basis:
            self._input_fields.add("graphormer_rbf")
            self._relevant_fields.add("graphormer_rbf")
            self._name_mapping["graphormer_rbf"] = "graphormer_rbf"

    def __str__(self) -> str:
        return """
#########################################################################
# MatterSim (Yang et al., arXiv:2405.04967)                            #
# M3GNet three-body blocks + Graphormer centrality and attention.      #
# Gaussian basis, readout, and conservative forces stay outside.       #
#########################################################################
"""

    def _allow_extra_radial(self) -> bool:
        return self.uses_graphormer_basis

    def build(self, built_layers: List[Module]) -> None:
        super().build(built_layers)
        if not self.uses_graphormer_basis:
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
        if int(layer.num_rbf) != self.graphormer_num_basis:
            raise ValueError(
                "GaussianRBF num_rbf "
                f"({layer.num_rbf}) must equal graphormer_num_basis "
                f"({self.graphormer_num_basis})"
            )
        if abs(float(layer.cutoff_sr) - self.cutoff_sr) > 1e-4:
            raise ValueError(
                "GaussianRBF cutoff_sr "
                f"({layer.cutoff_sr}) must equal MatterSim cutoff_sr "
                f"({self.cutoff_sr})"
            )
        layer.reset_field_name(rbf="graphormer_rbf")
        self.graphormer_basis = layer

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        graphormer_rbf: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if not self.uses_graphormer_basis:
            return super().get_output(
                atom_embedding, rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr
            )
        if graphormer_rbf is None:
            raise ValueError(
                "graphormer_rbf is required when centrality or attention is enabled"
            )
        n_edges = int(Dij_sr.shape[0])
        if (
            graphormer_rbf.ndim != 2
            or graphormer_rbf.shape[0] != n_edges
            or graphormer_rbf.shape[-1] != self.graphormer_num_basis
        ):
            raise ValueError(
                "graphormer_rbf must be "
                f"(E, {self.graphormer_num_basis}), got {tuple(graphormer_rbf.shape)} "
                f"for {n_edges} edges"
            )
        node, edge, pairs, joint, envelope = self._featurize(
            atom_embedding, rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr
        )
        if self.use_centrality:
            node = node + self.centrality(
                graphormer_rbf,
                Dij_sr,
                idx_i_sr,
                int(node.shape[0]),
                self.attention_cutoff,
            )
        mask = polynomial_envelope(Dij_sr, self.attention_cutoff, 3)
        n_local = len(self.blocks)
        n_attn = len(self.attention)
        for step in range(max(n_local, n_attn)):
            if step < n_local:
                node, edge = self.blocks[step](
                    node,
                    edge,
                    rbf,
                    idx_i_sr,
                    idx_j_sr,
                    pairs.idx_ij,
                    pairs.idx_ik,
                    joint,
                    envelope,
                )
            if step < n_attn:
                node = self.attention[step](
                    node, graphormer_rbf, mask, idx_i_sr, idx_j_sr
                )
        return {"atom_feature": node}
