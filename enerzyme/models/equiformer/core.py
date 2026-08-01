# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License)
# and adapted to Enerzyme BaseFFCore modular layer stack.
"""Equiformer Core: equivariant graph attention transformer latent."""

from typing import Dict, List, Literal, Optional

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..e3nn_nn import extract_scalar_0e, scalar_0e_dim
from ..layers._base_layer import BaseFFCore
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.electrostatics import ChargeConservationLayer
from ..layers.geometry import DistanceLayer, RangeSeparationLayer
from ..layers.rbf import BaseRBF
from ..e3nn_nn import EquivariantDropout
from .embedding import EdgeDegreeEmbeddingNetwork, ScaledScatter, EquiformerNodeEmbedding
from ..e3nn_nn import Activation, LinearRS
from .interaction import GraphAttention, TransBlock, get_norm_layer, _RESCALE

# MD17-style statistics (cutoff ~5 A); used for ScaledScatter normalization.
_AVG_DEGREE = 15.57930850982666
_AVG_NUM_NODES = 18.03065905448718

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 16,
    "max_Za": 86,
    "cutoff_sr": 5.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    "irreps_node_embedding": "32x0e+16x1e+8x2e",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation", "params": {"cutoff_fn": "polynomial"}},
    {
        "name": "ExpNormalSmearing",
        "params": {"trainable": False},
    },
    {
        "name": "EquiformerNodeEmbedding",
        "params": {
            "irreps_node_embedding": "32x0e+16x1e+8x2e",
        },
    },
    {
        "name": "Core",
        "params": {
            "irreps_node_embedding": "32x0e+16x1e+8x2e",
            "irreps_feature": "64x0e",
            "irreps_sh": "1x0e+1x1e+1x2e",
            "irreps_head": "16x0e+8x1o+4x2e",
            "irreps_mlp_mid": "32x0e+16x1e+8x2e",
            "num_layers": 2,
            "num_heads": 2,
            "fc_neurons": [64, 64],
            "nonlinear_message": True,
            "use_attn_head": False,
            "alpha_drop": 0.0,
            "proj_drop": 0.0,
            "out_drop": 0.0,
            "drop_path_rate": 0.0,
            "avg_degree": _AVG_DEGREE,
            "avg_num_nodes": _AVG_NUM_NODES,
            "output_mode": "feature",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea", "Qa"],
            "head_type": "two_layer",
            "dim_embedding": 64,
            "activation_fn": "swish",
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {
                "Ea": {"values": 0, "learnable": True},
                "Qa": {"values": 0, "learnable": True},
            },
            "scales": {
                "Ea": {"values": 1, "learnable": True},
                "Qa": {"values": 1, "learnable": True},
            },
        },
    },
    {"name": "ChargeConservation"},
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class EquiformerCore(BaseFFCore):
    """Equivariant Graph Attention Transformer Core (ICLR 2023).

    Consumes pre-core geometry (``vij_sr``, indices, ``rbf``) and irreps
    ``atom_embedding`` from :class:`EquiformerNodeEmbedding`, runs EdgeDegree
    embedding + TransBlocks, and emits either scalar ``atom_feature`` or
    atomic energy ``Ea`` (``output_mode``).
    """

    def __init__(
        self,
        num_rbf: int,
        irreps_node_embedding: str = "32x0e+16x1e+8x2e",
        irreps_feature: str = "64x0e",
        irreps_node_attr: str = "1x0e",
        irreps_sh: str = "1x0e+1x1e+1x2e",
        irreps_head: str = "16x0e+8x1o+4x2e",
        irreps_mlp_mid: str = "32x0e+16x1e+8x2e",
        irreps_pre_attn: Optional[str] = None,
        num_layers: int = 2,
        num_heads: int = 2,
        fc_neurons: Optional[List[int]] = None,
        rescale_degree: bool = False,
        nonlinear_message: bool = True,
        use_attn_head: bool = False,
        norm_layer: str = "layer",
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        out_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        avg_degree: float = _AVG_DEGREE,
        avg_num_nodes: float = _AVG_NUM_NODES,
        output_mode: Literal["direct", "feature"] = "feature",
        **kwargs,
    ) -> None:
        self.output_mode: Literal["direct", "feature"] = output_mode
        output_fields = {"Ea"} if output_mode == "direct" else {"atom_feature"}
        super().__init__(
            input_fields={
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "rbf",
                "atom_embedding",
                "batch_seg",
            },
            output_fields=output_fields,
        )
        if fc_neurons is None:
            fc_neurons = [64, 64]

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.irreps_head = o3.Irreps(irreps_head)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_rbf = num_rbf
        self.fc_neurons = [num_rbf] + list(fc_neurons)
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.use_attn_head = use_attn_head
        self.irreps_pre_attn = irreps_pre_attn
        # Scalar readout contract: dim_feature_out is 0e channel count.
        self.feature_irreps = str(self.irreps_feature)
        self.dim_feature_out = scalar_0e_dim(self.irreps_feature)

        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding,
            self.irreps_edge_attr,
            self.fc_neurons,
            avg_degree,
        )
        self.blocks = ModuleList()
        for i in range(num_layers):
            irreps_block_output = (
                self.irreps_node_embedding
                if i != (num_layers - 1)
                else self.irreps_feature
            )
            self.blocks.append(
                TransBlock(
                    irreps_node_input=self.irreps_node_embedding,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=self.irreps_edge_attr,
                    irreps_node_output=irreps_block_output,
                    fc_neurons=self.fc_neurons,
                    irreps_head=self.irreps_head,
                    num_heads=self.num_heads,
                    irreps_pre_attn=self.irreps_pre_attn,
                    rescale_degree=self.rescale_degree,
                    nonlinear_message=self.nonlinear_message,
                    alpha_drop=alpha_drop,
                    proj_drop=proj_drop,
                    drop_path_rate=drop_path_rate,
                    irreps_mlp_mid=self.irreps_mlp_mid,
                    norm_layer=norm_layer,
                )
            )
        self.norm = get_norm_layer(norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, out_drop)

        if self.output_mode == "direct":
            if self.use_attn_head:
                self.head = GraphAttention(
                    irreps_node_input=self.irreps_feature,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=self.irreps_edge_attr,
                    irreps_node_output=o3.Irreps("1x0e"),
                    fc_neurons=self.fc_neurons,
                    irreps_head=self.irreps_head,
                    num_heads=self.num_heads,
                    irreps_pre_attn=self.irreps_pre_attn,
                    rescale_degree=self.rescale_degree,
                    nonlinear_message=self.nonlinear_message,
                    alpha_drop=alpha_drop,
                    proj_drop=proj_drop,
                )
            else:
                self.head = torch.nn.Sequential(
                    LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
                    Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
                    LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
                )
            self.scale_scatter = ScaledScatter(avg_num_nodes)

    def __str__(self) -> str:
        return """
#########################################################################
# Equiformer (ICLR 2023 Spotlight, arXiv:2206.11990)                    #
# Equivariant Graph Attention Transformer for 3D Atomistic Graphs       #
#########################################################################
"""

    def build(self, built_layers: List[Module]) -> None:
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
                if isinstance(layer, RangeSeparationLayer):
                    self.range_separation = layer
                    self.range_separation.reset_field_name(
                        idx_i_lr="idx_i", idx_j_lr="idx_j"
                    )
                elif isinstance(layer, (EquiformerNodeEmbedding, BaseAtomEmbedding)):
                    self.atom_embedding = layer
                elif isinstance(layer, BaseRBF) or layer.__class__.__name__ in {
                    "GaussianSmearing",
                    "ExpNormalSmearing",
                }:
                    self.radial_basis_function = layer
                self.pre_sequence.append(layer)
            else:
                if isinstance(layer, ChargeConservationLayer):
                    self.charge_conservation = layer
                self.post_sequence.append(layer)

    def encode_irreps(
        self,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        rbf: Tensor,
        atom_embedding: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Tensor:
        """Return node irreps features after TransBlocks + final LayerNorm (pre-head).

        Used by numerical parity tests against upstream Equiformer; production
        :meth:`get_output` still emits only ``atom_feature`` / ``Ea``.
        """
        n_atoms = atom_embedding.shape[0]
        if batch_seg is None:
            batch_seg = torch.zeros(
                n_atoms, dtype=torch.long, device=atom_embedding.device
            )

        # Enerzyme: vij = Rj - Ri; Equiformer: edge_vec = pos[src] - pos[dst]
        # with src=neighbor (j), dst=center (i).
        edge_src = idx_j_sr
        edge_dst = idx_i_sr
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=vij_sr,
            normalize=True,
            normalization="component",
        )
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding,
            edge_sh,
            rbf,
            edge_src,
            edge_dst,
            batch_seg,
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=rbf,
                batch=batch_seg,
            )

        node_features = self.norm(node_features, batch=batch_seg)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        return node_features

    def get_output(
        self,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        rbf: Tensor,
        atom_embedding: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        edge_src = idx_j_sr
        edge_dst = idx_i_sr
        node_features = self.encode_irreps(
            vij_sr=vij_sr,
            idx_i_sr=idx_i_sr,
            idx_j_sr=idx_j_sr,
            rbf=rbf,
            atom_embedding=atom_embedding,
            batch_seg=batch_seg,
        )
        n_atoms = atom_embedding.shape[0]
        if batch_seg is None:
            batch_seg = torch.zeros(
                n_atoms, dtype=torch.long, device=atom_embedding.device
            )
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=vij_sr,
            normalize=True,
            normalization="component",
        )

        if self.output_mode == "feature":
            # Full irreps tensor; SimpleReadout extracts 0e via feature_irreps.
            return {"atom_feature": node_features}

        if self.use_attn_head:
            outputs = self.head(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=rbf,
                batch=batch_seg,
            )
        else:
            outputs = self.head(node_features)
        return {"Ea": outputs.view(-1)}
