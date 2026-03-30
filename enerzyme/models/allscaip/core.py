"""AllScAIP core force field: InputBlock + stacked GraphAttentionBlock (fairchem layout)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import (
    BaseAtomEmbedding,
    BaseElectronEmbedding,
    BaseFFCore,
    BaseRBF,
    ChargeConservationLayer,
    DistanceLayer,
    RangeSeparationLayer,
)
from .equivariant_input import EquivariantInputBlock
from .graph_blocks import GraphAttentionBlock, InputBlock
from .graph_preprocess import (
    AllScAIPGraphHParams,
    _default_freequency_list,
    build_graph_attention_data,
)

# Defaults align with fairchem AllScAIP-style configs (Global / MolecularGraph / GNN / Regularization).
DEFAULT_BUILD_PARAMS = {
    # Global
    "dim_embedding": 128,
    "num_layers": 4,
    "activation": "silu",
    "use_node_path": True,
    "use_residual_scaling": True,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    # Shared preprocessing / embeddings (Enerzyme pipeline)
    "num_rbf": 8,
    "max_Za": 86,
    "cutoff_sr": 5.0,
    "cutoff_fn": "polynomial",
    # Molecular graph
    "knn_k": 32,
    "knn_pad_size": None,
    "knn_soft": False,
    "knn_sigmoid_scale": 0.2,
    "knn_lse_scale": 0.1,
    "knn_use_low_mem": False,
    "use_envelope": True,
    "distance_function": "gaussian",
    "edge_distance_expansion_size": 8,
    "edge_direction_expansion_size": 3,
    "node_direction_expansion_size": 3,
    "preprocess_on_cpu": False,
    "pbc": False,
    "use_padding": False,
    "max_atoms": None,
    "max_batch_size": None,
    "single_system_no_padding": False,
    # GNN / attention
    "atten_num_heads": 4,
    "ffn_hidden_layer_multiplier": 2,
    "attn_num_freq": 32,
    "freequency_list": None,
    "use_freq_mask": True,
    "use_sincx_mask": True,
    # Regularization
    "normalization": "layer_norm",
    "mlp_dropout": 0.0,
    "atten_dropout": 0.0,
    # Equivariant LAE (TP + invariant scalars; disables SH frequency mask on Q/K)
    "equivariant_mode": False,
    "equivariant_tp_hidden": 32,
    "equivariant_tp_mul": 8,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "trainable": False,
        },
    },
    {
        "name": "NuclearEmbedding",
        "params": {
            "zero_init": False,
            "use_electron_config": True,
        },
    },
    {
        "name": "ElectronicEmbedding",
        "params": {
            "num_residual": 1,
            "attribute": "charge",
        },
    },
    {
        "name": "ElectronicEmbedding",
        "params": {
            "num_residual": 1,
            "attribute": "spin",
        },
    },
    {
        "name": "Core",
        "params": {},
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
    {
        "name": "ElectrostaticEnergy",
        "params": {"flavor": "PhysNet", "cutoff_lr": None},
    },
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class AllScAIPCore(BaseFFCore):
    """AllScAIP backbone: kNN graph → InputBlock → GraphAttentionBlocks → atomic readout."""

    def __str__(self) -> str:
        return """
###################################################################
# AllScAIP-style attention MLIP (All-to-all Scaled Attention IP)   #
###################################################################
"""

    def __init__(
        self,
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
    ) -> None:
        super().__init__(
            input_fields={
                "Ra",
                "Za",
                "batch_seg",
                "atom_embedding",
                "charge_embedding",
                "spin_embedding",
            },
            output_fields={"Ea", "Qa"},
        )
        self.dim_embedding = dim_embedding
        self.num_layers = num_layers
        self.atten_num_heads = atten_num_heads
        self.ffn_hidden_layer_multiplier = ffn_hidden_layer_multiplier
        self.mlp_dropout = mlp_dropout
        self.atten_dropout = atten_dropout
        self.use_node_path = use_node_path
        self.use_residual_scaling = use_residual_scaling
        self.activation = activation
        self.normalization = normalization
        self.cutoff_sr = cutoff_sr
        self.knn_k = knn_k
        self.knn_pad_size = knn_pad_size
        self.use_envelope = use_envelope
        self.distance_function = distance_function
        self.edge_distance_expansion_size = edge_distance_expansion_size
        self.edge_direction_expansion_size = edge_direction_expansion_size
        self.node_direction_expansion_size = node_direction_expansion_size
        self.use_freq_mask = use_freq_mask
        self.freequency_list = freequency_list
        self.use_sincx_mask = use_sincx_mask
        self.attn_num_freq = attn_num_freq
        self.preprocess_on_cpu = preprocess_on_cpu
        self.pbc = pbc
        self.use_padding = use_padding
        self.max_atoms = max_atoms
        self.max_batch_size = max_batch_size
        self.single_system_no_padding = single_system_no_padding
        self.knn_soft = knn_soft
        self.knn_sigmoid_scale = knn_sigmoid_scale
        self.knn_lse_scale = knn_lse_scale
        self.knn_use_low_mem = knn_use_low_mem
        self.equivariant_mode = equivariant_mode
        self.equivariant_tp_hidden = equivariant_tp_hidden
        self.equivariant_tp_mul = equivariant_tp_mul

        lmax_edge = edge_direction_expansion_size - 1
        if equivariant_mode:
            self.input_block = EquivariantInputBlock(
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
            self.input_block = InputBlock(
                hidden_size=dim_embedding,
                node_direction_expansion_size=node_direction_expansion_size,
                edge_distance_expansion_size=edge_distance_expansion_size,
                edge_direction_sh_dim=edge_dir_sh_dim,
                activation=activation,
                normalization=normalization,
            )
        head_dim = dim_embedding // atten_num_heads
        self._freequency_list_resolved: List[int] = (
            list(freequency_list)
            if freequency_list is not None
            else _default_freequency_list(head_dim, lmax_edge)
        )
        self._use_freq_mask_resolved = bool(use_freq_mask) and not equivariant_mode

        self.graph_blocks = ModuleList(
            [
                GraphAttentionBlock(
                    hidden_size=dim_embedding,
                    num_layers=num_layers,
                    use_node_path=use_node_path,
                    attn_num_heads=atten_num_heads,
                    attn_num_freq=attn_num_freq,
                    atten_dropout=atten_dropout,
                    mlp_dropout=mlp_dropout,
                    activation=activation,
                    ffn_hidden_layer_multiplier=ffn_hidden_layer_multiplier,
                    use_freq_mask=self._use_freq_mask_resolved,
                    freequency_list=list(self._freequency_list_resolved),
                    use_sincx_mask=use_sincx_mask,
                    use_residual_scaling=use_residual_scaling,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = torch.nn.Linear(dim_embedding, 2)

    def _graph_hparams(self) -> AllScAIPGraphHParams:
        return AllScAIPGraphHParams(
            hidden_size=self.dim_embedding,
            max_radius=self.cutoff_sr,
            knn_k=self.knn_k,
            knn_soft=self.knn_soft,
            knn_sigmoid_scale=self.knn_sigmoid_scale,
            knn_lse_scale=self.knn_lse_scale,
            knn_use_low_mem=self.knn_use_low_mem,
            knn_pad_size=self.knn_pad_size,
            use_envelope=self.use_envelope,
            distance_function=self.distance_function,
            edge_distance_expansion_size=self.edge_distance_expansion_size,
            edge_direction_expansion_size=self.edge_direction_expansion_size,
            node_direction_expansion_size=self.node_direction_expansion_size,
            atten_num_heads=self.atten_num_heads,
            freequency_list=self._freequency_list_resolved,
            use_freq_mask=self._use_freq_mask_resolved,
            use_node_path=self.use_node_path,
            use_sincx_mask=self.use_sincx_mask,
            attn_num_freq=self.attn_num_freq,
            single_system_no_padding=self.single_system_no_padding,
            use_padding=self.use_padding,
            max_atoms=self.max_atoms,
            max_batch_size=self.max_batch_size,
            preprocess_on_cpu=self.preprocess_on_cpu,
            pbc=self.pbc,
        )

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
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, BaseAtomEmbedding):
                    self.atom_embedding = layer
                elif isinstance(layer, BaseElectronEmbedding):
                    if layer.attribute == "charge":
                        self.charge_embedding = layer
                    elif layer.attribute == "spin":
                        self.spin_embedding = layer
                elif isinstance(layer, BaseRBF):
                    self.radial_basis_function = layer
                self.pre_sequence.append(layer)
            else:
                if isinstance(layer, ChargeConservationLayer):
                    self.charge_conservation = layer
                self.post_sequence.append(layer)

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        batch_seg: Tensor,
        atom_embedding: Tensor,
        charge_embedding: Optional[Tensor] = None,
        spin_embedding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if charge_embedding is None:
            charge_embedding = torch.zeros_like(atom_embedding)
        if spin_embedding is None:
            spin_embedding = torch.zeros_like(atom_embedding)
        node_init = atom_embedding + charge_embedding + spin_embedding

        graph_data = build_graph_attention_data(
            Ra,
            Za,
            batch_seg,
            hparams=self._graph_hparams(),
            charge=None,
            spin=None,
            cell=None,
        )

        neighbor_reps = self.input_block(graph_data, node_init)
        for layer_idx, block in enumerate(self.graph_blocks):
            neighbor_reps = block(graph_data, neighbor_reps, layer_idx=layer_idx)

        node_h = neighbor_reps[:, 0]
        out = self.readout(node_h)
        mask = graph_data.node_padding_mask.to(out.dtype).unsqueeze(-1)
        out = out * mask
        return {"Ea": out[:, 0], "Qa": out[:, 1]}
