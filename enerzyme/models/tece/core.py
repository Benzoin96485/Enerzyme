"""TECE Core — Edge Cluster Expansion with Radial Rotary Attention.

Paper: Xu et al., arXiv:2607.10664
Upstream: https://github.com/xvzemin/tace (MIT)

Feature-mode Core only: emits ``atom_feature`` for external ``SimpleReadout``.
Embeddings / energy-force heads / physics priors stay outside the Core.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps, SphericalHarmonics
from torch import Tensor
from torch.nn import ModuleList

from ..blocks.radial_mlp import RadialMLP
from ..e3nn_nn import IrrepsLinear, O3ScatterTensorProduct, scalar_0e_dim
from ..layers import (
    BaseAtomEmbedding,
    BaseElectronEmbedding,
    BaseFFCore,
    BaseRBF,
    ChargeConservationLayer,
    DistanceLayer,
    RangeSeparationLayer,
)
from ..so3 import WignerD
from ..tace.interaction import EDGE_EMBEDDING, EDGE_UPDATE, CgtpACE
from .interaction import UvSO2Interaction

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 48,
    "num_rbf": 8,
    "max_Za": 86,
    "cutoff_sr": 6.0,
    "cutoff_fn": "polynomial",
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "BesselRBF", "params": {"trainable": False}},
    {
        "name": "NuclearEmbedding",
        "params": {
            "element_embedding_cueq_config": None,
            "compute_cueq_config": None,
        },
    },
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "num_channel": 48,
            "Lmax": 2,
            "lmax": 2,
            "mmax": 2,
            "correlation": 2,
            "avg_num_neighbors": 8.0,
            "edge_embedding": "nonlinear",
            "edge_update": "element2",
            "scatter_norm": "avg_num_neighbors",
            "resnet_type": "BB",
            "resnet_linear_type": "aware",
            "use_first_resnet": False,
            "nonlinear": "sigmoid_gate",
            "edge_nonlinear": "so2_sigmoid_gate",
            "radial_mlp": [64, 64],
            "use_so2_edge_ace": True,
            "use_graph_softmax": True,
            "use_radial_phase": True,
            "use_temperature": True,
            "num_head": 1,
            "so2_linear_type": "w1",
            "gate_m0": False,
            "wigner_type": "recursive",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense", "keep_feature": False},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class _TensorSeed(torch.nn.Module):
    """Lift scalar node embedding to equivariant features via O3 edge TP (TACE tensor seed)."""

    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        num_radial_basis: int,
        avg_num_neighbors: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_channel = num_channel
        self.Lmax = Lmax
        self.lmax = lmax
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.source_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        self.target_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        with torch.no_grad():
            self.source_embedding.weight.uniform_(-0.001, 0.001)
            self.target_embedding.weight.uniform_(-0.001, 0.001)

        self.rejector = O3ScatterTensorProduct(
            [(num_channel, (0, 1))],
            [(1, (l, (-1) ** l)) for l in range(lmax + 1)],
            [(1, (l, (-1) ** l)) for l in range(Lmax + 1)],
        )
        self.irreps_out = self.rejector.irreps_out
        self.edge_info = RadialMLP(
            [
                num_radial_basis + num_channel * 2,
                num_channel,
                num_channel,
                self.rejector.weight_numel,
            ],
            use_layer_norm=True,
            use_offset=False,
            bias=True,
        )

    def forward(
        self,
        base_node_feats: Tensor,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_attrs: Tensor,
        edge_index: Tensor,
        cutoff: Optional[Tensor],
    ) -> Tensor:
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        if cutoff is not None:
            conv_weights = conv_weights * cutoff
        node_feats = (
            self.rejector(
                torch.ones_like(base_node_feats),
                edge_attrs,
                conv_weights,
                edge_index,
            )
            / self.avg_num_neighbors
        )
        node_feats = node_feats.clone()
        node_feats[:, : self.num_channel] = (
            node_feats.narrow(1, 0, self.num_channel) + base_node_feats
        )
        return node_feats


class TECECore(BaseFFCore):
    """Enerzyme Core for TECE (feature mode only).

    Emits ``atom_feature`` (flat e3nn irreps) for external ``SimpleReadout``.
    Requires ``Lmax == lmax`` (upstream SO2 constraint). A tensor seed expands
    scalar NuclearEmbedding features before the first uvSO2 layer.
    """

    def __init__(
        self,
        max_Za: int,
        dim_embedding: int,
        num_rbf: int,
        num_channel: int = 48,
        num_layers: int = 2,
        Lmax: int = 2,
        lmax: int = 2,
        mmax: Optional[int] = None,
        correlation: Union[int, List[int]] = 2,
        avg_num_neighbors: float = 8.0,
        edge_embedding: str = "nonlinear",
        edge_update: str = "element2",
        scatter_norm: str = "avg_num_neighbors",
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        nonlinear: str = "sigmoid_gate",
        edge_nonlinear: str = "so2_sigmoid_gate",
        radial_mlp: Optional[List[int]] = None,
        radial_bias: bool = False,
        use_so2_edge_ace: bool = True,
        use_graph_softmax: bool = True,
        use_radial_phase: bool = True,
        use_temperature: bool = True,
        num_head: int = 1,
        edge_ace_hidden: Optional[int] = None,
        edge_wise_hidden: Optional[int] = None,
        so2_linear_type: str = "w1",
        gate_m0: bool = False,
        scalar_act: Optional[str] = None,
        tensor_act: Optional[str] = None,
        wigner_type: Literal["recursive", "direct"] = "recursive",
        bias: bool = True,
        agnostic_product: bool = False,
    ):
        super().__init__(
            input_fields={
                "Za",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "rbf",
                "cutoff_values_sr",
                "atom_embedding",
                "charge_embedding",
                "spin_embedding",
            },
            output_fields={"atom_feature"},
        )
        if Lmax != lmax:
            raise ValueError(
                f"TECE requires Lmax == lmax (SO2); got Lmax={Lmax}, lmax={lmax}"
            )
        if mmax is None:
            mmax = Lmax
        if mmax > Lmax:
            raise ValueError(f"mmax ({mmax}) cannot exceed Lmax ({Lmax})")
        if radial_mlp is None:
            radial_mlp = [64, 64]
        if isinstance(correlation, int):
            correlation = [correlation] * num_layers
        else:
            correlation = list(correlation)
            if len(correlation) != num_layers:
                raise ValueError(
                    f"correlation list length ({len(correlation)}) must equal "
                    f"num_layers ({num_layers}); got {correlation!r}"
                )

        self.max_Za = max_Za
        self.num_elements = max_Za + 1
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.Lmax = Lmax
        self.lmax = lmax
        self.mmax = mmax
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.avg_num_neighbors = avg_num_neighbors

        target = Irreps([(1, (l, (-1) ** l)) for l in range(Lmax + 1)])
        self.target_irreps = target

        if self.dim_embedding != self.num_channel:
            self.node_proj = IrrepsLinear(
                f"{self.dim_embedding}x0e", f"{self.num_channel}x0e", bias=False
            )
        else:
            self.node_proj = torch.nn.Identity()

        self.spherical_harmonics = SphericalHarmonics(
            Irreps.spherical_harmonics(self.lmax, p=-1),
            normalize=True,
            normalization="component",
        )
        self.wigner = WignerD(
            mmax=self.mmax, lmax=self.Lmax, wigner_type=wigner_type
        )

        self.edge_embedding_mod = EDGE_EMBEDDING[edge_embedding](
            num_radial_basis=self.num_rbf,
            num_channel=self.num_channel,
            num_elements=self.num_elements,
            bias=False,
        )
        edge_emb_dim = self.edge_embedding_mod.out_dim

        self.tensor_seed = _TensorSeed(
            num_elements=self.num_elements,
            num_channel=self.num_channel,
            Lmax=self.Lmax,
            lmax=self.lmax,
            num_radial_basis=self.num_rbf,
            avg_num_neighbors=avg_num_neighbors,
            bias=bias,
        )

        self.edge_updates = ModuleList()
        self.interactions = ModuleList()
        self.products = ModuleList()

        irreps_in = self.tensor_seed.irreps_out
        for layer in range(self.num_layers):
            eu = EDGE_UPDATE[edge_update](
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                edge_embedding_channel=edge_emb_dim,
                bias=False,
            )
            self.edge_updates.append(eu)
            inter = UvSO2Interaction(
                layer=layer,
                num_layers=self.num_layers,
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                Lmax=self.Lmax,
                lmax=self.lmax,
                mmax=self.mmax,
                irreps_in=irreps_in,
                edge_feats_channel=eu.out_dim,
                num_radial_basis=self.num_rbf,
                avg_num_neighbors=avg_num_neighbors,
                radial_mlp=radial_mlp,
                radial_bias=radial_bias,
                scatter_norm=scatter_norm,
                nonlinear=nonlinear,
                edge_nonlinear=edge_nonlinear,
                resnet_type=resnet_type,
                resnet_linear_type=resnet_linear_type,
                use_first_resnet=use_first_resnet,
                parity=False,
                bias=bias,
                target_irreps=target,
                correlation=correlation[layer],
                use_so2_edge_ace=use_so2_edge_ace,
                use_graph_softmax=use_graph_softmax,
                use_radial_phase=use_radial_phase,
                use_temperature=use_temperature,
                num_head=num_head,
                edge_ace_hidden=edge_ace_hidden,
                edge_wise_hidden=edge_wise_hidden,
                so2_linear_type=so2_linear_type,
                gate_m0=gate_m0,
                scalar_act=scalar_act,
                tensor_act=tensor_act,
            )
            self.interactions.append(inter)
            prod = CgtpACE(
                layer=layer,
                num_layers=self.num_layers,
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                Lmax=self.Lmax,
                lmax=self.lmax,
                irreps_in=inter.irreps_out,
                correlation=correlation[layer],
                target_irreps=target,
                l1l2=None,
                parity=False,
                bias=bias,
                agnostic=agnostic_product,
            )
            if hasattr(inter, "resnetBB") and inter.irreps_sc != prod.irreps_out:
                raise ValueError(
                    f"TECE BB residual irreps mismatch at layer {layer}: "
                    f"skip={inter.irreps_sc} product={prod.irreps_out}"
                )
            self.products.append(prod)
            irreps_in = prod.irreps_out

        self.feature_irreps = str(irreps_in)
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)

    def __str__(self) -> str:
        return f"""
###############################################################
# TECE Core — arXiv:2607.10664 / xvzemin/tace (ECE + RRA)     #
###############################################################
"""

    def build(self, built_layers) -> None:
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
        Za: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        rbf: Tensor,
        atom_embedding: Tensor,
        charge_embedding: Optional[Tensor] = None,
        spin_embedding: Optional[Tensor] = None,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if charge_embedding is None:
            charge_embedding = torch.zeros_like(atom_embedding)
        if spin_embedding is None:
            spin_embedding = torch.zeros_like(atom_embedding)

        node_feats = self.node_proj(atom_embedding + charge_embedding + spin_embedding)
        node_attrs = F.one_hot(Za, num_classes=self.num_elements).to(
            dtype=node_feats.dtype
        )
        edge_index = torch.stack([idx_i_sr, idx_j_sr], dim=0)
        cutoff = None
        if cutoff_values_sr is not None:
            cutoff = cutoff_values_sr.reshape(-1, 1).to(dtype=node_feats.dtype)

        edge_emb = self.edge_embedding_mod(node_attrs, rbf, edge_index)
        edge_attrs = self.spherical_harmonics(vij_sr)
        wigner, wigner_inv = self.wigner.get_wigner(vij_sr)

        node_feats = self.tensor_seed(
            base_node_feats=node_feats,
            node_attrs=node_attrs,
            edge_feats=rbf,
            edge_attrs=edge_attrs,
            edge_index=edge_index,
            cutoff=cutoff,
        )

        for eu, inter, prod in zip(
            self.edge_updates, self.interactions, self.products
        ):
            edge_feats = eu(node_attrs, edge_emb, edge_index)
            node_feats, sc = inter(
                node_feats=node_feats,
                node_attrs=node_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                wigner=wigner,
                wigner_inv=wigner_inv,
                radial_basis=rbf,
                cutoff=cutoff,
            )
            node_feats = prod(node_feats=node_feats, node_attrs=node_attrs, sc=sc)
        return {"atom_feature": node_feats}
