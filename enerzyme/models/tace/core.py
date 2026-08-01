"""TACE Core — Tensor Atomic Cluster Expansion (spherical + Cartesian).

Papers:
- Xu et al., arXiv:2509.14961 (Cartesian TACE)
- Xu et al., Cartesian-3j, arXiv:2512.16882

Upstream: https://github.com/xvzemin/tace (MIT). TECE / SO2 / RRA are out of scope.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps, SphericalHarmonics
from torch import Tensor
from torch.nn import ModuleList

from ..e3nn_nn import IrrepsLinear, scalar_0e_dim
from ..layers import (
    BaseAtomEmbedding,
    BaseElectronEmbedding,
    BaseFFCore,
    BaseRBF,
    ChargeConservationLayer,
    DistanceLayer,
    RangeSeparationLayer,
)
from .interaction import EDGE_EMBEDDING, EDGE_UPDATE, CgtpACE, CgtpInteraction

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
            "tensor_basis": "spherical",
            "num_layers": 2,
            "num_channel": 48,
            "Lmax": 2,
            "lmax": 3,
            "correlation": 2,
            "avg_num_neighbors": 8.0,
            "parity": False,
            "edge_embedding": "nonlinear",
            "edge_update": "element2",
            "scatter_norm": "avg_num_neighbors",
            "resnet_type": "BB",
            "resnet_linear_type": "aware",
            "use_first_resnet": False,
            "nonlinear": "sigmoid_gate",
            "radial_mlp": [64, 64, 64],
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense", "keep_feature": False},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _natural_parity_irreps(Lmax: int) -> Irreps:
    return Irreps([(1, (l, (-1) ** l)) for l in range(Lmax + 1)])


class TACECore(BaseFFCore):
    """Enerzyme Core for TACE (feature mode only).

    Emits ``atom_feature`` (equivariant flat irreps for spherical; scalar
    channels for Cartesian) for external ``SimpleReadout``.
    """

    def __init__(
        self,
        max_Za: int,
        dim_embedding: int,
        num_rbf: int,
        num_channel: int = 48,
        num_layers: int = 2,
        Lmax: int = 2,
        lmax: int = 3,
        correlation: Union[int, List[int]] = 2,
        avg_num_neighbors: float = 8.0,
        tensor_basis: Literal["spherical", "cartesian"] = "spherical",
        parity: bool = False,
        edge_embedding: str = "nonlinear",
        edge_update: str = "element2",
        scatter_norm: str = "avg_num_neighbors",
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        nonlinear: str = "sigmoid_gate",
        radial_mlp: Optional[List[int]] = None,
        radial_bias: bool = False,
        l1l2: Optional[str] = None,
        agnostic_product: bool = False,
        bias: bool = True,
    ):
        super().__init__(
            input_fields={
                "Za",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "rbf",
                "atom_embedding",
                "charge_embedding",
                "spin_embedding",
            },
            output_fields={"atom_feature"},
        )
        if radial_mlp is None:
            radial_mlp = [64, 64, 64]
        if isinstance(correlation, int):
            correlation = [correlation] * num_layers

        self.tensor_basis = tensor_basis
        self.max_Za = max_Za
        self.num_elements = max_Za + 1
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.Lmax = Lmax
        self.lmax = lmax
        self.parity = parity
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf

        if tensor_basis == "spherical":
            self._build_spherical(
                correlation=correlation,
                avg_num_neighbors=avg_num_neighbors,
                edge_embedding=edge_embedding,
                edge_update=edge_update,
                scatter_norm=scatter_norm,
                resnet_type=resnet_type,
                resnet_linear_type=resnet_linear_type,
                use_first_resnet=use_first_resnet,
                nonlinear=nonlinear,
                radial_mlp=radial_mlp,
                radial_bias=radial_bias,
                l1l2=l1l2,
                agnostic_product=agnostic_product,
                bias=bias,
            )
        elif tensor_basis == "cartesian":
            self._build_cartesian(
                correlation=correlation,
                avg_num_neighbors=avg_num_neighbors,
                edge_embedding=edge_embedding,
                edge_update=edge_update,
                scatter_norm=scatter_norm,
                resnet_type=resnet_type,
                resnet_linear_type=resnet_linear_type,
                use_first_resnet=use_first_resnet,
                radial_mlp=radial_mlp,
                radial_bias=radial_bias,
                l1l2=l1l2,
                bias=bias,
            )
        else:
            raise ValueError(f"Unknown tensor_basis={tensor_basis}")

    def _build_spherical(self, **kw):
        target = _natural_parity_irreps(self.Lmax) if not self.parity else Irreps.spherical_harmonics(self.Lmax)
        self.target_irreps = target
        hidden0 = Irreps([(self.num_channel, (0, 1))])

        # Project NuclearEmbedding dim -> num_channel scalars if needed
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

        self.edge_embedding_mod = EDGE_EMBEDDING[kw["edge_embedding"]](
            num_radial_basis=self.num_rbf,
            num_channel=self.num_channel,
            num_elements=self.num_elements,
            bias=False,
        )
        edge_emb_dim = self.edge_embedding_mod.out_dim

        self.edge_updates = ModuleList()
        self.interactions = ModuleList()
        self.products = ModuleList()

        irreps_in = hidden0
        for layer in range(self.num_layers):
            eu = EDGE_UPDATE[kw["edge_update"]](
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                edge_embedding_channel=edge_emb_dim,
                bias=False,
            )
            self.edge_updates.append(eu)
            inter = CgtpInteraction(
                layer=layer,
                num_layers=self.num_layers,
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                Lmax=self.Lmax,
                lmax=self.lmax,
                irreps_in=irreps_in,
                edge_feats_channel=eu.out_dim,
                avg_num_neighbors=kw["avg_num_neighbors"],
                radial_mlp=kw["radial_mlp"],
                radial_bias=kw["radial_bias"],
                scatter_norm=kw["scatter_norm"],
                nonlinear=kw["nonlinear"],
                l1l2=kw["l1l2"],
                resnet_type=kw["resnet_type"],
                resnet_linear_type=kw["resnet_linear_type"],
                use_first_resnet=kw["use_first_resnet"],
                parity=self.parity,
                bias=kw["bias"],
                target_irreps=target,
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
                correlation=kw["correlation"][layer],
                target_irreps=target,
                l1l2=kw["l1l2"],
                parity=self.parity,
                bias=kw["bias"],
                agnostic=kw["agnostic_product"],
            )
            self.products.append(prod)
            irreps_in = prod.irreps_out

        self.feature_irreps = str(irreps_in)
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)

    def _build_cartesian(self, **kw):
        from .cartesian.core_blocks import CartesianLayerStack

        if self.dim_embedding != self.num_channel:
            self.node_proj = IrrepsLinear(
                f"{self.dim_embedding}x0e", f"{self.num_channel}x0e", bias=False
            )
        else:
            self.node_proj = torch.nn.Identity()

        self.edge_embedding_mod = EDGE_EMBEDDING[kw["edge_embedding"]](
            num_radial_basis=self.num_rbf,
            num_channel=self.num_channel,
            num_elements=self.num_elements,
            bias=False,
        )
        self.cartesian_stack = CartesianLayerStack(
            num_layers=self.num_layers,
            num_elements=self.num_elements,
            num_channel=self.num_channel,
            Lmax=self.Lmax,
            lmax=self.lmax,
            correlation=kw["correlation"],
            avg_num_neighbors=kw["avg_num_neighbors"],
            edge_embedding_channel=self.edge_embedding_mod.out_dim,
            edge_update=kw["edge_update"],
            scatter_norm=kw["scatter_norm"],
            radial_mlp=kw["radial_mlp"],
            radial_bias=kw["radial_bias"],
            use_first_resnet=kw["use_first_resnet"],
            resnet_linear_type=kw["resnet_linear_type"],
            l1l2=kw["l1l2"],
            bias=kw["bias"],
        )
        self.feature_irreps = f"{self.num_channel}x0e"
        self.dim_feature_out = self.num_channel

    def __str__(self) -> str:
        return f"""
###############################################################
# TACE Core ({self.tensor_basis}) — arXiv:2509.14961 / xvzemin/tace #
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
                    self.range_separation.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
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
    ) -> Dict[str, Tensor]:
        if charge_embedding is None:
            charge_embedding = torch.zeros_like(atom_embedding)
        if spin_embedding is None:
            spin_embedding = torch.zeros_like(atom_embedding)

        node_feats = self.node_proj(atom_embedding + charge_embedding + spin_embedding)
        node_attrs = F.one_hot(Za, num_classes=self.num_elements).to(dtype=node_feats.dtype)
        edge_index = torch.stack([idx_i_sr, idx_j_sr], dim=0)

        edge_emb = self.edge_embedding_mod(node_attrs, rbf, edge_index)

        if self.tensor_basis == "spherical":
            edge_attrs = self.spherical_harmonics(vij_sr)
            for eu, inter, prod in zip(self.edge_updates, self.interactions, self.products):
                edge_feats = eu(node_attrs, edge_emb, edge_index)
                node_feats, sc = inter(
                    node_feats=node_feats,
                    node_attrs=node_attrs,
                    edge_feats=edge_feats,
                    edge_attrs=edge_attrs,
                    edge_index=edge_index,
                    cutoff=None,
                )
                node_feats = prod(node_feats=node_feats, node_attrs=node_attrs, sc=sc)
            return {"atom_feature": node_feats}

        atom_feature = self.cartesian_stack(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_emb=edge_emb,
            edge_index=edge_index,
            edge_vec=vij_sr,
        )
        return {"atom_feature": atom_feature}
