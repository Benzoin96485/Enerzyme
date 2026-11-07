"""NequIP Core (Batzner et al., Nat. Commun. 2022).

The Core is the equivariant convolution stack only. It reads a scalar species
embedding and a radial basis on the short-range graph, and returns the latent
``atom_feature``. Chemical embedding, the Bessel × polynomial edge embedding,
the scalar energy head, per-element shifts, and ``Fa = -∇E`` stay in the
layer stack.

The convolution matches current ``mir-group/nequip`` ``InteractionBlock`` and
``ConvNetLayer`` (and the 2022 paper): spherical harmonics of
``R_sender - R_receiver``, a radial MLP on the edge embedding, a Clebsch–Gordan
tensor product, a self-connection against the fixed species embedding, and a
gate. The first layer omits the self-connection unless ``sc_first`` is set, so
an isolated atom stays at zero and its reference energy belongs in
``AtomicAffine``. Checkpoints from the ``nequip`` package are not loaded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..e3nn_nn import scalar_0e_dim
from ..layers import BaseAtomEmbedding, BaseFFCore, BaseRBF, DistanceLayer, RangeSeparationLayer
from .interaction import NequIPConv, hidden_irreps


# Foundation-model sizes from mir-group/nequip ``NequIPGNNModel`` presets.
# ``type_embed_num_features`` stays ``dim_embedding`` on the layer stack (32
# in those presets). ``paper`` is the 2022 molecular setting: both parities,
# two hidden radial layers of width 64.
_PRESETS = {
    "S": {
        "num_layers": 2,
        "l_max": 1,
        "num_features": [128, 64],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "M": {
        "num_layers": 4,
        "l_max": 2,
        "num_features": [128, 64, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "L": {
        "num_layers": 6,
        "l_max": 3,
        "num_features": [128, 64, 32, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "XL": {
        "num_layers": 6,
        "l_max": 4,
        "num_features": [320, 96, 64, 32, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "paper": {
        "num_layers": 4,
        "l_max": 1,
        "num_features": 32,
        "parity": True,
        "radial_mlp_depth": 2,
        "radial_mlp_width": 64,
    },
}

_DEFAULTS = {
    "num_layers": 4,
    "l_max": 1,
    "num_features": 32,
    "parity": True,
    "radial_mlp_depth": 1,
    "radial_mlp_width": 128,
    "use_sc": True,
    "sc_first": False,
    "resnet": False,
    "resnet_first": False,
    "nonlinearity_type": "gate",
    "avg_num_neighbors": 1.0,
}


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 4,
            "l_max": 1,
            "parity": True,
            "num_features": 32,
            "radial_mlp_depth": 1,
            "radial_mlp_width": 128,
            "use_sc": True,
            "sc_first": False,
            "avg_num_neighbors": 1.0,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0, "learnable": True}},
            "scales": {"Ea": {"values": 1, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _resolve(preset: Optional[str], overrides: Dict[str, object]) -> Dict[str, object]:
    resolved = dict(_DEFAULTS)
    if preset is not None:
        key = str(preset).upper() if str(preset).upper() in _PRESETS else str(preset).lower()
        if key == "PAPER":
            key = "paper"
        if key not in _PRESETS:
            known = ", ".join([*list(_PRESETS)[:-1], "paper"])
            raise ValueError(f"unknown NequIP preset {preset!r}; expected {known}")
        resolved.update(_PRESETS[key])
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value
    return resolved


class NequIPCore(BaseFFCore):
    """E(3)-equivariant convolution stack. Property heads stay outside."""

    def __str__(self) -> str:
        return f"""
################################################################################
# NequIP (Batzner et al., Nat. Commun. 2022; mir-group/nequip convolutions)    #
# {self.num_layers} layers, l_max={self.l_max}, parity={self.parity}, latent {self.feature_irreps:<24} #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        preset: Optional[str] = None,
        num_layers: Optional[int] = None,
        l_max: Optional[int] = None,
        parity: Optional[bool] = None,
        num_features: Optional[Union[int, Sequence[int]]] = None,
        radial_mlp_depth: Optional[int] = None,
        radial_mlp_width: Optional[int] = None,
        use_sc: Optional[bool] = None,
        sc_first: Optional[bool] = None,
        resnet: Optional[bool] = None,
        resnet_first: Optional[bool] = None,
        nonlinearity_type: Optional[str] = None,
        nonlinearity_scalars: Optional[Dict[str, str]] = None,
        nonlinearity_gates: Optional[Dict[str, str]] = None,
        avg_num_neighbors: Optional[float] = None,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
            },
            output_fields={"atom_feature"},
        )
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        cfg = _resolve(
            preset,
            {
                "num_layers": num_layers,
                "l_max": l_max,
                "parity": parity,
                "num_features": num_features,
                "radial_mlp_depth": radial_mlp_depth,
                "radial_mlp_width": radial_mlp_width,
                "use_sc": use_sc,
                "sc_first": sc_first,
                "resnet": resnet,
                "resnet_first": resnet_first,
                "nonlinearity_type": nonlinearity_type,
                "avg_num_neighbors": avg_num_neighbors,
            },
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.num_layers = int(cfg["num_layers"])
        self.l_max = int(cfg["l_max"])
        self.parity = bool(cfg["parity"])
        self.num_features = cfg["num_features"]
        self.use_sc = bool(cfg["use_sc"])
        self.sc_first = bool(cfg["sc_first"])
        self.resnet = bool(cfg["resnet"])
        self.resnet_first = bool(cfg["resnet_first"])
        self.nonlinearity_type = str(cfg["nonlinearity_type"])
        self.avg_num_neighbors = float(cfg["avg_num_neighbors"])
        self.radial_mlp_depth = int(cfg["radial_mlp_depth"])
        self.radial_mlp_width = int(cfg["radial_mlp_width"])
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {self.l_max}")
        if self.avg_num_neighbors <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {self.avg_num_neighbors}"
            )

        self.irreps_edge_sh = o3.Irreps.spherical_harmonics(self.l_max)
        self.sph = o3.SphericalHarmonics(
            self.irreps_edge_sh, normalize=True, normalization="component"
        )
        self.irreps_node_attr = o3.Irreps(f"{self.dim_embedding}x0e")
        hidden = hidden_irreps(self.num_features, self.l_max, self.parity)
        if isinstance(self.num_features, int):
            scalar_mul = int(self.num_features)
        else:
            scalar_mul = int(self.num_features[0])
        if scalar_mul < 1:
            raise ValueError(
                "the l=0 feature count must be >= 1 so the latent has scalar channels"
            )
        scalar_out = o3.Irreps([(scalar_mul, (0, 1))])
        targets = [hidden] * (self.num_layers - 1) + [scalar_out]

        irreps = self.irreps_node_attr
        self.convs = ModuleList()
        for layer_i, target in enumerate(targets):
            conv = NequIPConv(
                irreps_in=irreps,
                feature_irreps_hidden=target,
                irreps_edge_attr=self.irreps_edge_sh,
                irreps_node_attr=self.irreps_node_attr,
                num_rbf=self.num_rbf,
                radial_mlp_depth=self.radial_mlp_depth,
                radial_mlp_width=self.radial_mlp_width,
                avg_num_neighbors=self.avg_num_neighbors,
                use_sc=self.use_sc and (layer_i > 0 or self.sc_first),
                resnet=self.resnet and (layer_i > 0 or self.resnet_first),
                nonlinearity_type=self.nonlinearity_type,
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
            )
            self.convs.append(conv)
            irreps = conv.irreps_out
        self.feature_irreps = str(irreps)
        self.dim_feature_out = scalar_0e_dim(irreps)

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
                if isinstance(layer, DistanceLayer):
                    continue
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, (BaseAtomEmbedding, BaseRBF)):
                    pass
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        """Run the convolution stack.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``, the same displacement NequIP stores as
        ``pos[src] - pos[dst]``. ``atom_embedding`` is both the initial
        scalar state and the fixed node attribute for self-connections.
        """
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}"
            )
        if idx_i_sr.shape != idx_j_sr.shape:
            raise ValueError("idx_i_sr and idx_j_sr must have the same shape")
        if rbf.shape[0] != idx_i_sr.shape[0] or vij_sr.shape[0] != idx_i_sr.shape[0]:
            raise ValueError(
                "rbf, vij_sr, and the neighbor index must describe the same edges"
            )
        features = atom_embedding
        node_attrs = atom_embedding
        edge_src = idx_j_sr.long()
        edge_dst = idx_i_sr.long()
        edge_embed = rbf.to(dtype=features.dtype)
        if edge_src.numel() == 0:
            edge_attr = features.new_zeros((0, self.irreps_edge_sh.dim))
        else:
            edge_attr = self.sph(vij_sr.to(dtype=features.dtype))
        for conv in self.convs:
            features = conv(
                features, node_attrs, edge_src, edge_dst, edge_attr, edge_embed
            )
        return {"atom_feature": features}
