"""Allegro Core (Musaelian et al., Nat. Commun. 2023).

The Core maps a two-body scalar embedding and the short-range displacement
onto the latent pair features. Species tables, the Bessel × polynomial basis,
the pair-energy MLP, the directed-edge sum, per-element shifts, and
``Fa = -∇E`` stay outside.

Each layer tensor-products the current pair features with a sum of weighted
spherical harmonics over the center atom's neighbors. Scalar channels of
every layer are concatenated (the Allegro dense scalar track) and returned
as ``edge_feature``. One layer builds three-body features; each added layer
raises the body order by one. The receptive field never grows past
``cutoff_sr``.

The tensor product follows ``mir-group/allegro`` ``Allegro_Module`` (MIT).
Strided, Triton, and cuequivariance kernels are not used, and checkpoints
from that package are not loaded.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..blocks.mlp import VariancePreservingMLP
from ..layers import BaseFFCore
from .interaction import ChannelTensorProduct, WeightedSphericalChannels, scatter_sum
from .irreps import allegro_layer_irreps, edge_sh_irreps


# Widths from mir-group/allegro ``configs/tutorial.yaml``.
_PRESETS = {
    "tutorial": {
        "num_layers": 2,
        "l_max": 1,
        "parity": True,
        "num_scalar_features": 64,
        "num_tensor_features": 32,
        "scalar_embed_mlp_depth": 1,
        "scalar_embed_mlp_width": 64,
        "latent_mlp_depth": 1,
        "latent_mlp_width": 64,
        "nonlinearity": "silu",
        "tp_path_channel_coupling": True,
        "weight_individual_irreps": True,
    },
}

_DEFAULTS = dict(_PRESETS["tutorial"])


DEFAULT_BUILD_PARAMS = {
    "dim_edge": 64,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "avg_num_neighbors": 1.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
    },
    {"name": "PairChemicalRadial"},
    {"name": "Core"},
    {
        "name": "EdgeReadout",
        "params": {
            "output_fields": ["Eij"],
            "head_type": "variance_preserving",
            "hidden_layers_depth": 1,
            "nonlinearity": "silu",
            "use_bias": False,
        },
    },
    {"name": "EdgeEnergyReduce"},
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
        key = str(preset).lower()
        if key not in _PRESETS:
            known = ", ".join(_PRESETS)
            raise ValueError(f"unknown Allegro preset {preset!r}; expected {known}")
        resolved.update(_PRESETS[key])
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value
    return resolved


class AllegroCore(BaseFFCore):
    """Strictly local equivariant pair latent. The energy head stays outside."""

    def __str__(self) -> str:
        return f"""
################################################################################
# Allegro (Musaelian et al., Nat. Commun. 2023; mir-group/allegro tensor track)
# {self.num_layers} layers, l_max={self.l_max}, parity={self.parity}, scalar {self.num_scalar_features}, tensor {self.num_tensor_features:<4} #
################################################################################
"""

    def __init__(
        self,
        dim_edge: int,
        avg_num_neighbors: float,
        preset: Optional[str] = None,
        num_layers: Optional[int] = None,
        l_max: Optional[int] = None,
        parity: Optional[bool] = None,
        num_scalar_features: Optional[int] = None,
        num_tensor_features: Optional[int] = None,
        scalar_embed_mlp_depth: Optional[int] = None,
        scalar_embed_mlp_width: Optional[int] = None,
        latent_mlp_depth: Optional[int] = None,
        latent_mlp_width: Optional[int] = None,
        nonlinearity: Optional[str] = None,
        tp_path_channel_coupling: Optional[bool] = None,
        weight_individual_irreps: Optional[bool] = None,
    ) -> None:
        super().__init__(
            input_fields={"edge_embedding", "idx_i_sr", "idx_j_sr", "vij_sr"},
            output_fields={"edge_feature"},
        )
        cfg = _resolve(
            preset,
            {
                "num_layers": num_layers,
                "l_max": l_max,
                "parity": parity,
                "num_scalar_features": num_scalar_features,
                "num_tensor_features": num_tensor_features,
                "scalar_embed_mlp_depth": scalar_embed_mlp_depth,
                "scalar_embed_mlp_width": scalar_embed_mlp_width,
                "latent_mlp_depth": latent_mlp_depth,
                "latent_mlp_width": latent_mlp_width,
                "nonlinearity": nonlinearity,
                "tp_path_channel_coupling": tp_path_channel_coupling,
                "weight_individual_irreps": weight_individual_irreps,
            },
        )
        self.dim_edge = int(dim_edge)
        self.num_layers = int(cfg["num_layers"])
        self.l_max = int(cfg["l_max"])
        self.parity = bool(cfg["parity"])
        self.num_scalar_features = int(cfg["num_scalar_features"])
        self.num_tensor_features = int(cfg["num_tensor_features"])
        self.nonlinearity = str(cfg["nonlinearity"])
        self.tp_path_channel_coupling = bool(cfg["tp_path_channel_coupling"])
        self.weight_individual_irreps = bool(cfg["weight_individual_irreps"])
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.scalar_embed_mlp_depth = int(cfg["scalar_embed_mlp_depth"])
        self.latent_mlp_depth = int(cfg["latent_mlp_depth"])
        scalar_width = cfg["scalar_embed_mlp_width"]
        latent_width = cfg["latent_mlp_width"]
        self.scalar_embed_mlp_width = (
            self.num_scalar_features if scalar_width is None else int(scalar_width)
        )
        self.latent_mlp_width = (
            self.num_scalar_features if latent_width is None else int(latent_width)
        )
        # EdgeReadout uses this when its hidden width is left unset.
        self.readout_hidden_width = self.num_scalar_features
        self._check_config()

        self.irreps_edge_sh = edge_sh_irreps(self.l_max)
        self.sph = o3.SphericalHarmonics(
            self.irreps_edge_sh, normalize=True, normalization="component"
        )
        self.weighter = WeightedSphericalChannels(
            self.irreps_edge_sh,
            self.num_tensor_features,
            weight_individual_irreps=self.weight_individual_irreps,
        )
        self.scalar_mlp = self._mlp(
            self.dim_edge,
            self.num_scalar_features,
            self.scalar_embed_mlp_depth,
            self.scalar_embed_mlp_width,
        )
        self.tensor_weight = self._linear(
            self.num_scalar_features, self.weighter.weight_numel
        )
        self.pair_projection = self._linear(
            self.num_scalar_features,
            self.num_scalar_features + self.weighter.weight_numel,
        )
        pairs = allegro_layer_irreps(self.l_max, self.parity, self.num_layers)
        self.products = ModuleList()
        self.latents = ModuleList()
        for layer, (irreps_in, irreps_out) in enumerate(pairs):
            self.products.append(
                ChannelTensorProduct(
                    irreps_in,
                    self.irreps_edge_sh,
                    irreps_out,
                    mul=self.num_tensor_features,
                    path_channel_coupling=self.tp_path_channel_coupling,
                )
            )
            last = layer == self.num_layers - 1
            self.latents.append(
                self._mlp(
                    self.num_scalar_features * (layer + 1) + self.num_tensor_features,
                    self.num_scalar_features
                    + (0 if last else self.weighter.weight_numel),
                    self.latent_mlp_depth,
                    self.latent_mlp_width,
                )
            )
        self.dim_feature_out = self.num_scalar_features * (self.num_layers + 1)
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.register_buffer(
            "neighbor_scale",
            torch.tensor(1.0 / math.sqrt(self.avg_num_neighbors)),
            persistent=False,
        )

    def _check_config(self) -> None:
        if self.dim_edge < 2 or self.dim_edge % 2 != 0:
            raise ValueError(
                f"dim_edge must be a positive even integer, got {self.dim_edge}"
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {self.l_max}")
        if self.num_scalar_features < 1:
            raise ValueError(
                f"num_scalar_features must be >= 1, got {self.num_scalar_features}"
            )
        if self.num_tensor_features < 1:
            raise ValueError(
                f"num_tensor_features must be >= 1, got {self.num_tensor_features}"
            )
        if self.avg_num_neighbors <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {self.avg_num_neighbors}"
            )
        if self.scalar_embed_mlp_depth < 0 or self.latent_mlp_depth < 0:
            raise ValueError("MLP depths must be >= 0")
        if self.scalar_embed_mlp_depth > 0 and self.scalar_embed_mlp_width < 1:
            raise ValueError("scalar_embed_mlp_width must be >= 1 when depth > 0")
        if self.latent_mlp_depth > 0 and self.latent_mlp_width < 1:
            raise ValueError("latent_mlp_width must be >= 1 when depth > 0")

    def _mlp(self, n_in: int, n_out: int, depth: int, width: int) -> VariancePreservingMLP:
        return VariancePreservingMLP(
            n_in,
            n_out,
            hidden_layers_depth=depth,
            hidden_layers_width=width if depth > 0 else None,
            nonlinearity=self.nonlinearity if depth > 0 else None,
            bias=False,
            forward_weight_init=True,
        )

    def _linear(self, n_in: int, n_out: int) -> VariancePreservingMLP:
        return VariancePreservingMLP(
            n_in,
            n_out,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
            forward_weight_init=True,
        )

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        for layer in built_layers:
            if layer is self:
                continue
            dim_edge = getattr(layer, "dim_edge", None)
            if dim_edge is not None and int(dim_edge) != self.dim_edge:
                raise ValueError(
                    f"Allegro dim_edge {self.dim_edge} != embedding dim_edge {int(dim_edge)}"
                )

    def get_output(
        self,
        edge_embedding: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        """Return the dense scalar track, one row per directed edge.

        Centers are ``idx_i_sr``. Environment embeddings are summed on the
        center and divided by ``sqrt(avg_num_neighbors)`` before the tensor
        product.
        """
        if edge_embedding.shape[-1] != self.dim_edge:
            raise ValueError(
                f"edge_embedding width {edge_embedding.shape[-1]} != "
                f"dim_edge {self.dim_edge}"
            )
        if idx_i_sr.shape != idx_j_sr.shape:
            raise ValueError("idx_i_sr and idx_j_sr must have the same shape")
        if edge_embedding.shape[0] != idx_i_sr.shape[0] or vij_sr.shape[0] != idx_i_sr.shape[0]:
            raise ValueError(
                "edge_embedding, vij_sr, and the neighbor index must describe the same edges"
            )
        if vij_sr.ndim != 2 or vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must have shape [edges, 3], got {tuple(vij_sr.shape)}")
        dtype = self.scalar_mlp.net[0].weight.dtype
        embedded = edge_embedding.to(dtype=dtype)
        scalar = self.scalar_mlp(embedded)
        centers = idx_i_sr.long()
        if centers.numel() == 0:
            basis = embedded.new_zeros((0, self.irreps_edge_sh.dim))
        else:
            basis = self.sph(vij_sr.to(dtype=dtype))
        tensor_features = self.weighter(basis, self.tensor_weight(scalar))
        projected = self.pair_projection(scalar)
        accumulated = [projected[:, : self.num_scalar_features]]
        env_weight = projected[:, self.num_scalar_features :]
        n_atoms = int(centers.max().item()) + 1 if centers.numel() else 0
        scale = self.neighbor_scale.to(dtype=dtype)
        for product, latent in zip(self.products, self.latents):
            env_edges = self.weighter(basis, env_weight)
            env_atoms = scatter_sum(env_edges, centers, n_atoms) * scale
            tensor_features = product(tensor_features, env_atoms[centers])
            invariants = tensor_features[:, : self.num_tensor_features]
            updated = latent(torch.cat([*accumulated, invariants], dim=-1))
            accumulated.append(updated[:, : self.num_scalar_features])
            if updated.shape[-1] > self.num_scalar_features:
                env_weight = updated[:, self.num_scalar_features :]
        return {"edge_feature": torch.cat(accumulated, dim=-1)}
