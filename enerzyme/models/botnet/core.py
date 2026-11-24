"""BOTNet message-passing core.

Batatia et al., the design space of E(3)-equivariant atom-centred
interatomic potentials (Nat. Mach. Intell., doi:10.1038/s42256-024-00956-x,
arXiv:2205.06643). The reference implementation is
`gncs/botnet <https://github.com/gncs/botnet>`_.

Each interaction is one two-body Clebsch–Gordan convolution of the sender
features with real spherical harmonics. Stacking layers raises the body
order by one, and the site energy reads every layer. MACE keeps these
convolutions and inserts a symmetric product after each one; this Core does
not.

The Bessel × polynomial basis and the species embedding stay outside. The
per-layer energy head is :class:`~enerzyme.models.botnet.readout.BOTNetEnergyReadout`.
Species scale and shift are :class:`~enerzyme.models.layers.denormalize.AtomicAffineLayer`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps, SphericalHarmonics
from torch import Tensor

from ..mp.equivariant.e3nn import scalar_0e_dim
from ..layers import BaseFFCore
from .interaction import INTERACTION_CLASSES, VARIANTS


# CLI defaults in gncs/botnet (r_max 4 Å, 8 Bessel functions, polynomial
# order 6, 5 interactions, natural parity through l = 3, 32 channels).
# The checked-in stack is smaller. The 3BPA run in the README uses 80
# channels and both parities; later layers then grow extra irreps, which
# this Core zero-fills in the stacked latent.
PUBLISHED_CORE_PARAMS = {
    "max_ell": 3,
    "num_interactions": 5,
    "hidden_irreps": "32x0e+32x1o+32x2e+32x3o",
    "variant": "agnostic",
    "MLP_irreps": "16x0e",
    "radial_MLP": [64, 64, 64],
    "gate": "silu",
    "output_mode": "layers",
}
PUBLISHED_3BPA_HIDDEN_IRREPS = (
    "80x0e+80x0o+80x1o+80x1e+80x2e+80x2o+80x3o+80x3e"
)
PUBLISHED_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "cutoff_sr": 4.0,
    "polynomial_p": 6,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 16,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 4.0,
    "avg_num_neighbors": 1.0,
}
DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "mace", "trainable": False, "polynomial_p": 6},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "max_ell": 2,
            "num_interactions": 2,
            "variant": "agnostic",
            "output_mode": "layers",
            "MLP_irreps": "16x0e",
            "radial_MLP": [64, 64, 64],
            "gate": "silu",
        },
    },
    {
        "name": "BOTNetEnergyReadout",
        "params": {"MLP_irreps": "16x0e", "gate": "silu"},
    },
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
            "scales": {"Ea": {"values": 1.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def natural_hidden_irreps(num_channels: int, max_ell: int) -> str:
    """Even-scalar channels plus natural parity ``p = (-1)^l`` through ``max_ell``.

    A scalar embedding tensor-producted with ``Y_l`` produces exactly this set,
    and every later layer can produce it again, so the stacked latent has one
    irrep layout.
    """
    channels = int(num_channels)
    degree = int(max_ell)
    if channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if degree < 0:
        raise ValueError(f"max_ell must be >= 0, got {max_ell}")
    parts = []
    for ell in range(degree + 1):
        parity = "e" if ell % 2 == 0 else "o"
        parts.append(f"{channels}x{ell}{parity}")
    return str(Irreps("+".join(parts)).simplify())


def _scalar_0e_multiplicity(irreps: Irreps) -> int:
    return sum(mul for mul, ir in irreps if ir.l == 0 and ir.p == 1)


def _column_index(src: Irreps, dst: Irreps) -> Tensor:
    """Columns of ``dst`` that receive each column of ``src``, in ``src`` order."""
    dst = Irreps(dst).simplify()
    src = Irreps(src).simplify()
    starts = {}
    offset = 0
    for mul, ir in dst:
        starts[ir] = (offset, mul)
        offset += mul * ir.dim
    columns: List[int] = []
    for mul, ir in src:
        if ir not in starts:
            raise ValueError(f"{ir} is in a layer output but not in the stacked irreps")
        start, dst_mul = starts[ir]
        if mul != dst_mul:
            raise ValueError(
                f"Multiplicity of {ir} is {mul} in one layer and {dst_mul} in the stack"
            )
        width = mul * ir.dim
        columns.extend(range(start, start + width))
    if len(columns) != src.dim:
        raise ValueError(
            f"Placement for {src} has {len(columns)} columns, expected {src.dim}"
        )
    return torch.tensor(columns, dtype=torch.long)


def _union_irreps(layer_irreps: Sequence[Irreps]) -> Irreps:
    multiplicity = {}
    for irreps in layer_irreps:
        for mul, ir in Irreps(irreps).simplify():
            multiplicity[ir] = max(multiplicity.get(ir, 0), mul)
    if not multiplicity:
        raise ValueError("BOTNet produced empty irreps")
    return Irreps(
        [(multiplicity[ir], ir) for ir in sorted(multiplicity, key=lambda ir: (ir.l, -ir.p))]
    ).simplify()


def _place(features: Tensor, columns: Tensor, dim_out: int) -> Tensor:
    laid = features.new_zeros(features.shape[0], dim_out)
    laid[:, columns] = features
    return laid


class BOTNetCore(BaseFFCore):
    def __init__(
        self,
        max_Za: int,
        dim_embedding: int,
        num_rbf: int,
        max_ell: int = 2,
        num_interactions: int = 2,
        avg_num_neighbors: float = 1.0,
        hidden_irreps: Optional[str] = None,
        additional_hidden_irreps: Optional[str] = None,
        variant: str = "agnostic",
        interaction_cls_first: Optional[str] = None,
        interaction_cls: Optional[str] = None,
        MLP_irreps: str = "16x0e",
        radial_MLP: Optional[List[int]] = None,
        gate: str = "silu",
        output_mode: str = "layers",
    ) -> None:
        if output_mode not in {"layers", "feature"}:
            raise ValueError(
                f"output_mode must be 'layers' or 'feature', got {output_mode!r}"
            )
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
        self.output_mode = output_mode
        self.max_Za = int(max_Za)
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.max_ell = int(max_ell)
        self.num_interactions = int(num_interactions)
        self.avg_num_neighbors = float(avg_num_neighbors)
        if self.max_Za < 1:
            raise ValueError(f"max_Za must be >= 1, got {max_Za}")
        if self.dim_embedding < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if self.num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if self.max_ell < 0:
            raise ValueError(f"max_ell must be >= 0, got {max_ell}")
        if self.num_interactions < 1:
            raise ValueError(
                f"num_interactions must be >= 1, got {num_interactions}"
            )
        if self.avg_num_neighbors <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
            )
        if hidden_irreps is not None and additional_hidden_irreps:
            raise ValueError(
                "Pass hidden_irreps or additional_hidden_irreps, not both"
            )
        if hidden_irreps is None:
            if additional_hidden_irreps:
                hidden_irreps = f"{self.dim_embedding}x0e+{additional_hidden_irreps}"
            else:
                hidden_irreps = natural_hidden_irreps(self.dim_embedding, self.max_ell)
        target = Irreps(hidden_irreps).simplify()
        scalar_mul = _scalar_0e_multiplicity(target)
        if scalar_mul != self.dim_embedding:
            raise ValueError(
                "dim_embedding must equal the 0e multiplicity of hidden_irreps "
                f"({scalar_mul}), got {self.dim_embedding}. The species table is "
                "the initial scalar state."
            )
        self.requested_hidden_irreps = str(target)
        first_name, rest_name = _resolve_interactions(
            variant, interaction_cls_first, interaction_cls
        )
        self.variant = variant
        self.interaction_cls_first = first_name
        self.interaction_cls = rest_name
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = [int(width) for width in radial_MLP]
        if any(width < 1 for width in self.radial_MLP):
            raise ValueError(f"radial_MLP widths must be >= 1, got {self.radial_MLP}")
        self.readout_mlp_irreps = str(Irreps(MLP_irreps).simplify())
        self.readout_gate = str(gate)
        # One interaction has only the linear head. A second interaction is
        # what makes the final head the scalar MLP in NonLinearBodyOrderedModel.
        self.last_readout = "nonlinear" if self.num_interactions > 1 else "linear"

        sh_irreps = Irreps.spherical_harmonics(self.max_ell)
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        node_attrs_irreps = Irreps([(self.max_Za + 1, (0, 1))])
        edge_feats_irreps = Irreps([(self.num_rbf, (0, 1))])
        node_feats_irreps = Irreps([(self.dim_embedding, (0, 1))])
        self.interactions = torch.nn.ModuleList()
        layer_irreps: List[Irreps] = []
        for index in range(self.num_interactions):
            name = first_name if index == 0 else rest_name
            block = INTERACTION_CLASSES[name](
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target,
                hidden_irreps=target,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_MLP=self.radial_MLP,
            )
            self.interactions.append(block)
            produced = Irreps(block.irreps_out).simplify()
            layer_irreps.append(produced)
            node_feats_irreps = produced
        self.layer_irreps = layer_irreps
        self.uniform_irreps = all(irreps == layer_irreps[0] for irreps in layer_irreps)
        union = layer_irreps[0] if self.uniform_irreps else _union_irreps(layer_irreps)
        if self.output_mode == "layers":
            self.feature_irreps = str(union)
            self.num_output_blocks = self.num_interactions
        else:
            self.feature_irreps = str(layer_irreps[-1])
            self.num_output_blocks = 1
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)
        for index, irreps in enumerate(layer_irreps):
            columns = _column_index(irreps, union)
            self.register_buffer(f"placement_{index}", columns, persistent=False)
        self._union_dim = union.dim

    def __str__(self) -> str:
        return f"""
########################################################
# BOTNet (Nat. Mach. Intell., arXiv:2205.06643)        #
# output_mode={self.output_mode} features={self.feature_irreps}
# variant={self.variant} interactions={self.num_interactions}
########################################################
"""

    def build(self, built_layers) -> None:
        self.adopt_radius_stack(built_layers)

    def _messages(
        self,
        node_feats: Tensor,
        node_attrs: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> List[Tensor]:
        layers: List[Tensor] = []
        for interaction in self.interactions:
            node_feats = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                sender=idx_i_sr,
                receiver=idx_j_sr,
            )
            if not isinstance(node_feats, Tensor):
                raise TypeError(
                    "BOTNet interactions must return the updated node state. "
                    f"{type(interaction).__name__} returned {type(node_feats).__name__}."
                )
            layers.append(node_feats)
        return layers

    def _stack(self, layers: List[Tensor]) -> Tensor:
        if self.uniform_irreps:
            return torch.stack(layers, dim=-1)
        placed = [
            _place(features, getattr(self, f"placement_{index}"), self._union_dim)
            for index, features in enumerate(layers)
        ]
        return torch.stack(placed, dim=-1)

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
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be [N, dim_embedding="
                f"{self.dim_embedding}], got {tuple(atom_embedding.shape)}"
            )
        if idx_i_sr.shape != idx_j_sr.shape:
            raise ValueError("idx_i_sr and idx_j_sr must have the same shape")
        n_edges = idx_i_sr.shape[0]
        if rbf.shape[0] != n_edges or vij_sr.shape[0] != n_edges:
            raise ValueError(
                f"rbf and vij_sr must have one row per edge ({n_edges}), "
                f"got rbf {tuple(rbf.shape)} and vij {tuple(vij_sr.shape)}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf width must be num_rbf={self.num_rbf}, got {rbf.shape[-1]}"
            )
        if vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must be [E, 3], got {tuple(vij_sr.shape)}")
        atomic_numbers = Za.long().view(-1)
        n_atoms = atom_embedding.shape[0]
        if atomic_numbers.shape[0] != n_atoms:
            raise ValueError(
                f"Za has {atomic_numbers.shape[0]} atoms, embedding has {n_atoms}"
            )
        if atomic_numbers.numel():
            highest = int(atomic_numbers.max())
            lowest = int(atomic_numbers.min())
            if lowest < 0 or highest > self.max_Za:
                raise ValueError(
                    f"Za must lie in 0..{self.max_Za}, got {lowest}..{highest}"
                )
        node_feats = atom_embedding
        if charge_embedding is not None:
            node_feats = node_feats + charge_embedding
        if spin_embedding is not None:
            node_feats = node_feats + spin_embedding
        node_attrs = F.one_hot(atomic_numbers, num_classes=self.max_Za + 1).to(
            dtype=node_feats.dtype
        )
        edge_attrs = self.spherical_harmonics(vij_sr)
        layers = self._messages(
            node_feats, node_attrs, edge_attrs, rbf, idx_i_sr, idx_j_sr
        )
        if self.output_mode == "feature":
            return {"atom_feature": layers[-1]}
        return {"atom_feature": self._stack(layers)}


def _resolve_interactions(
    variant: str,
    interaction_cls_first: Optional[str],
    interaction_cls: Optional[str],
) -> tuple:
    if variant not in VARIANTS and (
        interaction_cls_first is None or interaction_cls is None
    ):
        raise ValueError(
            f"Unknown variant {variant!r}. Choices: {sorted(VARIANTS)}. "
            "Set both interaction class names to bypass the variant."
        )
    if variant in VARIANTS:
        default_first, default_rest = VARIANTS[variant]
    else:
        default_first, default_rest = interaction_cls_first, interaction_cls
    first = default_first if interaction_cls_first is None else interaction_cls_first
    rest = default_rest if interaction_cls is None else interaction_cls
    unknown = [name for name in (first, rest) if name not in INTERACTION_CLASSES]
    if unknown:
        raise KeyError(
            f"Unknown interaction class {unknown[0]!r}. "
            f"Choices: {sorted(INTERACTION_CLASSES)}"
        )
    return first, rest
