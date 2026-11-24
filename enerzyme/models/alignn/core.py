"""ALIGNN Core (Choudhary and DeCost, npj Comput. Mater. 2021).

The Core maps a species embedding and a Gaussian bond basis to a scalar
atomic state. Bond-angle cosines are expanded inside the Core because they
live on the directed line graph, which is part of the interaction. The
species table, the distance Gaussian, the energy head, and ``Fa = -∇E``
stay outside.

``idx_i_sr`` is the destination and ``idx_j_sr`` is the source.
``vij_sr = R[source] - R[destination]``. The neighbor list is the graph
cutoff. The released force field does not put a smooth envelope on the
messages; ``multiply_envelope`` scales only the initial bond embedding by
:func:`enerzyme.models.mp.basis.envelope.polynomial_envelope`, the same optional
cutoff as ``ALIGNNAtomWise``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..mp.basis.envelope import polynomial_envelope
from ..layers._base_layer import BaseFFCore
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import BaseRBF, GaussianRBFLayer, alignn_gaussian_terms
from .geometry import directed_line_graph, line_graph_cosine
from .interaction import ALIGNNConv, EdgeGatedGraphConv, FeatureMLP, embedding_mlp


def _norm_name(value: Optional[str]) -> Optional[str]:
    if value in (None, "none", "null"):
        return None
    if value not in ("layer", "batch"):
        raise ValueError(f"norm must be 'layer', 'batch', or null, got {value!r}")
    return str(value)


class ALIGNNCore(BaseFFCore):
    """Edge-gated message passing on the bond graph and its line graph.

    Emits ``atom_feature`` with shape ``(N, hidden_features)`` and
    ``feature_irreps = "{F}x0e"``. An isolated atom still passes through the
    self branch of every convolution. No neighbor or angle message is added,
    so that state does not depend on coordinates.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        num_angle_rbf: int = 40,
        embedding_features: int = 64,
        hidden_features: Optional[int] = None,
        alignn_layers: int = 2,
        gcn_layers: int = 2,
        norm: Optional[str] = "layer",
        residual: bool = True,
        update_order: str = "code",
        backtracking: bool = True,
        multiply_envelope: bool = False,
        envelope_exponent: int = 5,
        gate_eps: float = 1e-6,
    ) -> None:
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 2:
            raise ValueError(f"num_rbf must be >= 2, got {num_rbf}")
        if int(num_angle_rbf) < 2:
            raise ValueError(f"num_angle_rbf must be >= 2, got {num_angle_rbf}")
        if int(embedding_features) < 1:
            raise ValueError(f"embedding_features must be >= 1, got {embedding_features}")
        if int(alignn_layers) < 0:
            raise ValueError(f"alignn_layers must be >= 0, got {alignn_layers}")
        if int(gcn_layers) < 0:
            raise ValueError(f"gcn_layers must be >= 0, got {gcn_layers}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if update_order not in ("code", "paper"):
            raise ValueError(f"update_order must be 'code' or 'paper', got {update_order!r}")
        if int(envelope_exponent) < 1:
            raise ValueError(f"envelope_exponent must be >= 1, got {envelope_exponent}")
        if float(gate_eps) <= 0:
            raise ValueError(f"gate_eps must be positive, got {gate_eps}")
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
            },
            output_fields={"atom_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_angle_rbf = int(num_angle_rbf)
        self.embedding_features = int(embedding_features)
        self.hidden_features = int(
            self.dim_embedding if hidden_features is None else hidden_features
        )
        if self.hidden_features < 1:
            raise ValueError(f"hidden_features must be >= 1, got {hidden_features}")
        self.alignn_layers = int(alignn_layers)
        self.gcn_layers = int(gcn_layers)
        self.norm = _norm_name(norm)
        self.residual = bool(residual)
        self.update_order = str(update_order)
        self.backtracking = bool(backtracking)
        self.multiply_envelope = bool(multiply_envelope)
        self.envelope_exponent = int(envelope_exponent)
        self.gate_eps = float(gate_eps)
        self.dim_feature_out = self.hidden_features
        self.feature_irreps = f"{self.hidden_features}x0e"

        self.atom_mlp = FeatureMLP(self.dim_embedding, self.hidden_features, self.norm)
        self.edge_mlp = embedding_mlp(
            self.num_rbf,
            self.embedding_features,
            self.hidden_features,
            self.norm,
        )
        width = self.hidden_features
        if self.alignn_layers > 0:
            centers, coeff = alignn_gaussian_terms(-1.0, 1.0, self.num_angle_rbf)
            self.register_buffer("angle_centers", centers)
            self.register_buffer("angle_coeff", coeff)
            self.angle_mlp = embedding_mlp(
                self.num_angle_rbf,
                self.embedding_features,
                width,
                self.norm,
            )
            self.alignn = ModuleList(
                [
                    ALIGNNConv(width, self.norm, self.residual, self.gate_eps)
                    for _ in range(self.alignn_layers)
                ]
            )
        else:
            self.angle_mlp = None
            self.alignn = ModuleList()
        self.gcn = ModuleList(
            [
                EdgeGatedGraphConv(width, self.norm, self.residual, self.gate_eps)
                for _ in range(self.gcn_layers)
            ]
        )

    def __str__(self) -> str:
        return """
#########################################################################
# ALIGNN (Choudhary and DeCost, npj Comput. Mater. 2021)               #
# Edge-gated bond graph and directed line graph. Readout stays outside.#
#########################################################################
"""

    def build(self, built_layers: list[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        for candidate in built_layers:
            if isinstance(candidate, BaseAtomEmbedding):
                if int(candidate.dim_embedding) != self.dim_embedding:
                    raise ValueError(
                        "Atom embedding width "
                        f"({candidate.dim_embedding}) must equal dim_embedding "
                        f"({self.dim_embedding})"
                    )
                self.atom_embedding_layer = candidate
            if isinstance(candidate, RangeSeparationLayer):
                if abs(float(candidate.cutoff_sr) - self.cutoff_sr) > 1e-4:
                    raise ValueError(
                        "RangeSeparation cutoff_sr "
                        f"({candidate.cutoff_sr}) must equal ALIGNN cutoff_sr "
                        f"({self.cutoff_sr})"
                    )
        radial = [layer for layer in built_layers if isinstance(layer, BaseRBF)]
        gaussian = [
            layer
            for layer in radial
            if isinstance(layer, GaussianRBFLayer) and getattr(layer, "flavor", None) == "alignn"
        ]
        if len(radial) != 1 or len(gaussian) != 1:
            found = [
                f"{type(layer).__name__}:{getattr(layer, 'flavor', None)}"
                for layer in radial
            ]
            raise ValueError(
                "ALIGNNCore requires exactly one GaussianRBF layer with flavor "
                f"'alignn' before the Core, found {found or 'none'}"
            )
        layer = gaussian[0]
        if int(layer.num_rbf) != self.num_rbf:
            raise ValueError(
                "GaussianRBF num_rbf "
                f"({layer.num_rbf}) must equal the Core num_rbf ({self.num_rbf})"
            )
        if abs(float(layer.cutoff_sr) - self.cutoff_sr) > 1e-4:
            raise ValueError(
                "GaussianRBF cutoff_sr "
                f"({layer.cutoff_sr}) must equal ALIGNN cutoff_sr ({self.cutoff_sr})"
            )
        self.radial_basis_function = layer

    def _embed(self, atom_embedding: Tensor, rbf: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
        atom = self.atom_mlp(atom_embedding)
        bond = self.edge_mlp(rbf)
        if self.multiply_envelope:
            envelope = polynomial_envelope(distance, self.cutoff_sr, self.envelope_exponent)
            bond = bond * envelope.unsqueeze(-1)
        return atom, bond

    def _angles(
        self,
        vij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        n_atoms: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        src_edge, dst_edge = directed_line_graph(
            idx_i,
            idx_j,
            n_atoms,
            backtracking=self.backtracking,
            vij=vij,
        )
        cosine = line_graph_cosine(vij, src_edge, dst_edge)
        expanded = torch.exp(
            self.angle_coeff * (cosine.unsqueeze(-1) - self.angle_centers) ** 2
        )
        return src_edge, dst_edge, self.angle_mlp(expanded)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> dict[str, Tensor]:
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be "
                f"(N, {self.dim_embedding}), got {tuple(atom_embedding.shape)}"
            )
        n_edges = int(idx_i_sr.shape[0])
        if idx_j_sr.shape[0] != n_edges:
            raise ValueError(
                "idx_i_sr and idx_j_sr must share the edge count, "
                f"got {n_edges} and {idx_j_sr.shape[0]}"
            )
        if Dij_sr.ndim != 1 or Dij_sr.shape[0] != n_edges:
            raise ValueError(f"Dij_sr must be ({n_edges},), got {tuple(Dij_sr.shape)}")
        if vij_sr.ndim != 2 or vij_sr.shape[0] != n_edges or vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must be ({n_edges}, 3), got {tuple(vij_sr.shape)}")
        if rbf.ndim != 2 or rbf.shape[0] != n_edges or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must be ({n_edges}, {self.num_rbf}), got {tuple(rbf.shape)}"
            )
        n_atoms = int(atom_embedding.shape[0])
        if n_edges and (
            int(idx_i_sr.max()) >= n_atoms
            or int(idx_j_sr.max()) >= n_atoms
            or int(idx_i_sr.min()) < 0
            or int(idx_j_sr.min()) < 0
        ):
            raise ValueError(
                f"edge indices must lie in 0..{n_atoms - 1}"
            )
        atom, bond = self._embed(atom_embedding, rbf, Dij_sr)
        if self.alignn_layers > 0:
            line_src, line_dst, angle = self._angles(vij_sr, idx_i_sr, idx_j_sr, n_atoms)
            for layer in self.alignn:
                atom, bond, angle = layer(
                    atom,
                    bond,
                    angle,
                    idx_j_sr,
                    idx_i_sr,
                    line_src,
                    line_dst,
                    self.update_order,
                )
        for layer in self.gcn:
            atom, bond = layer(atom, bond, idx_j_sr, idx_i_sr)
        return {"atom_feature": atom}
