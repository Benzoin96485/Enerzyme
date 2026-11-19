"""M3GNet Core (Chen and Ong, Nat. Comput. Sci. 2022).

The Core maps a species embedding and a spherical-Bessel bond basis to a
scalar atomic state. Each block updates bonds with a three-body sum, then
runs a gated graph convolution. The energy head, the elemental affine map,
and the force derivative stay outside.

``idx_i_sr`` is the receiver. ``idx_j_sr`` is the sender.
``vij_sr = R[sender] - R[receiver]``. The neighbor list is the graph cutoff.
The three-body polynomial envelope uses ``threebody_cutoff``, which must not
exceed that list.
"""

from __future__ import annotations

from typing import List, Optional

from torch import Tensor
from torch.nn import Linear, Module, ModuleList, SiLU, Sequential

from ..cutoff import polynomial_envelope
from ..layers._base_layer import BaseFFCore
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.rbf import BaseRBF
from .basis import joint_zonal_basis
from .geometry import bond_angle, receiver_triplets
from .interaction import M3GNetBlock


class M3GNetCore(BaseFFCore):
    """Three-body bond update and gated convolution.

    Emits ``atom_feature`` with shape ``(N, dim_embedding)`` and
    ``feature_irreps = "{F}x0e"``. An isolated atom keeps its species
    embedding: the gated readout, not this Core, turns that vector into an
    elemental baseline.
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
    ) -> None:
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1 or int(num_rbf) > 64:
            raise ValueError(f"num_rbf must lie in 1..64, got {num_rbf}")
        if int(max_l) < 1:
            raise ValueError(f"max_l must be >= 1, got {max_l}")
        if int(num_blocks) < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if float(threebody_cutoff) <= 0:
            raise ValueError(
                f"threebody_cutoff must be positive, got {threebody_cutoff}"
            )
        if float(threebody_cutoff) > float(cutoff_sr) + 1e-6:
            raise ValueError(
                "threebody_cutoff "
                f"({threebody_cutoff}) cannot exceed cutoff_sr ({cutoff_sr}): "
                "edges outside the neighbor list are not available"
            )
        if int(n_conv_hidden) < 0:
            raise ValueError(f"n_conv_hidden must be >= 0, got {n_conv_hidden}")
        if float(dropout) < 0 or float(dropout) >= 1:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        if int(envelope_exponent) < 1:
            raise ValueError(
                f"envelope_exponent must be >= 1, got {envelope_exponent}"
            )
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
        self.max_l = int(max_l)
        self.degree = self.max_l * self.num_rbf
        self.cutoff_sr = float(cutoff_sr)
        self.threebody_cutoff = float(threebody_cutoff)
        self.num_blocks = int(num_blocks)
        self.units = int(dim_embedding if units is None else units)
        if self.units < 1:
            raise ValueError(f"units must be >= 1, got {units}")
        self.n_conv_hidden = int(n_conv_hidden)
        self.dim_edge = int(
            dim_embedding if dim_edge_embedding is None else dim_edge_embedding
        )
        if self.dim_edge < 1:
            raise ValueError(
                f"dim_edge_embedding must be >= 1, got {dim_edge_embedding}"
            )
        self.dropout = float(dropout)
        self.envelope_exponent = int(envelope_exponent)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_embedding}x0e"
        self.readout_hidden_width = self.units

        self.edge_embed = Sequential(
            Linear(self.degree, self.dim_edge),
            SiLU(),
        )
        self.blocks = ModuleList(
            [
                M3GNetBlock(
                    degree=self.degree,
                    dim_node=self.dim_embedding,
                    dim_edge=self.dim_edge,
                    units=self.units,
                    n_conv_hidden=self.n_conv_hidden,
                    dropout=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def __str__(self) -> str:
        return """
#########################################################################
# M3GNet (Chen and Ong, Nat. Comput. Sci. 2022)                        #
# Three-body bonds + gated convolution. Readout and references outside.#
#########################################################################
"""

    def _allow_extra_radial(self) -> bool:
        """Subclasses that add another basis (MatterSim's Gaussian) return True."""
        return False

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        radial = [
            layer for layer in built_layers if isinstance(layer, BaseRBF)
        ]
        spherical = [
            layer for layer in radial if getattr(layer, "flavor", None) == "m3gnet"
        ]
        extra = [layer for layer in radial if layer not in spherical]
        if len(spherical) != 1:
            if len(radial) == 1:
                flavor = getattr(radial[0], "flavor", None)
                raise ValueError(
                    "M3GNetCore requires SphericalBesselRBF (flavor 'm3gnet'), "
                    f"found {flavor!r}"
                )
            raise ValueError(
                "M3GNetCore requires exactly one SphericalBesselRBF layer "
                f"before the Core, found {len(spherical)}"
            )
        if extra and not self._allow_extra_radial():
            raise ValueError(
                "M3GNetCore requires exactly one SphericalBesselRBF layer "
                f"before the Core, found {len(radial)}"
            )
        layer = spherical[0]
        if int(layer.max_l) != self.max_l or int(layer.num_rbf) != self.num_rbf:
            raise ValueError(
                "SphericalBesselRBF "
                f"(max_l={layer.max_l}, num_rbf={layer.num_rbf}) must match "
                f"the Core (max_l={self.max_l}, num_rbf={self.num_rbf})"
            )
        if abs(float(layer.cutoff_sr) - self.cutoff_sr) > 1e-4:
            raise ValueError(
                "SphericalBesselRBF cutoff_sr "
                f"({layer.cutoff_sr}) must equal M3GNet cutoff_sr "
                f"({self.cutoff_sr})"
            )
        self.radial_basis_function = layer
        for candidate in built_layers:
            if isinstance(candidate, BaseAtomEmbedding):
                self.atom_embedding = candidate

    def _featurize(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ):
        """Species state, initial bond state, and the three-body geometry."""
        if atom_embedding.ndim != 2:
            raise ValueError(
                f"atom_embedding must be (N, F), got {tuple(atom_embedding.shape)}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.degree:
            raise ValueError(
                f"rbf must be (E, {self.degree}), got {tuple(rbf.shape)}"
            )
        n_edges = int(Dij_sr.shape[0])
        if rbf.shape[0] != n_edges or idx_i_sr.shape[0] != n_edges or idx_j_sr.shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, rbf, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {n_edges}, {rbf.shape[0]}, {idx_i_sr.shape[0]}, {idx_j_sr.shape[0]}"
            )
        if vij_sr.shape[0] != n_edges or vij_sr.shape[-1] != 3:
            raise ValueError(
                f"vij_sr must be (E, 3), got {tuple(vij_sr.shape)}"
            )
        n_atoms = int(atom_embedding.shape[0])
        pairs = receiver_triplets(
            idx_i_sr, Dij_sr, n_atoms, self.threebody_cutoff
        )
        angle = bond_angle(vij_sr, pairs.idx_ij, pairs.idx_ik)
        joint = joint_zonal_basis(
            rbf, angle, pairs.idx_ik, self.max_l, self.num_rbf
        )
        envelope = polynomial_envelope(
            Dij_sr, self.threebody_cutoff, self.envelope_exponent
        )
        node = atom_embedding
        edge = self.edge_embed(rbf)
        return node, edge, pairs, joint, envelope

    def _run_blocks(self, node, edge, rbf, idx_i_sr, idx_j_sr, pairs, joint, envelope):
        for block in self.blocks:
            node, edge = block(
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
        return node, edge

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> dict[str, Tensor]:
        node, edge, pairs, joint, envelope = self._featurize(
            atom_embedding, rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr
        )
        node, _edge = self._run_blocks(
            node, edge, rbf, idx_i_sr, idx_j_sr, pairs, joint, envelope
        )
        return {"atom_feature": node}
