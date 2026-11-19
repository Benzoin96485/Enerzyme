"""CHGNet Core (Deng et al., Nat. Mach. Intell. 2023).

The Core maps a species embedding and two smooth Bessel bond bases to a
scalar atomic state. Directed edges are paired into undirected bonds. Each
block updates atoms, then bonds, then angles. The last block updates atoms
only. ``atom_feature_charge`` is the state before that last update, which the
magnetic-moment head reads. The energy MLP, the elemental reference, and
``Fa = -∇E`` stay outside.

``idx_i_sr`` is the receiver. The neighbor list is the atom-graph cutoff and
must contain both directions of every bond. ``bond_cutoff`` cannot exceed it.
The smooth radial envelope lives on the two ``BesselRBF(flavor="gemnet")``
layers, not in this Core.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from torch import Tensor
from torch.nn import Linear, Module, ModuleList

from ..layers._base_layer import BaseFFCore
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import BaseRBF, BesselRBFLayer
from .basis import FourierExpansion
from .geometry import bond_graph, directed_bond_angle, undirected_bond_index
from .interaction import AngleUpdate, AtomConv, BondConv, _feature_norm


HiddenSpec = Union[int, Sequence[int], None]


def _hidden_widths(spec: HiddenSpec, fallback: int, name: str) -> list[int]:
    if spec is None:
        spec = fallback
    if isinstance(spec, int):
        if spec == 0:
            return []
        if spec < 1:
            raise ValueError(f"{name} must be >= 0, got {spec}")
        return [int(spec)]
    widths = [int(width) for width in spec]
    if any(width < 1 for width in widths):
        raise ValueError(f"{name} widths must be >= 1, got {widths}")
    return widths


def _norm_choice(value: Optional[str], name: str) -> Optional[str]:
    if value in (None, "none", "null"):
        return None
    if value not in ("layer", "batch"):
        raise ValueError(f"{name} must be 'layer', 'batch', or null, got {value!r}")
    return str(value)


class CHGNetCore(BaseFFCore):
    """Atom, bond, and angle updates.

    Emits ``atom_feature`` ``(N, dim_embedding)`` after the last atom
    convolution and optional readout normalization, and
    ``atom_feature_charge`` with the same shape from the previous state.
    ``feature_irreps = "{F}x0e"``. An isolated atom keeps its species
    embedding until that normalization.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        bond_cutoff: float = 3.0,
        num_angular: int = 9,
        num_blocks: int = 4,
        dim_bond: Optional[int] = None,
        dim_angle: Optional[int] = None,
        atom_hidden: HiddenSpec = None,
        bond_hidden: HiddenSpec = None,
        angle_hidden: HiddenSpec = 0,
        dropout: float = 0.0,
        envelope_exponent: int = 5,
        gmlp_norm: Optional[str] = "layer",
        readout_norm: Optional[str] = "layer",
        conv_norm: Optional[str] = None,
        update_bond: bool = True,
        update_angle: bool = True,
        learnable_frequency: bool = True,
        mlp_out_bias: bool = False,
    ) -> None:
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(num_angular) < 1 or int(num_angular) % 2 == 0:
            raise ValueError(f"num_angular must be a positive odd integer, got {num_angular}")
        if int(num_blocks) < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if float(bond_cutoff) <= 0:
            raise ValueError(f"bond_cutoff must be positive, got {bond_cutoff}")
        if float(bond_cutoff) > float(cutoff_sr) + 1e-6:
            raise ValueError(
                f"bond_cutoff ({bond_cutoff}) cannot exceed cutoff_sr ({cutoff_sr}): "
                "angles are pairs of edges inside the neighbor list"
            )
        if float(dropout) < 0 or float(dropout) >= 1:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        if int(envelope_exponent) < 1:
            raise ValueError(
                f"envelope_exponent must be >= 1, got {envelope_exponent}"
            )
        if update_angle and not update_bond:
            raise ValueError(
                "update_angle requires update_bond: the reference network "
                "skips the angle update when the bond convolution is off"
            )
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "rbf_bg",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
            },
            output_fields={"atom_feature", "atom_feature_charge"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.bond_cutoff = float(bond_cutoff)
        self.num_angular = int(num_angular)
        self.num_blocks = int(num_blocks)
        self.dim_bond = int(self.dim_embedding if dim_bond is None else dim_bond)
        self.dim_angle = int(self.dim_embedding if dim_angle is None else dim_angle)
        if self.dim_bond < 1 or self.dim_angle < 1:
            raise ValueError(
                f"dim_bond and dim_angle must be >= 1, got {dim_bond}, {dim_angle}"
            )
        self.atom_hidden = _hidden_widths(atom_hidden, self.dim_embedding, "atom_hidden")
        self.bond_hidden = _hidden_widths(bond_hidden, self.dim_embedding, "bond_hidden")
        self.angle_hidden = _hidden_widths(angle_hidden, 0, "angle_hidden")
        self.dropout = float(dropout)
        self.envelope_exponent = int(envelope_exponent)
        self.gmlp_norm = _norm_choice(gmlp_norm, "gmlp_norm")
        self.readout_norm_name = _norm_choice(readout_norm, "readout_norm")
        self.conv_norm = _norm_choice(conv_norm, "conv_norm")
        self.update_bond = bool(update_bond)
        self.update_angle = bool(update_angle)
        self.learnable_frequency = bool(learnable_frequency)
        self.mlp_out_bias = bool(mlp_out_bias)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_embedding}x0e"
        order = (self.num_angular - 1) // 2

        self.bond_embed = Linear(self.num_rbf, self.dim_bond, bias=False)
        self.atom_weight = Linear(self.num_rbf, self.dim_embedding, bias=False)
        self.bond_weight = Linear(self.num_rbf, self.dim_bond, bias=False)
        self.angle_embed = Linear(self.num_angular, self.dim_angle, bias=False)
        self.fourier = FourierExpansion(order, learnable=self.learnable_frequency)
        self.atom_convs = ModuleList(
            [
                AtomConv(
                    self.dim_embedding,
                    self.dim_bond,
                    self.atom_hidden,
                    self.dropout,
                    self.gmlp_norm,
                    self.conv_norm,
                    self.mlp_out_bias,
                )
                for _ in range(self.num_blocks)
            ]
        )
        n_line = self.num_blocks - 1
        self.bond_convs = ModuleList(
            [
                BondConv(
                    self.dim_embedding,
                    self.dim_bond,
                    self.dim_angle,
                    self.bond_hidden,
                    self.dropout,
                    self.gmlp_norm,
                    self.conv_norm,
                    self.mlp_out_bias,
                )
                for _ in range(n_line if self.update_bond else 0)
            ]
        )
        self.angle_updates = ModuleList(
            [
                AngleUpdate(
                    self.dim_embedding,
                    self.dim_bond,
                    self.dim_angle,
                    self.angle_hidden,
                    self.dropout,
                    self.gmlp_norm,
                    self.conv_norm,
                )
                for _ in range(n_line if self.update_angle else 0)
            ]
        )
        self.readout_norm = _feature_norm(self.readout_norm_name, self.dim_embedding)

    def __str__(self) -> str:
        return """
#########################################################################
# CHGNet (Deng et al., Nat. Mach. Intell. 2023)                        #
# Atom, bond, and angle graphs. Magmom head and energy MLP stay outside.#
#########################################################################
"""

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        for candidate in built_layers:
            if isinstance(candidate, BaseAtomEmbedding):
                if int(candidate.dim_embedding) != self.dim_embedding:
                    raise ValueError(
                        "Atom embedding width "
                        f"({candidate.dim_embedding}) must equal dim_embedding "
                        f"({self.dim_embedding})"
                    )
                self.atom_embedding = candidate
            if isinstance(candidate, RangeSeparationLayer):
                if abs(float(candidate.cutoff_sr) - self.cutoff_sr) > 1e-4:
                    raise ValueError(
                        "RangeSeparation cutoff_sr "
                        f"({candidate.cutoff_sr}) must equal CHGNet cutoff_sr "
                        f"({self.cutoff_sr})"
                    )
        radial = [layer for layer in built_layers if isinstance(layer, BaseRBF)]
        gemnet = [
            layer
            for layer in radial
            if isinstance(layer, BesselRBFLayer) and getattr(layer, "flavor", None) == "gemnet"
        ]
        if len(radial) != 2 or len(gemnet) != 2:
            found = [
                f"{type(layer).__name__}:{getattr(layer, 'flavor', None)}"
                for layer in radial
            ]
            raise ValueError(
                "CHGNetCore requires exactly two BesselRBF layers with flavor "
                f"'gemnet' before the Core, found {found or 'none'}"
            )
        same_cutoff = abs(self.bond_cutoff - self.cutoff_sr) <= 1e-4
        if same_cutoff:
            atom_basis, bond_basis = gemnet[0], gemnet[1]
        else:
            atom_layers = [
                layer
                for layer in gemnet
                if abs(float(layer.basis_cutoff) - self.cutoff_sr) <= 1e-4
                and abs(float(layer.cutoff_sr) - self.cutoff_sr) <= 1e-4
            ]
            bond_layers = [
                layer
                for layer in gemnet
                if abs(float(layer.basis_cutoff) - self.bond_cutoff) <= 1e-4
            ]
            if len(atom_layers) != 1 or len(bond_layers) != 1 or atom_layers[0] is bond_layers[0]:
                raise ValueError(
                    "CHGNetCore needs one gemnet Bessel basis at cutoff_sr "
                    f"({self.cutoff_sr}) and one at bond_cutoff ({self.bond_cutoff}). "
                    "Set the bond layer's cutoff_sr to bond_cutoff."
                )
            atom_basis, bond_basis = atom_layers[0], bond_layers[0]
        for layer, expected in (
            (atom_basis, self.cutoff_sr),
            (bond_basis, self.bond_cutoff),
        ):
            if int(layer.num_rbf) != self.num_rbf:
                raise ValueError(
                    "BesselRBF num_rbf "
                    f"({layer.num_rbf}) must equal the Core num_rbf ({self.num_rbf})"
                )
            if int(layer.envelope_exponent) != self.envelope_exponent:
                raise ValueError(
                    "BesselRBF envelope_exponent "
                    f"({layer.envelope_exponent}) must equal the Core "
                    f"envelope_exponent ({self.envelope_exponent})"
                )
            if abs(float(layer.basis_cutoff) - float(expected)) > 1e-4:
                raise ValueError(
                    "BesselRBF basis_cutoff "
                    f"({layer.basis_cutoff}) must equal {expected}"
                )
        bond_basis.reset_field_name(rbf="rbf_bg")
        self.atom_basis = atom_basis
        self.bond_basis = bond_basis

    def _initial_state(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        rbf_bg: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ):
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be "
                f"(N, {self.dim_embedding}), got {tuple(atom_embedding.shape)}"
            )
        n_edges = int(Dij_sr.shape[0])
        if Dij_sr.ndim != 1:
            raise ValueError(f"Dij_sr must be (E,), got {tuple(Dij_sr.shape)}")
        if vij_sr.ndim != 2 or vij_sr.shape[-1] != 3 or vij_sr.shape[0] != n_edges:
            raise ValueError(f"vij_sr must be (E, 3), got {tuple(vij_sr.shape)}")
        if idx_i_sr.shape[0] != n_edges or idx_j_sr.shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {n_edges}, {idx_i_sr.shape[0]}, {idx_j_sr.shape[0]}"
            )
        for name, basis in (("rbf", rbf), ("rbf_bg", rbf_bg)):
            if basis.ndim != 2 or basis.shape[0] != n_edges or basis.shape[-1] != self.num_rbf:
                raise ValueError(
                    f"{name} must be (E, {self.num_rbf}), got {tuple(basis.shape)} "
                    f"for {n_edges} edges"
                )
        n_atoms = int(atom_embedding.shape[0])
        bonds = undirected_bond_index(idx_i_sr, idx_j_sr, vij_sr)
        angles = bond_graph(idx_i_sr, Dij_sr, n_atoms, self.bond_cutoff)
        if bonds.representative.numel() == 0:
            bond = self.bond_embed(rbf.new_zeros(0, self.num_rbf))
            atom_weight = self.atom_weight(rbf.new_zeros(0, self.num_rbf))
            bond_weight = self.bond_weight(rbf_bg.new_zeros(0, self.num_rbf))
        else:
            bond = self.bond_embed(rbf.index_select(0, bonds.representative))
            atom_weight = self.atom_weight(rbf.index_select(0, bonds.representative))
            bond_weight = self.bond_weight(rbf_bg.index_select(0, bonds.representative))
        if angles.idx_ij.numel() == 0:
            angle = self.angle_embed(rbf.new_zeros(0, self.num_angular))
        else:
            theta = directed_bond_angle(vij_sr, angles.idx_ij, angles.idx_ik)
            angle = self.angle_embed(self.fourier(theta))
        return (
            atom_embedding,
            bond,
            atom_weight,
            bond_weight,
            angle,
            bonds.directed_to_undirected,
            angles,
        )

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        rbf_bg: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> dict[str, Tensor]:
        node, edge, atom_weight, bond_weight, angle, directed, angles = self._initial_state(
            atom_embedding, rbf, rbf_bg, Dij_sr, vij_sr, idx_i_sr, idx_j_sr
        )
        charge = node
        for step, atom_conv in enumerate(self.atom_convs):
            if step == self.num_blocks - 1:
                charge = node
            node = atom_conv(
                node,
                edge,
                atom_weight,
                idx_i_sr,
                idx_j_sr,
                directed,
            )
            if step >= len(self.bond_convs) or angle.shape[0] == 0:
                continue
            edge = self.bond_convs[step](
                node,
                edge,
                bond_weight,
                angle,
                idx_i_sr[angles.idx_ij],
                directed[angles.idx_ij],
                directed[angles.idx_ik],
            )
            if step < len(self.angle_updates):
                angle = self.angle_updates[step](
                    node,
                    edge,
                    angle,
                    idx_i_sr[angles.idx_ij],
                    directed[angles.idx_ij],
                    directed[angles.idx_ik],
                )
        if self.readout_norm is not None:
            node = self.readout_norm(node)
        return {"atom_feature": node, "atom_feature_charge": charge}
