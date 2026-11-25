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

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.head import Head

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import BaseRBF, BesselRBFLayer
from .geometry import BondGraphRequest, directed_bond_angle


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


class CHGNetEmbed(Embed):
    def __init__(
        self,
        dim_embedding,
        num_rbf,
        bond_cutoff,
        num_angular,
        bond_embed,
        atom_weight,
        bond_weight,
        angle_embed,
        fourier,
    ):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.bond_cutoff = bond_cutoff
        self.num_angular = num_angular
        self.bond_embed = bond_embed
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.angle_embed = angle_embed
        self.fourier = fourier

    def _initial_state(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        rbf_bg: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        ctx,
    ):
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be " f"(N, {self.dim_embedding}), got {tuple(atom_embedding.shape)}"
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
                    f"{name} must be (E, {self.num_rbf}), got {tuple(basis.shape)} " f"for {n_edges} edges"
                )
        bonds, angles = BondGraphRequest(self.bond_cutoff).prepare(ctx)
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

    def forward(self, inputs, ctx):
        node, bond, atom_weight, bond_weight, angle, directed, angles = self._initial_state(
            inputs["atom_embedding"],
            ctx.rbf,
            inputs["rbf_bg"],
            ctx.dist,
            ctx.vec,
            ctx.receiver,
            ctx.sender,
            ctx,
        )
        ctx.cache["atom_graph"] = atom_weight, directed
        if angle.shape[0]:
            ctx.cache["bond_graph"] = (
                bond_weight,
                ctx.receiver[angles.idx_ij],
                directed[angles.idx_ij],
                directed[angles.idx_ik],
            )
        return BlockState(scalar=node, equi=angle, edge=bond)


def build_stack(core, built_layers: List[Module]) -> None:
    core.adopt_radius_stack(built_layers)
    for candidate in built_layers:
        if isinstance(candidate, BaseAtomEmbedding):
            if int(candidate.dim_embedding) != core.dim_embedding:
                raise ValueError(
                    "Atom embedding width "
                    f"({candidate.dim_embedding}) must equal dim_embedding "
                    f"({core.dim_embedding})"
                )
            core.atom_embedding = candidate
        if isinstance(candidate, RangeSeparationLayer):
            if abs(float(candidate.cutoff_sr) - core.cutoff_sr) > 1e-4:
                raise ValueError(
                    "RangeSeparation cutoff_sr "
                    f"({candidate.cutoff_sr}) must equal CHGNet cutoff_sr "
                    f"({core.cutoff_sr})"
                )
    radial = [layer for layer in built_layers if isinstance(layer, BaseRBF)]
    gemnet = [
        layer
        for layer in radial
        if isinstance(layer, BesselRBFLayer) and getattr(layer, "flavor", None) == "gemnet"
    ]
    if len(radial) != 2 or len(gemnet) != 2:
        found = [f"{type(layer).__name__}:{getattr(layer, 'flavor', None)}" for layer in radial]
        raise ValueError(
            "CHGNetCore requires exactly two BesselRBF layers with flavor "
            f"'gemnet' before the Core, found {found or 'none'}"
        )

    def _at_cutoff(layer: BesselRBFLayer, cutoff: float) -> bool:
        return (
            abs(float(layer.basis_cutoff) - cutoff) <= 1e-4 and abs(float(layer.cutoff_sr) - cutoff) <= 1e-4
        )

    atom_layers = [layer for layer in gemnet if _at_cutoff(layer, core.cutoff_sr)]
    bond_layers = [layer for layer in gemnet if _at_cutoff(layer, core.bond_cutoff)]
    same_cutoff = abs(core.bond_cutoff - core.cutoff_sr) <= 1e-4
    if same_cutoff:
        if len(atom_layers) != 2:
            raise ValueError(
                "When bond_cutoff equals cutoff_sr "
                f"({core.cutoff_sr}), both gemnet Bessel bases must set "
                "cutoff_sr and basis_cutoff to that value. The first "
                "layer is the atom graph and the second is the bond graph."
            )
        atom_basis, bond_basis = atom_layers[0], atom_layers[1]
    elif len(atom_layers) != 1 or len(bond_layers) != 1 or atom_layers[0] is bond_layers[0]:
        raise ValueError(
            "CHGNetCore needs one gemnet Bessel basis at cutoff_sr "
            f"({core.cutoff_sr}) and one at bond_cutoff ({core.bond_cutoff}). "
            "Set the bond layer's cutoff_sr and basis_cutoff to bond_cutoff."
        )
    else:
        atom_basis, bond_basis = atom_layers[0], bond_layers[0]
    for layer, expected in (
        (atom_basis, core.cutoff_sr),
        (bond_basis, core.bond_cutoff),
    ):
        if int(layer.num_rbf) != core.num_rbf:
            raise ValueError(
                "BesselRBF num_rbf " f"({layer.num_rbf}) must equal the Core num_rbf ({core.num_rbf})"
            )
        if int(layer.envelope_exponent) != core.envelope_exponent:
            raise ValueError(
                "BesselRBF envelope_exponent "
                f"({layer.envelope_exponent}) must equal the Core "
                f"envelope_exponent ({core.envelope_exponent})"
            )
        if abs(float(layer.basis_cutoff) - float(expected)) > 1e-4:
            raise ValueError("BesselRBF basis_cutoff " f"({layer.basis_cutoff}) must equal {expected}")
    bond_basis.reset_field_name(rbf="rbf_bg")
    core.atom_basis = atom_basis
    core.bond_basis = bond_basis


class BondGraphBlock(Block):
    def __init__(self, atom, bond, angle, last):
        super().__init__()
        self.atom = atom
        self.bond = bond
        self.angle = angle
        self.last = last

    def forward(self, state, ctx):
        atom_weight, directed = ctx.cache["atom_graph"]
        atom = self.atom(state.scalar, state.edge, atom_weight, ctx.receiver, ctx.sender, directed)
        bond, angle = state.edge, state.equi
        if self.bond is not None and angle.shape[0]:
            weight, center, first, second = ctx.cache["bond_graph"]
            bond = self.bond(atom, bond, weight, angle, center, first, second)
            if self.angle is not None:
                angle = self.angle(atom, bond, angle, center, first, second)
        return BlockState(
            scalar=atom, equi=angle, edge=bond, aux={"charge": state.scalar} if self.last else None
        )


class ChargeFeatureHead(Head):
    output_fields = ("atom_feature", "atom_feature_charge")

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, state, ctx, taps, inputs):
        node = state.scalar if self.norm is None else self.norm(state.scalar)
        return {"atom_feature": node, "atom_feature_charge": state.aux["charge"]}


@torch.compiler.disable
def independent_magnetic_readout(core):
    """Recognize the standard moment head whose output is independent of the force loss."""
    from ..layers.gradient import ForceLayer
    from ..layers.readout import AbsoluteLinearReadout, SimpleReadout
    from ..layers.reduce import EnergyReduceLayer
    from ..mp.nn.mlp import DenseLayer

    stages = core.post_sequence
    if tuple(type(layer) for layer in stages) != (
        AbsoluteLinearReadout, SimpleReadout, EnergyReduceLayer, ForceLayer
    ):
        return None
    moment, energy, _, _ = stages
    if (moment._output_fields != {"Ma"} or energy._output_fields != {"Ea"}
            or type(moment.head) is not DenseLayer or moment.head.activation_fn is not None):
        return None
    for layer in stages:
        if layer.test_only or layer.test_exclude or layer.eval_only or layer.train_only:
            return None
        if any(field != mapped for field, mapped in layer._name_mapping.items()):
            return None
    if any("forward" in vars(module) for module in stages.modules()):
        return None
    return moment


@torch.compiler.disable
def _native_magnetic_readout(moment, fields):
    # One linear layer per atom: dispatching a second compiled graph costs more
    # than running it natively. Its autograd still propagates magnetic supervision.
    return moment(fields)

@torch.compiler.disable(recursive=False)
def train_with_magnetic_readout(core, inputs, edges, topology, post_pairs):
    from ..layers.compiled_grad import forward_with_coordinate_gradients, has_module_hooks, native_forces
    from ..mp.graph import PreparedTopology

    if has_module_hooks(core.post_sequence):
        return native_forces(core, inputs)
    if isinstance(topology, PreparedTopology):
        bond_graph = topology.root.get(BondGraphRequest.force_cache_key)
        if bond_graph is not None and bond_graph[1].idx_ij.numel() == 0:
            # Without angles, auxiliary radial bases receive no energy gradient.
            # AOT can give these weights a zero gradient and trigger weight decay.
            return native_forces(core, inputs)
    moment = independent_magnetic_readout(core)
    if moment is None:
        return forward_with_coordinate_gradients(core, inputs, edges, topology, post_pairs)
    # AOTAutograd materializes zero gradients for unused outputs of a joint
    # graph. Keep magnetic parameters out of the force VJP and expose Ma so
    # a magnetic loss can still differentiate both the head and atom features.
    fields = forward_with_coordinate_gradients(core, inputs, edges, topology, post_pairs, skip_post=(0,))
    return _native_magnetic_readout(moment, fields)
