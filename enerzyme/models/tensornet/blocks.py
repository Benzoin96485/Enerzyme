"""Cartesian rank-two embedding, tensor update and charge equilibration for TensorNet."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..mp.ops.self_pairs import insert_missing_self_pairs, unit_direction
from ..mp.basis.envelope import cosine_transition
from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.head import Head
from .interaction import TensorNetEmbedding
from ..mp.equivariant.cartesian import scalar_invariants, skew_tensor_to_vector


def normalize_equivariance_group(value: str) -> str:
    """Return ``O(3)`` or ``SO(3)``."""
    key = str(value).upper().replace(" ", "")
    if key in {"O(3)", "O3"}:
        return "O(3)"
    if key in {"SO(3)", "SO3"}:
        return "SO(3)"
    raise ValueError(
        f"equivariance_group must be 'O(3)' or 'SO(3)', got {value!r}"
    )


class TensorNetEmbed(Embed):
    input_fields = ("atom_embedding",)

    def __init__(self, dim_embedding, num_rbf, cutoff_sr, include_self, charge_scale):
        super().__init__()
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.include_self = bool(include_self)
        self.charge_scale = float(charge_scale)
        self.embedding = TensorNetEmbedding(self.dim_embedding, self.num_rbf)
        self.charge_head = None
        # The radial layer is already registered on the surrounding pre_sequence.
        self._radial_holder = []

    def _window(self, dist: Tensor, cutoff_values_sr: Optional[Tensor]) -> Tensor:
        if cutoff_values_sr is None:
            return cosine_transition(dist, self.cutoff_sr).reshape(-1).to(dtype=dist.dtype)
        window = cutoff_values_sr.reshape(-1).to(dtype=dist.dtype)
        if window.shape[0] != dist.shape[0]:
            raise ValueError(
                "cutoff_values_sr and Dij_sr must describe the same edges, "
                f"got {window.shape[0]} and {dist.shape[0]}"
            )
        return window

    def _augment_self(
        self,
        n_atoms: int,
        idx_i: Tensor,
        idx_j: Tensor,
        vij: Tensor,
        dist: Tensor,
        rbf: Tensor,
        window: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        radial = self._radial_holder[0] if self._radial_holder else None
        return insert_missing_self_pairs(
            n_atoms,
            idx_i,
            idx_j,
            vij,
            dist,
            rbf,
            window,
            enabled=self.include_self,
            radial_basis=radial,
            num_rbf=self.num_rbf,
        )

    def resolve_total_charge(
        self,
        n_atoms: int,
        charge: Optional[Tensor],
        batch_seg: Optional[Tensor],
        like: Tensor,
    ) -> Tensor:
        """Molecular total charge written on every atom. Missing ``Q`` is 0.

        A length-``N`` ``Q`` is that total already copied onto each atom. Values
        inside one molecule have to match; pass one value per molecule with
        ``batch_seg`` when the molecules differ.
        """
        if n_atoms == 0:
            return like.new_zeros(0)
        if charge is None:
            return like.new_zeros(n_atoms)
        values = charge.to(device=like.device, dtype=like.dtype).reshape(-1)
        if values.numel() == 1:
            return values.expand(n_atoms)
        if values.numel() == n_atoms:
            from ..layers.electrostatics import require_uniform_molecular_charge

            require_uniform_molecular_charge(values, batch_seg)
            return values
        if batch_seg is None:
            raise ValueError(
                f"Q has {values.numel()} values for {n_atoms} atoms; pass batch_seg "
                "when Q is the per-molecule total charge"
            )
        index = batch_seg.to(device=like.device).long()
        if index.numel() != n_atoms:
            raise ValueError(
                f"batch_seg length {index.numel()} != number of atoms {n_atoms}"
            )
        if torch.compiler.is_compiling():
            torch._assert_async((index < values.numel()).all(),
                                "batch_seg indexes outside the supplied molecular charges")
        elif index.numel() and int(index.max()) >= values.numel():
            raise ValueError(
                f"Q has {values.numel()} molecule values but batch_seg "
                f"indexes {int(index.max())}"
            )
        return values[index]

    def _charge_scale(
        self,
        n_atoms: int,
        charge: Optional[Tensor],
        batch_seg: Optional[Tensor],
        like: Tensor,
    ) -> Tensor:
        if charge is None or self.charge_scale == 0.0 or n_atoms == 0:
            return like.new_ones(n_atoms)
        return 1.0 + self.charge_scale * self.resolve_total_charge(
            n_atoms, charge, batch_seg, like
        )

    def _prepare_graph(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor],
    ) -> tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Validate the short-range graph and insert missing self pairs."""
        n_atoms = int(atom_embedding.shape[0])
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must have shape [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must have shape [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        if idx_i_sr.shape != idx_j_sr.shape or idx_i_sr.ndim != 1:
            raise ValueError(
                "idx_i_sr and idx_j_sr must be 1D edge indices of the same length, "
                f"got {tuple(idx_i_sr.shape)} and {tuple(idx_j_sr.shape)}"
            )
        n_edges = int(idx_i_sr.shape[0])
        if tuple(vij_sr.shape) != (n_edges, 3):
            raise ValueError(
                f"vij_sr must have shape [{n_edges}, 3], got {tuple(vij_sr.shape)}"
            )
        if Dij_sr.reshape(-1).shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {Dij_sr.reshape(-1).shape[0]} and {n_edges}"
            )
        if rbf.shape[0] != n_edges:
            raise ValueError(
                f"rbf has {rbf.shape[0]} rows for {n_edges} edges"
            )

        dtype = atom_embedding.dtype
        dist = Dij_sr.reshape(-1).to(dtype=dtype)
        vij = vij_sr.to(dtype=dtype)
        basis = rbf.to(dtype=dtype)
        window = self._window(dist, cutoff_values_sr)
        idx_i, idx_j, vij, dist, basis, window = self._augment_self(
            n_atoms, idx_i_sr.long(), idx_j_sr.long(), vij, dist, basis, window
        )
        return n_atoms, idx_i, idx_j, unit_direction(vij, dist), basis, window

    def forward(self, inputs, ctx):
        embedding = inputs["atom_embedding"]
        n, receiver, sender, unit, basis, window = self._prepare_graph(
            embedding, ctx.rbf, ctx.receiver, ctx.sender, ctx.vec, ctx.dist, ctx.cutoff_values,
        )
        features = self.embedding(embedding, receiver, sender, unit, basis, window)
        edges = (receiver, sender, basis, window)
        if self.charge_head is None:
            scale = self._charge_scale(n, inputs.get("Q"), ctx.batch, embedding)
            return BlockState(equi=features, edge=edges, aux={"charge_scale": scale})
        total = self.resolve_total_charge(n, inputs.get("Q"), ctx.batch, embedding)
        charges = self.charge_head(features, total, ctx.batch)
        return BlockState(scalar=charges, equi=features, edge=edges, aux={"total_charge": total})


class TensorNetBlock(Block):
    def __init__(self, interaction, charge_head=None):
        super().__init__()
        self.interaction = interaction
        self.charge_head = charge_head

    def forward(self, state, ctx):
        if self.charge_head is None:
            features = self.interaction(state.equi, *state.edge, state.aux["charge_scale"])
            return state._replace(equi=features)
        features = self.interaction(state.equi, state.scalar, *state.edge)
        charges = self.charge_head(features, state.aux["total_charge"], ctx.batch)
        return state._replace(scalar=charges, equi=features)


class TensorNetHead(Head):
    output_fields = ("atom_feature", "atom_vector_feature", "atom_tensor_feature")

    def __init__(self, width, charged):
        super().__init__()
        self.out_norm = nn.LayerNorm(3 * width)
        self.out_linear = nn.Linear(3 * width, width)
        self.needs_taps = charged
        if charged:
            self.output_fields = (*self.output_fields, "charge_channels")

    def forward(self, state, ctx, taps, inputs):
        invariants, _, skew, _ = scalar_invariants(state.equi)
        result = {
            "atom_feature": F.silu(self.out_linear(self.out_norm(invariants))),
            "atom_vector_feature": skew_tensor_to_vector(skew),
            "atom_tensor_feature": state.equi,
        }
        if self.needs_taps:
            result["charge_channels"] = torch.cat([stage.scalar for stage in taps], dim=-1)
        return result


def build_tensornet(core, built_layers):
    core.adopt_radius_stack(built_layers)
    radial = None
    for layer in built_layers:
        if layer is core:
            break
        fields = getattr(layer, "_output_fields", ())
        if "rbf" in fields and callable(getattr(layer, "get_rbf", None)):
            radial = layer
    if radial is not None:
        radial_cutoff = getattr(radial, "cutoff_sr", None)
        if radial_cutoff is not None and abs(float(radial_cutoff) - core.cutoff_sr) > 1e-5:
            raise ValueError(
                f"radial cutoff_sr ({radial_cutoff}) != Core cutoff_sr ({core.cutoff_sr})"
            )
        radial_width = getattr(radial, "num_rbf", None)
        if radial_width is not None and int(radial_width) != core.num_rbf:
            raise ValueError(
                f"radial num_rbf ({radial_width}) != Core num_rbf ({core.num_rbf})"
            )
    core._radial_holder = [] if radial is None else [radial]


def build_tensornet2(core, built_layers):
    from ..layers.electrostatics import WeightedCoulombEnergyLayer, expand_coulomb_weights

    build_tensornet(core, built_layers)
    after_core = False
    for layer in built_layers:
        if layer is core:
            after_core = True
            continue
        if not after_core or not isinstance(layer, WeightedCoulombEnergyLayer):
            continue
        try:
            expand_coulomb_weights(
                layer.layer_weights, core.charge_channels_dim, torch.zeros(1)
            )
        except ValueError as exc:
            raise ValueError(
                f"{exc} TensorNet2 emits {core.charge_channels_dim} charge "
                f"channels ({core.num_charge_blocks} blocks x q_dim {core.q_dim}). "
                "Set WeightedCoulombEnergy layer_weights to one weight per "
                "block, one weight per channel, or omit them."
            ) from exc
