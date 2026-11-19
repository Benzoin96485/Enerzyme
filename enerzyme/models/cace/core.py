"""CACE Core (Cheng, npj Comput. Mater. 2024).

The Core maps a chemical embedding and a short-range graph to invariant
atomic features. One evaluation builds the Cartesian density inside the
cutoff, mixes radial channels per angular shell, and contracts that density
into polynomially independent invariants. Further steps pass messages and
repeat the contraction. The embedding table, the Bessel basis, the energy
head, and elemental references stay in the layer stack.

``idx_i_sr`` is the receiver (the atom that owns the density). ``idx_j_sr``
is the sender. ``vij_sr`` is ``r_sender - r_receiver``.

This is an original implementation of the published Cartesian contraction.
The reference package is CC BY-NC 4.0 and is not imported; its checkpoints
are not weight-compatible. The preset table follows the public cacefit
training scripts.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ..cutoff import polynomial_envelope
from ..layers._base_layer import BaseFFCore
from .basis import CartesianAngularBasis, InvariantSymmetrizer
from .interaction import MessagePassing, RadialChannelMix, parse_messages, scatter_sum
from .presets import resolve_cace_preset


DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 3,
    "num_rbf": 6,
    "cutoff_sr": 5.5,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "default",
            "trainable": True,
            "apply_cutoff_fn": False,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "water"}},
    {"name": "CACEEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class CACECore(BaseFFCore):
    """Cartesian atomic cluster expansion on an external embedding.

    Emits ``atom_feature`` of shape ``[N, (T + 1) * n_radial * n_invariants * channels]``
    with ``feature_irreps = "{F}x0e"``. ``T`` is the number of message-passing
    steps. Each block is flattened as radial, then invariant slot, then edge
    channel, and the initial density comes first.
    """

    def __str__(self) -> str:
        return """
################################################################################
# CACE (Cheng, 2024)                                                           #
# Cartesian cluster invariants from a chemical embedding; head stays outside  #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 3,
        cutoff_sr: float = 5.5,
        num_rbf: int = 6,
        preset: Optional[str] = "water",
        lmax: Optional[int] = None,
        max_nu: Optional[int] = None,
        n_radial: Optional[int] = None,
        num_message_passing: Optional[int] = None,
        messages: Optional[str] = None,
        envelope_p: Optional[int] = None,
        avg_num_neighbors: Optional[float] = None,
        readout_hidden: Optional[tuple] = None,
        shared_channels: Optional[bool] = None,
        shared_l: Optional[bool] = None,
        separate_receiver: Optional[bool] = None,
        bchi_layers: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "cutoff_values_sr",
            },
            output_fields={"atom_feature"},
        )
        spec = resolve_cace_preset(
            preset,
            lmax=lmax,
            max_nu=max_nu,
            n_radial=n_radial,
            num_message_passing=num_message_passing,
            messages=messages,
            envelope_p=envelope_p,
            avg_num_neighbors=avg_num_neighbors,
            readout_hidden=readout_hidden,
            shared_channels=shared_channels,
            shared_l=shared_l,
            separate_receiver=separate_receiver,
            bchi_layers=bchi_layers,
        )
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(spec["lmax"]) < 0:
            raise ValueError(f"lmax must be >= 0, got {spec['lmax']}")
        if int(spec["n_radial"]) < 1:
            raise ValueError(f"n_radial must be >= 1, got {spec['n_radial']}")
        if int(spec["num_message_passing"]) < 0:
            raise ValueError(
                f"num_message_passing must be >= 0, got {spec['num_message_passing']}"
            )
        if float(spec["avg_num_neighbors"]) <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {spec['avg_num_neighbors']}"
            )
        if int(spec["envelope_p"]) < 1:
            raise ValueError(f"envelope_p must be >= 1, got {spec['envelope_p']}")
        self.preset = None if preset is None else str(preset)
        self.is_cace_core = True
        self.dim_embedding = int(dim_embedding)
        self.cutoff_sr = float(cutoff_sr)
        self.num_rbf = int(num_rbf)
        self.lmax = int(spec["lmax"])
        self.max_nu = int(spec["max_nu"])
        self.n_radial = int(spec["n_radial"])
        self.num_message_passing = int(spec["num_message_passing"])
        self.messages = parse_messages(spec["messages"])
        self.envelope_p = int(spec["envelope_p"])
        self.avg_num_neighbors = float(spec["avg_num_neighbors"])
        self.readout_hidden = tuple(spec["readout_hidden"])
        self.shared_channels = bool(spec["shared_channels"])
        self.shared_l = bool(spec["shared_l"])
        self.separate_receiver = bool(spec["separate_receiver"])
        self.bchi_layers = int(spec["bchi_layers"])
        self.eps = 1e-12
        self.n_channels = self.dim_embedding * self.dim_embedding
        self.norm = self.avg_num_neighbors ** -0.5

        self.angular = CartesianAngularBasis(self.lmax)
        self.n_angular = self.angular.n_angular
        self.symmetrizer = InvariantSymmetrizer(self.lmax, self.max_nu)
        self.n_invariants = self.symmetrizer.n_invariants
        self.mix = RadialChannelMix(
            self.num_rbf,
            self.n_radial,
            self.n_channels,
            self.angular.shells,
        )
        self.receiver_proj = (
            nn.Linear(self.dim_embedding, self.dim_embedding, bias=False)
            if self.separate_receiver
            else None
        )
        if self.receiver_proj is not None:
            nn.init.eye_(self.receiver_proj.weight)
        self.layers = nn.ModuleList()
        for _ in range(self.num_message_passing):
            self.layers.append(
                MessagePassing(
                    self.messages,
                    n_raw=self.num_rbf,
                    n_radial=self.n_radial,
                    n_angular=self.n_angular,
                    n_channels=self.n_channels,
                    n_invariants=self.n_invariants,
                    lmax=self.lmax,
                    shells=self.angular.shells,
                    l_index=self.angular.l_index,
                    cutoff=self.cutoff_sr,
                    norm=self.norm,
                    shared_channels=self.shared_channels,
                    shared_l=self.shared_l,
                    bchi_layers=self.bchi_layers,
                )
            )
        self.dim_feature_out = (
            (self.num_message_passing + 1)
            * self.n_radial
            * self.n_invariants
            * self.n_channels
        )
        self.feature_irreps = f"{self.dim_feature_out}x0e"

    def build(self, built_layers) -> None:
        self.adopt_radius_stack(built_layers)

    def _edge_basis(
        self,
        embedding: Tensor,
        receiver: Tensor,
        rbf: Tensor,
        distance: Tensor,
        vectors: Tensor,
        receivers: Tensor,
        senders: Tensor,
        window: Tensor,
    ) -> Tensor:
        radial = rbf * window.unsqueeze(-1)
        direction = vectors / distance.clamp_min(self.eps).unsqueeze(-1)
        monomials = self.angular(direction)
        edge_type = (
            receiver.index_select(0, receivers).unsqueeze(-1)
            * embedding.index_select(0, senders).unsqueeze(-2)
        ).reshape(receivers.shape[0], self.n_channels)
        return (
            radial[:, :, None, None]
            * monomials[:, None, :, None]
            * edge_type[:, None, None, :]
        )

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must be [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must be [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        n_atoms = atom_embedding.shape[0]
        n_edges = int(idx_i_sr.shape[0])
        if (
            idx_j_sr.shape[0] != n_edges
            or Dij_sr.shape[0] != n_edges
            or vij_sr.shape[0] != n_edges
        ):
            raise ValueError(
                "Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {Dij_sr.shape[0]}, {vij_sr.shape[0]}, {n_edges}, {idx_j_sr.shape[0]}"
            )
        if n_edges and vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must be [E, 3], got {tuple(vij_sr.shape)}")
        receiver = (
            self.receiver_proj(atom_embedding)
            if self.receiver_proj is not None
            else atom_embedding
        )
        if n_edges == 0:
            mixed = atom_embedding.new_zeros(
                n_atoms, self.n_radial, self.n_angular, self.n_channels
            )
            edge = atom_embedding.new_zeros(
                0, self.num_rbf, self.n_angular, self.n_channels
            )
            window = atom_embedding.new_zeros(0)
        else:
            if cutoff_values_sr is None:
                window = polynomial_envelope(Dij_sr, self.cutoff_sr, self.envelope_p)
            else:
                window = cutoff_values_sr.reshape(-1).to(dtype=Dij_sr.dtype)
                if window.shape[0] != n_edges:
                    raise ValueError(
                        f"cutoff_values_sr has {window.shape[0]} values for {n_edges} edges"
                    )
            edge = self._edge_basis(
                atom_embedding,
                receiver,
                rbf,
                Dij_sr,
                vij_sr,
                idx_i_sr,
                idx_j_sr,
                window,
            )
            mixed = self.mix(scatter_sum(edge, idx_i_sr, n_atoms))
        blocks = []
        updates = list(self.layers)
        updates.append(None)
        for layer in updates:
            invariant = self.symmetrizer(mixed)
            blocks.append(invariant.reshape(n_atoms, -1))
            if layer is None:
                break
            mixed = layer(
                mixed,
                invariant,
                edge,
                Dij_sr,
                window,
                idx_i_sr,
                idx_j_sr,
                n_atoms,
                self.mix,
            )
        return {"atom_feature": torch.cat(blocks, dim=-1)}
