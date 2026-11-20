"""TensorNet Core (Simeon and De Fabritiis, NeurIPS 2023, arXiv:2306.06482).

Each atom carries a full Cartesian tensor ``[3, 3, F]``. The Core is the
embedding aggregation and the interaction stack only. It reads a species
embedding and a radial basis and emits:

* ``atom_feature`` — invariant norm embedding ``[N, F]`` (``feature_irreps = "Fx0e"``)
* ``atom_vector_feature`` — skew component as a Cartesian vector ``[N, 3, F]``
* ``atom_tensor_feature`` — full tensor ``[N, 3, 3, F]``

``ExpNormalSmearing``, ``RandomAtomEmbedding``, the cosine window, and the
energy MLP stay outside. TorchMD-Net's neighbor list includes a self pair
(``loop=True``). The shared Enerzyme neighbor list does not, so the Core
inserts those pairs with ``r = 0`` through the same radial basis.

The squared-Frobenius map ``[3 I², ||A||², ||S||²] → F`` (LayerNorm, linear,
SiLU) is the representation output in TorchMD-Net. The remaining energy MLP
``F → F/2 → 1`` is :class:`~enerzyme.models.layers.readout.SimpleReadout`
with ``head_type: two_layer`` and ``pyramidal: true``.

Adapted from the pure PyTorch path in TorchMD-Net (MIT):
https://github.com/torchmd/torchmd-net
Warp kernels are not used. Checkpoints from that package do not load.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList

from ..cutoff import cosine_transition
from ..layers import BaseFFCore
from .interaction import TensorNetEmbedding, TensorNetInteraction
from .ops import scalar_invariants, skew_tensor_to_vector, unit_vectors


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 4.5,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 2,
            "equivariance_group": "O(3)",
            "charge_scale": 0.1,
            "include_self": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


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


class TensorNetCore(BaseFFCore):
    """Cartesian rank-2 message-passing Core. Property heads stay in the layer stack."""

    def __str__(self) -> str:
        return """
################################################################################
# TensorNet (Simeon and De Fabritiis, NeurIPS 2023, arXiv:2306.06482)          #
# Cartesian 3x3 tensors; atom_feature [N, F], atom_tensor_feature [N, 3, 3, F] #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 32,
        cutoff_sr: float = 4.5,
        num_interactions: int = 2,
        equivariance_group: str = "O(3)",
        charge_scale: float = 0.1,
        include_self: bool = True,
    ) -> None:
        if int(num_interactions) < 0:
            raise ValueError(
                f"num_interactions must be >= 0, got {num_interactions}"
            )
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
                "cutoff_values_sr",
                "Q",
                "batch_seg",
            },
            output_fields={
                "atom_feature",
                "atom_vector_feature",
                "atom_tensor_feature",
            },
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_interactions = int(num_interactions)
        self.equivariance_group = normalize_equivariance_group(equivariance_group)
        self.charge_scale = float(charge_scale)
        self.include_self = bool(include_self)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.vector_feature_layout = "cartesian_n3f"
        self.tensor_feature_layout = "cartesian_n33f"
        # A list, not a submodule attribute: the basis already lives in pre_sequence.
        self._radial_holder: list = []

        self.embedding = TensorNetEmbedding(self.dim_embedding, self.num_rbf)
        self.interactions = ModuleList(
            [
                TensorNetInteraction(
                    self.dim_embedding, self.num_rbf, self.equivariance_group
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.out_norm = LayerNorm(3 * self.dim_embedding)
        self.out_linear = Linear(3 * self.dim_embedding, self.dim_embedding)

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        radial = None
        for layer in built_layers:
            if layer is self:
                break
            fields = getattr(layer, "_output_fields", ())
            if "rbf" in fields and callable(getattr(layer, "get_rbf", None)):
                radial = layer
        if radial is not None:
            radial_cutoff = getattr(radial, "cutoff_sr", None)
            if radial_cutoff is not None and abs(float(radial_cutoff) - self.cutoff_sr) > 1e-5:
                raise ValueError(
                    f"radial cutoff_sr ({radial_cutoff}) != Core cutoff_sr ({self.cutoff_sr})"
                )
            radial_width = getattr(radial, "num_rbf", None)
            if radial_width is not None and int(radial_width) != self.num_rbf:
                raise ValueError(
                    f"radial num_rbf ({radial_width}) != Core num_rbf ({self.num_rbf})"
                )
        self._radial_holder = [] if radial is None else [radial]

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
        if not self.include_self or n_atoms == 0:
            return idx_i, idx_j, vij, dist, rbf, window
        present = torch.zeros(n_atoms, dtype=torch.bool, device=idx_i.device)
        if idx_i.numel():
            self_edge = idx_i == idx_j
            if self_edge.any():
                present[idx_i[self_edge]] = True
        missing = (~present).nonzero(as_tuple=False).flatten()
        if missing.numel() == 0:
            return idx_i, idx_j, vij, dist, rbf, window
        if not self._radial_holder:
            raise RuntimeError(
                "TensorNet include_self needs the pre-core radial basis "
                "(a layer that writes rbf and implements get_rbf). "
                "Call build(), or pass a self pair for every atom."
            )
        zeros = dist.new_zeros(missing.numel())
        self_rbf = self._radial_holder[0].get_rbf(zeros).to(dtype=rbf.dtype)
        if self_rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"self-pair rbf width {self_rbf.shape[-1]} != num_rbf {self.num_rbf}"
            )
        self_window = window.new_ones(missing.numel())
        return (
            torch.cat((idx_i, missing)),
            torch.cat((idx_j, missing)),
            torch.cat((vij, vij.new_zeros(missing.numel(), 3))),
            torch.cat((dist, zeros)),
            torch.cat((rbf, self_rbf), dim=0),
            torch.cat((window, self_window)),
        )

    def _charge_scale(
        self,
        n_atoms: int,
        charge: Optional[Tensor],
        batch_seg: Optional[Tensor],
        like: Tensor,
    ) -> Tensor:
        if charge is None or self.charge_scale == 0.0 or n_atoms == 0:
            return like.new_ones(n_atoms)
        values = charge.to(device=like.device, dtype=like.dtype).reshape(-1)
        if values.numel() == 1:
            per_atom = values.expand(n_atoms)
        elif values.numel() == n_atoms:
            per_atom = values
        elif batch_seg is None:
            raise ValueError(
                f"Q has {values.numel()} values for {n_atoms} atoms; pass batch_seg "
                "when Q is the per-molecule total charge"
            )
        else:
            index = batch_seg.to(device=like.device).long()
            if index.numel() != n_atoms:
                raise ValueError(
                    f"batch_seg length {index.numel()} != number of atoms {n_atoms}"
                )
            if index.numel() and int(index.max()) >= values.numel():
                raise ValueError(
                    f"Q has {values.numel()} molecule values but batch_seg "
                    f"indexes {int(index.max())}"
                )
            per_atom = values[index]
        return 1.0 + self.charge_scale * per_atom

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run the tensor embedding and ``num_interactions`` updates.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``. ``Q`` is the molecular total charge. When it
        is omitted, or ``charge_scale`` is 0, the charge factor is 1 and the
        residual matches the paper's ``Y + Y²``.
        """
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
        hat = unit_vectors(vij, dist)
        features = self.embedding(
            atom_embedding, idx_i, idx_j, hat, basis, window
        )
        charge_scale = self._charge_scale(n_atoms, Q, batch_seg, atom_embedding)
        for interaction in self.interactions:
            features = interaction(
                features, idx_i, idx_j, basis, window, charge_scale
            )

        invariants, _iso, skew, _traceless = scalar_invariants(features)
        atom_feature = F.silu(self.out_linear(self.out_norm(invariants)))
        return {
            "atom_feature": atom_feature,
            "atom_vector_feature": skew_tensor_to_vector(skew),
            "atom_tensor_feature": features,
        }
