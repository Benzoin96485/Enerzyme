"""NewtonNet Core (Haghighatlari et al., Digital Discovery 2022, arXiv:2108.02913).

The Core consumes a scalar atom embedding and a radial basis and emits:

* ``atom_feature`` — invariant features ``[N, F]`` (``feature_irreps = "Fx0e"``)
* ``atom_vector_feature`` — Cartesian force features ``[N, 3, F]``
* ``latent_force`` — equivariant ``[N, 3]`` vector for the direction loss

Nuclear embedding, the Bessel expansion, the energy head, and ``Fa = -∇E``
stay outside the Core. The polynomial envelope is applied inside each
interaction, after that layer's radial linear map. The default order is 9,
matching both THGLab versions; ``envelope_p=7`` is the polynomial written in
the paper. ``variant="paper"`` is Eqs. 7–17. ``variant="repo"`` is
``InteractionNet`` in https://github.com/THGLab/NewtonNet (``main``). On
that path ``latent_force`` is the sum of ``atom_vector_feature`` over
channels. ``layer_norm`` normalizes the invariant features only, as both
THGLab versions do. The default energy head is the v1 network
``F → 128 → 64 → 1``. Current ``main`` uses ``F → F → F → 1``.

Reference checkpoints do not load.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn

from ..mp.basis.envelope import polynomial_envelope
from ..layers import BaseFFCore
from .interaction import NewtonNetPaperInteraction, NewtonNetRepoInteraction


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 20,
    "max_Za": 94,
    "cutoff_sr": 5.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "default", "trainable": False, "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 3,
            "variant": "paper",
            "envelope_p": 9,
            "layer_norm": False,
            "shared_interactions": False,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "hidden_dims": [128, 64],
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class NewtonNetCore(BaseFFCore):
    """Newtonian message-passing Core. Property heads stay in the layer stack."""

    def __str__(self) -> str:
        return f"""
################################################################################
# NewtonNet (Haghighatlari et al., Digital Discovery 2022, arXiv:2108.02913)  #
# variant={self.variant:<5}  atom_feature [N, F], atom_vector_feature [N, 3, F]   #
# latent_force [N, 3]                                                         #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 20,
        cutoff_sr: float = 5.0,
        num_interactions: int = 3,
        variant: str = "paper",
        envelope_p: int = 9,
        layer_norm: bool = False,
        shared_interactions: bool = False,
        displacement_init_gain: float = 1e-3,
    ) -> None:
        if int(num_interactions) < 1:
            raise ValueError(f"num_interactions must be >= 1, got {num_interactions}")
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if float(cutoff_sr) <= 0.0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if int(envelope_p) < 2:
            raise ValueError(
                f"envelope_p must be >= 2 so the force is continuous at the cutoff, got {envelope_p}"
            )
        variant_key = str(variant).lower()
        if variant_key not in {"paper", "repo"}:
            raise ValueError(f"variant must be 'paper' or 'repo', got {variant!r}")
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
            },
            output_fields={"atom_feature", "atom_vector_feature", "latent_force"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_interactions = int(num_interactions)
        self.variant = variant_key
        self.envelope_p = int(envelope_p)
        self.layer_norm = bool(layer_norm)
        self.shared_interactions = bool(shared_interactions)
        self.displacement_init_gain = float(displacement_init_gain)

        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.vector_feature_layout = "cartesian_n3f"

        n_blocks = 1 if self.shared_interactions else self.num_interactions
        if self.variant == "paper":
            self.blocks = nn.ModuleList(
                [
                    NewtonNetPaperInteraction(
                        self.dim_embedding,
                        self.num_rbf,
                        displacement_init_gain=self.displacement_init_gain,
                    )
                    for _ in range(n_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    NewtonNetRepoInteraction(self.dim_embedding, self.num_rbf)
                    for _ in range(n_blocks)
                ]
            )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.dim_embedding) for _ in range(self.num_interactions)]
            if self.layer_norm
            else []
        )

    def build(self, built_layers: List[nn.Module]) -> None:
        from ..layers.rbf import BesselRBFLayer

        for layer in built_layers:
            if isinstance(layer, BesselRBFLayer) and layer.apply_cutoff_fn:
                raise ValueError(
                    "NewtonNetCore multiplies the polynomial envelope onto the "
                    "radial basis. Set BesselRBF apply_cutoff_fn: false."
                )
        self.adopt_radius_stack(built_layers)

    def _envelope(self, dist: Tensor) -> Tensor:
        window = polynomial_envelope(dist, self.cutoff_sr, self.envelope_p)
        return window.reshape(-1, 1).to(dtype=dist.dtype)

    def _check_inputs(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
    ) -> None:
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must have shape [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        n_edges = idx_i_sr.shape[0]
        if idx_j_sr.shape[0] != n_edges or vij_sr.shape[0] != n_edges or Dij_sr.shape[0] != n_edges:
            raise ValueError(
                "idx_i_sr, idx_j_sr, vij_sr, and Dij_sr must cover the same edges, "
                f"got {n_edges}, {idx_j_sr.shape[0]}, {vij_sr.shape[0]}, {Dij_sr.shape[0]}"
            )
        if vij_sr.ndim != 2 or vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must have shape [E, 3], got {tuple(vij_sr.shape)}")

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        """Run ``num_interactions`` Newtonian updates.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``.
        """
        self._check_inputs(atom_embedding, rbf, idx_i_sr, idx_j_sr, vij_sr, Dij_sr)
        atom = atom_embedding
        n_atoms = atom.shape[0]
        idx_i = idx_i_sr.long()
        idx_j = idx_j_sr.long()
        vij = vij_sr.to(dtype=atom.dtype)
        dist = Dij_sr.to(dtype=atom.dtype).reshape(-1)
        rbf = rbf.to(dtype=atom.dtype)
        direction = vij / torch.linalg.vector_norm(vij, dim=-1, keepdim=True).clamp_min(1e-12)
        envelope = self._envelope(dist)
        force = atom.new_zeros(n_atoms, 3, self.dim_embedding)
        if self.variant == "paper":
            displacement = atom.new_zeros(n_atoms, 3, self.dim_embedding)
            latent = atom.new_zeros(n_atoms, 3)
            for step in range(self.num_interactions):
                block = self.blocks[0 if self.shared_interactions else step]
                atom, force, displacement, latent = block(
                    atom,
                    force,
                    displacement,
                    latent,
                    rbf,
                    envelope,
                    direction,
                    idx_i,
                    idx_j,
                    n_atoms,
                )
                if self.layer_norm:
                    atom = self.norms[step](atom)
        else:
            for step in range(self.num_interactions):
                block = self.blocks[0 if self.shared_interactions else step]
                atom, force = block(
                    atom, force, rbf, envelope, direction, idx_i, idx_j, n_atoms
                )
                if self.layer_norm:
                    atom = self.norms[step](atom)
            latent = force.sum(dim=-1)
        return {
            "atom_feature": atom,
            "atom_vector_feature": force,
            "latent_force": latent,
        }
