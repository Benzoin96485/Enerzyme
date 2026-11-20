"""AlphaNet Core (Yin et al., npj Comput. Mater. 2025, arXiv:2501.07155).

AlphaNet keeps the LEFTNet local frame and adds two scalar pathways:

* a complex phase, predicted from each message, that rotates the next
  layer's invariant features (RoPE);
* a matrix-product state contracted with the scalar features at every
  depth (the temporal connection).

The Core reads an atom embedding and a radial basis and emits

* ``atom_feature`` — invariant features ``[N, F]``, including the temporal state
* ``atom_vector_feature`` — Cartesian vectors ``[N, 3, F]``
* ``quantum_feature`` — real and imaginary parts of the normalized state,
  divided by ``main_chi1``, shape ``[N, 2 main_chi1]``

The species table, the Bessel basis, the energy head, species scale and
shift, and pair priors stay outside. The released ZBL term is computed in
that repository and then left out of the total energy; use
``ZBLRepulsionEnergy`` or ``ScreenedZBLRepulsionEnergy`` to add a wall.

Reference: https://github.com/zmyybc/AlphaNet (branch ``jax_and_zbl``,
``alphanet/models/alphanet.py``). Checkpoints from that repository do not
load. The edge frame follows the released code (unnormalized centroid
cross product), not the normalized frame written in the paper. The bond
axis is the unit ``vij``, and ``Dij_sr`` must be that length.
The default ``fte_scalarization`` is ``l2``, the Euclidean norm, so the
scalar state and the Cartesian vectors stay rotationally equivariant.
``l1`` is the released sum of absolute Cartesian components. It is not
equivariant and has to be requested explicitly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..cutoff import cosine_transition
from ..layers import BaseFFCore
from ..local_frame.encoding import LocalSubstructure, RadialFilter, aggregate_neighbors
from ..local_frame.geometry import (
    alphanet_edge_frame,
    require_matching_distance,
    require_same_graph,
)
from ..local_frame.message import (
    AlphaNetMessage,
    FrameTransition,
    contract_quantum,
    init_quantum,
)


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 96,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "equflash", "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 3,
            "head": 16,
            "main_chi1": 24,
            "mp_chi1": 24,
            "chi2": 6,
            "hidden_channels_chi": 96,
            "has_dropout_flag": True,
            "has_norm_before_flag": True,
            "has_norm_after_flag": False,
            "reduce_mode": "sum",
            "fte_scalarization": "l2",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense"},
    },
    {
        "name": "AtomicAffine",
        "params": {
            "scales": {"Ea": {"values": 1, "learnable": True}},
            "shifts": {"Ea": {"values": 0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class AlphaNetCore(BaseFFCore):
    """Local-frame Core with rotary phases and a temporal matrix-product state."""

    def __str__(self) -> str:
        return f"""
################################################################################
# AlphaNet (Yin et al., npj Comput. Mater. 2025, arXiv:2501.07155)            #
# atom_feature [N, F], atom_vector_feature [N, 3, F], quantum_feature         #
# layers={self.num_layers}  chi1={self.main_chi1}  heads={self.head}                              #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 96,
        cutoff_sr: float = 5.0,
        num_layers: int = 3,
        head: int = 16,
        main_chi1: int = 24,
        mp_chi1: int = 24,
        chi2: int = 6,
        hidden_channels_chi: int = 96,
        has_dropout_flag: bool = True,
        has_norm_before_flag: bool = True,
        has_norm_after_flag: bool = False,
        reduce_mode: str = "sum",
        fte_scalarization: str = "l2",
        eps: float = 1e-10,
    ) -> None:
        hidden = int(dim_embedding)
        layers = int(num_layers)
        head = int(head)
        main_chi1 = int(main_chi1)
        mp_chi1 = int(mp_chi1)
        chi2 = int(chi2)
        hidden_chi = int(hidden_channels_chi)
        if hidden < 4 or hidden % 2:
            raise ValueError(
                "dim_embedding must be even and >= 4 so the rotary embedding "
                f"can split it in half, got {dim_embedding}"
            )
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if float(cutoff_sr) <= 0.0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if head < 1 or hidden // head < 1:
            raise ValueError(
                f"head must be in 1..dim_embedding, got head={head}, dim_embedding={hidden}"
            )
        if hidden_chi < 2 or hidden_chi % head:
            raise ValueError(
                "hidden_channels_chi must be >= 2 and divisible by head, "
                f"got {hidden_chi} and head {head}"
            )
        if min(main_chi1, mp_chi1, chi2) < 1:
            raise ValueError(
                "main_chi1, mp_chi1, and chi2 must be >= 1, "
                f"got {(main_chi1, mp_chi1, chi2)}"
            )
        if reduce_mode not in {"sum", "mean"}:
            raise ValueError(f"reduce_mode must be 'sum' or 'mean', got {reduce_mode!r}")
        if fte_scalarization not in {"l1", "l2"}:
            raise ValueError(
                "fte_scalarization must be 'l1' or 'l2', got "
                f"{fte_scalarization!r}. The default 'l2' is rotationally "
                "equivariant. 'l1' matches the released AlphaNet transition "
                "and is not."
            )
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
                "Ra",
                "batch_seg",
                "cutoff_values_sr",
            },
            output_fields={"atom_feature", "atom_vector_feature", "quantum_feature"},
        )
        self.dim_embedding = hidden
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_layers = layers
        self.head = head
        self.main_chi1 = main_chi1
        self.mp_chi1 = mp_chi1
        self.chi2 = chi2
        self.hidden_channels_chi = hidden_chi
        self.has_dropout_flag = bool(has_dropout_flag)
        self.has_norm_before_flag = bool(has_norm_before_flag)
        self.has_norm_after_flag = bool(has_norm_after_flag)
        self.reduce_mode = str(reduce_mode)
        self.fte_scalarization = str(fte_scalarization)
        self.eps = float(eps)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.vector_feature_layout = "cartesian_n3f"

        self.embed_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.radial = RadialFilter(self.num_rbf, hidden)
        self.lse = LocalSubstructure(
            hidden, normalize=True, scale=True, endpoint="target_source"
        )
        self.messages = nn.ModuleList(
            [
                AlphaNetMessage(
                    hidden,
                    self.num_rbf,
                    hidden_chi,
                    head,
                    mp_chi1,
                    chi2,
                    dropout=self.has_dropout_flag,
                    norm_before=self.has_norm_before_flag,
                    norm_after=self.has_norm_after_flag,
                    reduce=self.reduce_mode,
                )
                for _ in range(layers)
            ]
        )
        self.transitions = nn.ModuleList(
            [FrameTransition(hidden, self.fte_scalarization) for _ in range(layers)]
        )
        self.kernel1 = nn.Parameter(torch.randn(hidden, main_chi1 * 2))
        self.kernels_real = nn.Parameter(torch.randn(layers, hidden, main_chi1, main_chi1))
        self.kernels_imag = nn.Parameter(torch.randn(layers, hidden, main_chi1, main_chi1))
        self.quantum_mix = nn.Linear(main_chi1 * 2, hidden)
        nn.init.xavier_uniform_(self.quantum_mix.weight)
        nn.init.zeros_(self.quantum_mix.bias)

    def build(self, built_layers: List[nn.Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}")
        if idx_i_sr.shape != idx_j_sr.shape or idx_i_sr.ndim != 1:
            raise ValueError("idx_i_sr and idx_j_sr must be 1D and the same length")
        if Ra.shape[0] != atom_embedding.shape[0] or Ra.shape[-1] != 3:
            raise ValueError("Ra must have shape [N, 3] with the same N as atom_embedding")
        require_same_graph(batch_seg, idx_i_sr, idx_j_sr)
        require_matching_distance(vij_sr, Dij_sr)

        window = _window(Dij_sr, self.cutoff_sr, cutoff_values_sr, rbf.dtype)
        species = self.embed_norm(atom_embedding)
        radial_hidden = self.radial(rbf, window)
        scalar = aggregate_neighbors(species, radial_hidden, idx_i_sr, idx_j_sr)
        vector = scalar.new_zeros(scalar.shape[0], 3, self.dim_embedding)
        quantum = init_quantum(species, self.kernel1)
        if idx_i_sr.numel():
            frame, edge_unit = alphanet_edge_frame(
                Ra, vij_sr, Dij_sr, idx_i_sr, idx_j_sr, self.eps
            )
            weight = self.lse(
                scalar,
                edge_unit,
                frame,
                radial_hidden,
                rbf,
                window,
                idx_i_sr,
                idx_j_sr,
            )
        else:
            weight = edge_unit = None

        rope = None
        for layer, (message, transition) in enumerate(zip(self.messages, self.transitions)):
            if weight is not None:
                rope, delta_s, delta_v = message(
                    scalar, vector, rbf, weight, edge_unit, idx_i_sr, idx_j_sr, rope
                )
                scalar = scalar + delta_s
                vector = vector + delta_v
            quantum = contract_quantum(
                scalar,
                quantum,
                self.kernels_real[layer],
                self.kernels_imag[layer],
                self.eps,
            )
            delta_s, delta_v = transition(scalar, vector)
            scalar = scalar + delta_s
            vector = vector + delta_v

        packed = torch.cat((quantum.real, quantum.imag), dim=-1)
        scalar = scalar + self.quantum_mix(packed) / self.main_chi1
        return {
            "atom_feature": scalar,
            "atom_vector_feature": vector,
            "quantum_feature": packed / self.main_chi1,
        }


def _window(
    dij: Tensor,
    cutoff: float,
    given: Optional[Tensor],
    dtype: torch.dtype,
) -> Tensor:
    if given is None:
        values = cosine_transition(dij.reshape(-1), cutoff)
    else:
        values = given.reshape(-1)
    return values.to(dtype=dtype)
