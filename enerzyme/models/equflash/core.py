"""EquFlash Core (SamsungDS/GGNN).

E(3)-equivariant message passing built from Clebsch–Gordan tensor products.
The Core reads a scalar atom embedding and a radial basis and emits flat
irreps ``atom_feature``. Species embedding, the Bessel expansion, the XPLOR
window, and energy / force heads stay in the layer stack.

Reference: https://github.com/SamsungDS/GGNN (CC BY-NC-SA 4.0). This module
reimplements the interaction with Enerzyme's e3nn helpers. It does not vendor
that repository. The released checkpoints also offer cuequivariance and
FlashTP kernels; the numerics here are the e3nn uvu product.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList

from ..e3nn_nn import scalar_0e_dim
from ..layers import BaseFFCore
from .interaction import EquFlashInteraction
from .irreps import equflash_irreps, spherical_harmonics_irreps


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# Published EquFlash training config (GGNN ``configs/equflash/equflash.yml``):
# 256 channels, lmax 3, 10 convolutions, cutoff 6 Å with XPLOR from 5.5 Å,
# 8 Bessel functions. ``conv_denominator`` there is a dataset neighbor count
# (~72.5 on OMat). The default here is 1 so a new fit is not tied to that set.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "xplor",
    "cuton": 5.5,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "equflash", "trainable": True, "apply_cutoff_fn": True},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "lmax": 3,
            "num_layers": 10,
            "parity": True,
            "radial_hidden": [64, 64],
            "conv_denominator": 1.0,
            "train_denominator": False,
            "use_bias": False,
            "normalize_sph": True,
            "feed_forward": False,
            "act_radial": "silu",
            "act_scalar_even": "silu",
            "act_scalar_odd": "tanh",
            "act_gate_even": "silu",
            "act_gate_odd": "tanh",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class EquFlashCore(BaseFFCore):
    """Equivariant convolution stack. Property heads stay outside the Core."""

    def __str__(self) -> str:
        return """
################################################################################
# EquFlash (SamsungDS/GGNN)                                                    #
# uvu Clebsch–Gordan convolution + equivariant gate; atom_feature irreps      #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 256,
        num_rbf: int = 8,
        lmax: int = 3,
        num_layers: int = 10,
        parity: bool = True,
        radial_hidden: Optional[Sequence[int]] = None,
        conv_denominator: float = 1.0,
        train_denominator: bool = False,
        use_bias: bool = False,
        normalize_sph: bool = True,
        feed_forward: bool = False,
        act_radial: str = "silu",
        act_scalar_even: str = "silu",
        act_scalar_odd: str = "tanh",
        act_gate_even: str = "silu",
        act_gate_odd: str = "tanh",
        irreps_manual: Optional[Sequence[str]] = None,
        irreps_schedule: Literal["natural", "closure"] = "natural",
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
            },
            output_fields={"atom_feature"},
        )
        if radial_hidden is None:
            radial_hidden = [64, 64]
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.lmax = int(lmax)
        self.parity = bool(parity)
        self.normalize_sph = bool(normalize_sph)

        schedule = equflash_irreps(
            self.dim_embedding,
            self.lmax,
            int(num_layers) if irreps_manual is None else max(len(irreps_manual) - 1, 1),
            self.parity,
            irreps_manual,
            schedule=irreps_schedule,
        )
        if irreps_manual is not None:
            num_layers = len(schedule) - 1
        self.num_layers = int(num_layers)
        scalar_in = o3.Irreps(f"{self.dim_embedding}x0e")
        if schedule[0] != scalar_in:
            raise ValueError(
                "EquFlash starts from the scalar atom embedding "
                f"({scalar_in}), got input irreps {schedule[0]}"
            )

        self.irreps_sh = spherical_harmonics_irreps(self.lmax, self.parity)
        sh_in = "1o" if self.parity else "1e"
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=self.normalize_sph,
            normalization="component",
            irreps_in=sh_in,
        )
        blocks = []
        irreps_in = schedule[0]
        recorded = [str(irreps_in)]
        for layer in range(self.num_layers):
            block = EquFlashInteraction(
                irreps_in=irreps_in,
                irreps_sh=self.irreps_sh,
                irreps_out=schedule[layer + 1],
                radial_hidden=radial_hidden,
                num_rbf=self.num_rbf,
                act_radial=act_radial,
                scalar_even=act_scalar_even,
                scalar_odd=act_scalar_odd,
                gate_even=act_gate_even,
                gate_odd=act_gate_odd,
                denominator=conv_denominator,
                train_denominator=train_denominator,
                bias=use_bias,
                feed_forward=feed_forward,
            )
            blocks.append(block)
            irreps_in = block.irreps_out
            recorded.append(str(irreps_in))
        self.interactions = ModuleList(blocks)
        # Gate output can reorder relative to the requested schedule.
        self.feature_irreps = recorded[-1]
        self.dim_feature_out = scalar_0e_dim(self.feature_irreps)
        self.irreps_schedule = recorded

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        """Run the convolution stack.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``. ``rbf`` is the radial basis already multiplied
        by the cutoff window when ``BesselRBF`` is configured that way.
        """
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}"
            )
        features = atom_embedding
        idx_receiver = idx_i_sr.long()
        idx_sender = idx_j_sr.long()
        vectors = vij_sr.to(dtype=features.dtype)
        radial = rbf.to(dtype=features.dtype)
        sph = self.spherical_harmonics(vectors)
        for block in self.interactions:
            features = block(features, sph, radial, idx_sender, idx_receiver)
        return {"atom_feature": features}
