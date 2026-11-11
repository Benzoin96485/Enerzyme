"""FACTO Core (Liu et al., arXiv:2611.03276).

The Core is the map from an atomic graph to latent scalar features. It owns
the fixed elemental descriptor, the shared radial/angular geometry, and the
interaction blocks. The energy head (:class:`FACTOEnergyReadout`) and the
ZBL pair repulsion stay in the layer stack, outside the Core.

Published backbones are selected with ``preset``:

* ``S`` / ``M`` — no tensor transport, ``L = 4``
* ``L`` / ``XL`` — tensor transport on, ``XL`` uses ``L = 6``

All four use 5 blocks, a 6 Å cutoff, and a 1 Å short-range window. Only the
scalar state is updated in the last block. Custom aggregation kernels and
BF16×3 tensor-core emulation from the paper are not part of this port; the
math here is the FP32 reference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import BaseFFCore
from .descriptors import (
    DESCRIPTOR_DIM,
    elemental_descriptor_table,
)
from .geometry import FACTOGeometry
from .interaction import FACTOInteraction, RESIDUAL_SCALE

PAPER_PRESETS = {
    "S": {
        "scalar_dim": 384,
        "num_groups": 3,
        "tensor_channels": (128, 64, 32, 16),
        "num_radial": 32,
        "num_blocks": 5,
        "tensor_transport": False,
    },
    "M": {
        "scalar_dim": 768,
        "num_groups": 6,
        "tensor_channels": (128, 64, 32, 16),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": False,
    },
    "L": {
        "scalar_dim": 1024,
        "num_groups": 4,
        "tensor_channels": (256, 128, 64, 32),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": True,
    },
    "XL": {
        "scalar_dim": 2048,
        "num_groups": 4,
        "tensor_channels": (512, 256, 128, 64, 32, 16),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": True,
    },
}

_CUSTOM_PRESETS = {"custom", "none", ""}


def resolve_facto_config(
    preset: Optional[str],
    *,
    scalar_dim: Optional[int] = None,
    num_groups: Optional[int] = None,
    tensor_channels: Optional[Sequence[int]] = None,
    num_radial: Optional[int] = None,
    num_blocks: Optional[int] = None,
    tensor_transport: Optional[bool] = None,
) -> dict:
    """Merge a paper preset with explicit overrides. ``None`` keeps the preset."""
    key = "" if preset is None else str(preset).strip()
    if key.lower() in _CUSTOM_PRESETS:
        config = {
            "scalar_dim": 32,
            "num_groups": 4,
            "tensor_channels": (8, 4),
            "num_radial": 8,
            "num_blocks": 2,
            "tensor_transport": False,
        }
    else:
        name = key.upper()
        if name not in PAPER_PRESETS:
            known = ", ".join(("S", "M", "L", "XL", "custom"))
            raise ValueError(f"Unknown FACTO preset {preset!r}. Expected one of: {known}.")
        config = dict(PAPER_PRESETS[name])
    overrides = {
        "scalar_dim": scalar_dim,
        "num_groups": num_groups,
        "tensor_channels": tensor_channels,
        "num_radial": num_radial,
        "num_blocks": num_blocks,
        "tensor_transport": tensor_transport,
    }
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    config["tensor_channels"] = tuple(int(width) for width in config["tensor_channels"])
    return config


def _validate_config(config: dict, cutoff_sr: float, r_short: float) -> None:
    scalar_dim = int(config["scalar_dim"])
    num_groups = int(config["num_groups"])
    channels = config["tensor_channels"]
    num_blocks = int(config["num_blocks"])
    num_radial = int(config["num_radial"])
    if scalar_dim < 1 or scalar_dim % num_groups != 0:
        raise ValueError(
            f"scalar_dim={scalar_dim} must be positive and divisible by num_groups={num_groups}"
        )
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if num_radial < 2:
        raise ValueError(f"num_radial must be >= 2, got {num_radial}")
    if not channels or any(int(width) < 1 for width in channels):
        raise ValueError(f"tensor_channels must be positive, got {channels}")
    if not (0.0 < float(r_short) < float(cutoff_sr)):
        raise ValueError(
            f"r_short must lie in (0, cutoff_sr), got Rs={r_short}, Rc={cutoff_sr}"
        )


DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "facto",
    "Hartree_in_E": 1.0,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "Core", "params": {"preset": "S"}},
    {"name": "FACTOEnergyReadout"},
    {"name": "ZBLRepulsionEnergy"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class FACTOCore(BaseFFCore):
    """Atom-centered tensor interaction stack.

    Emits hierarchical post-block scalars ``atom_feature`` of shape
    ``(N, D, K)`` and the matching isolated-atom states
    ``isolated_atom_feature``. ``K`` does not include the initial embedding.
    Tensor channels depend on the degree, so there is no uniform
    ``atom_sphere_feature``.
    """

    def __str__(self) -> str:
        return """
################################################################################
# FACTO (Liu et al., arXiv:2611.03276)                                         #
# Atom-centered scalar + tensor MP; hierarchical atom_feature (N, D, K)        #
################################################################################
"""

    def __init__(
        self,
        max_Za: int = 94,
        cutoff_sr: float = 6.0,
        r_short: float = 1.0,
        preset: Optional[str] = "S",
        scalar_dim: Optional[int] = None,
        num_groups: Optional[int] = None,
        tensor_channels: Optional[Sequence[int]] = None,
        num_radial: Optional[int] = None,
        num_blocks: Optional[int] = None,
        tensor_transport: Optional[bool] = None,
        residual_scale: Optional[float] = None,
        rms_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            input_fields={"Za", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"},
            output_fields={"atom_feature", "isolated_atom_feature"},
        )
        config = resolve_facto_config(
            preset,
            scalar_dim=scalar_dim,
            num_groups=num_groups,
            tensor_channels=tensor_channels,
            num_radial=num_radial,
            num_blocks=num_blocks,
            tensor_transport=tensor_transport,
        )
        _validate_config(config, cutoff_sr, r_short)
        self.max_Za = int(max_Za)
        self.cutoff_sr = float(cutoff_sr)
        self.r_short = float(r_short)
        self.preset = None if preset is None else str(preset)
        self.scalar_dim = int(config["scalar_dim"])
        self.num_groups = int(config["num_groups"])
        self.tensor_channels: Tuple[int, ...] = tuple(config["tensor_channels"])
        self.num_radial = int(config["num_radial"])
        self.num_blocks = int(config["num_blocks"])
        self.tensor_transport = bool(config["tensor_transport"])
        self.residual_scale = (
            RESIDUAL_SCALE if residual_scale is None else float(residual_scale)
        )
        self.rms_epsilon = float(rms_epsilon)
        self.lmax = len(self.tensor_channels)
        self.num_output_blocks = self.num_blocks
        self.dim_feature_out = self.scalar_dim
        self.feature_irreps = f"{self.scalar_dim}x0e"
        self.emits_isolated_atom_feature = True

        self.register_buffer(
            "element_descriptors",
            elemental_descriptor_table(self.max_Za),
            persistent=False,
        )
        if self.element_descriptors.shape[-1] != DESCRIPTOR_DIM:
            raise RuntimeError("elemental descriptor width drifted from 31")
        self.embed = torch.nn.Linear(DESCRIPTOR_DIM, self.scalar_dim, bias=True)
        self.geometry = FACTOGeometry(
            scalar_dim=self.scalar_dim,
            n_radial=self.num_radial,
            channels=self.tensor_channels,
            cutoff=self.cutoff_sr,
            r_short=self.r_short,
        )
        self.blocks = ModuleList(
            [
                FACTOInteraction(
                    self.scalar_dim,
                    self.tensor_channels,
                    self.num_groups,
                    # The last block updates scalars only. Transport is still
                    # allocated on the first tensor block: its input is the
                    # zero initial state, so that application is a no-op, and
                    # later blocks carry the angular features it does not see.
                    update_tensors=block < self.num_blocks - 1,
                    tensor_transport=self.tensor_transport,
                    residual_scale=self.residual_scale,
                    rms_epsilon=self.rms_epsilon,
                )
                for block in range(self.num_blocks)
            ]
        )

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def initial_state(self, atomic_numbers: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """SiLU embedding and a zero tensor state, one row per atom."""
        descriptors = self.element_descriptors[atomic_numbers].to(
            dtype=self.embed.weight.dtype
        )
        scalars = F.silu(self.embed(descriptors))
        tensors = [
            scalars.new_zeros((atomic_numbers.shape[0], 2 * degree + 1, width))
            for degree, width in enumerate(self.tensor_channels, start=1)
        ]
        return scalars, tensors

    def _empty_geometry(self, like: Tensor):
        distance = like.new_zeros((0,))
        displacement = like.new_zeros((0, 3))
        return self.geometry(distance, displacement)

    def run_blocks(
        self,
        scalars: Tensor,
        tensors: List[Tensor],
        geometry,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        """Apply every block and stack the post-block scalar states on a new axis."""
        states = []
        for block in self.blocks:
            scalars, tensors = block(scalars, tensors, geometry, idx_i, idx_j)
            states.append(scalars)
        return torch.stack(states, dim=-1)

    def isolated_features(self, like: Tensor) -> Tensor:
        """Post-block states of one neighbor-free atom per atomic number.

        Shape ``(max_Za + 1, D, K)``. The row for Z is ``h_iso`` in the paper
        energy head. It depends on the parameters, not on coordinates.
        """
        atomic_numbers = torch.arange(
            self.max_Za + 1, device=like.device, dtype=torch.long
        )
        scalars, tensors = self.initial_state(atomic_numbers)
        idx_i = atomic_numbers.new_zeros((0,))
        idx_j = atomic_numbers.new_zeros((0,))
        return self.run_blocks(
            scalars, tensors, self._empty_geometry(scalars), idx_i, idx_j
        )

    def get_output(
        self,
        Za: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        scalars, tensors = self.initial_state(Za)
        geometry = self.geometry(Dij_sr, vij_sr)
        atom_feature = self.run_blocks(scalars, tensors, geometry, idx_i_sr, idx_j_sr)
        isolated = self.isolated_features(scalars)
        return {
            "atom_feature": atom_feature,
            "isolated_atom_feature": isolated[Za],
        }
