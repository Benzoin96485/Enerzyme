"""DPA4C Core: one local aggregation and a polynomial invariant descriptor.

Li et al., arXiv:2608.19041. The Core reads an atom embedding and the
short-range graph and returns the calibrated invariant vector. The residual
energy MLP, the elemental reference, and any long-range prior stay outside.
DeePMD-kit is LGPL-3.0 and is not imported; its checkpoints do not load.
Compressed CUDA evaluation is a deployment path of that package, not of this
Core. Further message-passing blocks, native spin, and charge conditioning
are outside this port.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..cutoff import C3CutoffEnvelope
from ..functional import reduce_edge
from ..layers import BaseAtomEmbedding, BaseFFCore, RangeSeparationLayer
from ..so3.radial import BesselC3RadialBasis
from .geometry import derive_degree_channels, invariant_width, require_profile
from .geometry import cartesian_harmonics
from .radial import OrderedPairFiLM, RadialAmplitude
from .readout import InvariantReadout


ENVELOPE_EXPONENT = 5
DISTANCE_EPS = 1.0e-7
NORM_FLOOR = 0.25
# A geometric column this small on the calibration set has no magnitude to
# match to the embedding. Leave its scale at 1 instead of dividing by it.
CALIBRATION_RMS_FLOOR = 1.0e-12

# Grades in Li et al. and the DeePMD-kit descriptor docs. ``fitting_hidden``
# is the width of the external residual MLP, three layers in the released nets.
PUBLISHED_GRADES = {
    "nano": {"channels": 8, "lmax": 2, "radial_modes": 0, "fitting_hidden": 96, "fitting_layers": 3},
    "mini": {"channels": 32, "lmax": 2, "radial_modes": 0, "fitting_hidden": 192, "fitting_layers": 3},
    "neo": {"channels": 32, "lmax": 2, "radial_modes": 4, "fitting_hidden": 192, "fitting_layers": 3},
    "air": {"channels": 64, "lmax": 3, "radial_modes": 4, "fitting_hidden": 256, "fitting_layers": 3},
    "plus": {"channels": 128, "lmax": 3, "radial_modes": 4, "fitting_hidden": 384, "fitting_layers": 3},
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "channels": 8,
            "lmax": 2,
            "radial_modes": 0,
            "n_radial": 16,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "hidden_dims": [96, 96, 96],
            "identity_residual": True,
            "use_bias_out": True,
            "activation_fn": "swish",
            "activation_params": {"initial_beta": 1.0, "learnable": False},
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


class DPA4CCore(BaseFFCore):
    """Single-hop degree-wise descriptor. ``atom_feature`` is the invariant vector."""

    def __init__(
        self,
        dim_embedding: int,
        channels: Optional[int] = None,
        lmax: int = 2,
        radial_modes: int = 0,
        n_radial: int = 16,
        basis_type: str = "bessel",
        cutoff_sr: float = 6.0,
    ) -> None:
        if channels is None:
            channels = int(dim_embedding)
        channels, lmax = require_profile(int(channels), int(lmax))
        if int(dim_embedding) != channels:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must equal channels ({channels}); "
                "the atom embedding is the initial scalar feature"
            )
        radial_modes = _nonnegative_int("radial_modes", radial_modes)
        if isinstance(n_radial, bool) or not isinstance(n_radial, int) or n_radial < 1:
            raise ValueError(f"n_radial must be an integer >= 1, got {n_radial}")
        if float(cutoff_sr) <= 0.0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        basis_type = str(basis_type).lower()
        if basis_type not in ("bessel", "gaussian"):
            raise ValueError(f"basis_type must be 'bessel' or 'gaussian', got {basis_type!r}")

        width = invariant_width(channels, lmax)
        super().__init__(
            input_fields={"atom_embedding", "idx_i_sr", "idx_j_sr", "vij_sr"},
            output_fields={"atom_feature"},
        )
        self.dim_embedding = channels
        self.channels = channels
        self.lmax = lmax
        self.radial_modes = radial_modes
        self.n_radial = int(n_radial)
        self.cutoff_sr = float(cutoff_sr)
        self.degree_channels = derive_degree_channels(channels, lmax)
        self.dim_feature_out = width
        self.feature_irreps = f"{width}x0e"
        self.radial_basis = BesselC3RadialBasis(
            rcut=self.cutoff_sr,
            n_radial=self.n_radial,
            basis_type=basis_type,
            exponent=ENVELOPE_EXPONENT,
            apply_envelope=False,
        )
        self.envelope = C3CutoffEnvelope(rcut=self.cutoff_sr, exponent=ENVELOPE_EXPONENT)
        self.radial = RadialAmplitude(self.n_radial, channels, radial_modes)
        self.pair_film = OrderedPairFiLM(channels, radial_modes)
        self.readout = InvariantReadout(channels, lmax)
        if self.readout.dim_out + 2 + channels != width:
            raise RuntimeError(
                f"descriptor width {width} does not match readout {self.readout.dim_out}"
            )
        self.register_buffer("calib_mean", torch.zeros(width))
        self.register_buffer("calib_scale", torch.ones(width))

    def __str__(self) -> str:
        return f"""
###############################################################################
# DPA4C (arXiv:2608.19041) — one aggregation, polynomial invariants          #
# channels={self.channels}, lmax={self.lmax}, radial_modes={self.radial_modes}, descriptor={self.dim_feature_out} #
# Atom embedding and the energy MLP stay outside the Core.                  #
###############################################################################
"""

    def build(self, built_layers: List[nn.Module]) -> None:
        self.adopt_radius_stack(built_layers)
        for layer in built_layers:
            if isinstance(layer, BaseAtomEmbedding) and int(layer.dim_embedding) != self.channels:
                raise ValueError(
                    f"atom embedding width {layer.dim_embedding} must equal "
                    f"DPA4C channels {self.channels}"
                )
            if isinstance(layer, RangeSeparationLayer):
                if layer.cutoff_fn is not None:
                    raise ValueError(
                        "DPA4C requires RangeSeparation without cutoff_fn. "
                        "The neighbor list is a hard ball and the C³ envelope "
                        "is the only smooth cutoff."
                    )
                if float(layer.cutoff_sr) + 1.0e-8 < self.cutoff_sr:
                    raise ValueError(
                        f"RangeSeparation cutoff_sr ({layer.cutoff_sr}) must be "
                        f">= DPA4C cutoff_sr ({self.cutoff_sr})"
                    )

    def reset_output_calibration(self) -> None:
        """Restore the identity preconditioner."""
        self.calib_mean.zero_()
        self.calib_scale.fill_(1.0)

    @torch.no_grad()
    def fit_output_calibration(self, raw: Tensor) -> None:
        """Fix the diagonal preconditioner from uncalibrated descriptor rows.

        Geometric columns with RMS above :data:`CALIBRATION_RMS_FLOOR` are
        scaled to the root-mean-square of the center embedding and keep a zero
        stored mean. A column at or below that floor stays at scale 1, so a
        feature that is zero on this sample is neither rejected nor amplified.
        The two neighborhood masses are standardized. The center embedding
        itself is left unchanged. Pass the tensor returned by
        :meth:`raw_descriptor`.
        """
        if raw.ndim != 2 or raw.shape[-1] != self.dim_feature_out:
            raise ValueError(
                f"raw descriptor must have shape (N, {self.dim_feature_out}), "
                f"got {tuple(raw.shape)}"
            )
        if raw.shape[0] < 2:
            raise ValueError("output calibration needs at least two atoms")
        values = raw.detach().to(dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("output calibration received a non-finite descriptor")
        channels = self.channels
        geometry = values[:, :-channels]
        tail = values[:, -channels:]
        target = torch.sqrt(torch.mean(tail * tail))
        if not torch.isfinite(target) or float(target) <= 0.0:
            raise ValueError(f"center embedding RMS is degenerate ({float(target)})")
        rms = torch.sqrt(torch.mean(geometry * geometry, dim=0))
        if not torch.isfinite(rms).all():
            raise ValueError("geometric descriptor columns must have a finite RMS")
        mean = torch.zeros(self.dim_feature_out, dtype=torch.float64)
        scale = torch.ones(self.dim_feature_out, dtype=torch.float64)
        alive = rms > CALIBRATION_RMS_FLOOR
        scale[:-channels] = torch.where(alive, rms / target, torch.ones_like(rms))
        mass = geometry[:, -2:]
        mass_mean = mass.mean(dim=0)
        mass_std = torch.sqrt(torch.mean((mass - mass_mean) ** 2, dim=0))
        if bool((mass_std <= 1.0e-12).any()):
            raise ValueError(
                "neighborhood masses are constant on this calibration sample"
            )
        mean[-channels - 2 : -channels] = mass_mean
        scale[-channels - 2 : -channels] = mass_std / target
        self.calib_mean.copy_(mean.to(dtype=self.calib_mean.dtype, device=self.calib_mean.device))
        self.calib_scale.copy_(scale.to(dtype=self.calib_scale.dtype, device=self.calib_scale.device))

    def raw_descriptor(
        self,
        atom_embedding: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Tensor:
        """Uncalibrated invariant vector, shape ``(N, dim_feature_out)``."""
        return self._assemble(atom_embedding, idx_i_sr, idx_j_sr, vij_sr)

    def get_output(
        self,
        atom_embedding: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        raw = self._assemble(atom_embedding, idx_i_sr, idx_j_sr, vij_sr)
        scale = self.calib_scale.to(dtype=raw.dtype)
        mean = self.calib_mean.to(dtype=raw.dtype)
        return {"atom_feature": (raw - mean) / scale}

    def _assemble(
        self,
        atom_embedding: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
    ) -> Tensor:
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.channels:
            raise ValueError(
                f"atom_embedding must have shape (N, {self.channels}), "
                f"got {tuple(atom_embedding.shape)}"
            )
        if idx_i_sr.shape != idx_j_sr.shape or idx_i_sr.ndim != 1:
            raise ValueError("idx_i_sr and idx_j_sr must be 1D and the same length")
        if vij_sr.ndim != 2 or vij_sr.shape[-1] != 3 or vij_sr.shape[0] != idx_i_sr.shape[0]:
            raise ValueError(
                "vij_sr must have shape (E, 3) for the same edges as idx_i_sr and idx_j_sr, "
                f"got {tuple(vij_sr.shape)} and {tuple(idx_i_sr.shape)}"
            )
        nodes = atom_embedding.shape[0]
        center = idx_i_sr.long()
        neighbor = idx_j_sr.long()
        if center.numel():
            if int(center.min()) < 0 or int(neighbor.min()) < 0:
                raise ValueError("edge indices must be non-negative")
            if int(center.max()) >= nodes or int(neighbor.max()) >= nodes:
                raise ValueError("edge indices must address rows of atom_embedding")
            if bool(torch.any(center == neighbor)):
                raise ValueError(
                    "DPA4C edges must not contain self-loops; a zero displacement "
                    "has no direction for the Cartesian harmonics"
                )

        moments, masses = self._moments(atom_embedding, center, neighbor, vij_sr, nodes)
        invariants = self.readout(moments)
        return torch.cat((invariants, masses, atom_embedding), dim=-1)

    def _moments(
        self,
        embedding: Tensor,
        center: Tensor,
        neighbor: Tensor,
        vij: Tensor,
        nodes: int,
    ) -> tuple[Tensor, Tensor]:
        dtype = embedding.dtype
        if center.numel() == 0:
            moments = embedding.new_zeros(nodes, self.readout.offsets[-1])
            masses = embedding.new_full((nodes, 2), 0.5)
            return moments, masses

        displacement = vij.to(dtype=dtype)
        rho = torch.sqrt(torch.sum(displacement * displacement, dim=-1, keepdim=True) + DISTANCE_EPS**2)
        direction = displacement / rho
        envelope = self.envelope(rho).reshape(-1)
        radial_basis = self.radial_basis(rho.reshape(-1))
        shared, modes = self.radial(radial_basis)
        scale, shift, mixing = self.pair_film(embedding[center], embedding[neighbor])
        amplitude = scale * shared + shift
        if mixing is not None:
            amplitude = amplitude + torch.sum(mixing * modes.unsqueeze(-2), dim=-1)
        enveloped = amplitude * envelope.unsqueeze(-1)

        chi2 = envelope * envelope
        mass_0 = reduce_edge(chi2.unsqueeze(-1), center, (nodes, 1)).reshape(nodes)
        mass_1 = reduce_edge((chi2 * chi2).unsqueeze(-1), center, (nodes, 1)).reshape(nodes)
        divisor_0 = torch.sqrt(mass_0 + NORM_FLOOR)
        divisor_1 = torch.sqrt(mass_1 + NORM_FLOOR)

        scalar = reduce_edge(enveloped, center, (nodes, self.channels)) / divisor_0.unsqueeze(-1)
        harmonics = cartesian_harmonics(direction, self.lmax)
        angular_parts = []
        for degree, width in enumerate(self.degree_channels):
            if degree == 0:
                continue
            block = harmonics[:, degree * degree : (degree + 1) * (degree + 1)]
            message = block.unsqueeze(-1) * enveloped[:, None, :width]
            message = message.reshape(message.shape[0], -1) * envelope.unsqueeze(-1)
            angular_parts.append(message)
        angular = torch.cat(angular_parts, dim=-1)
        directional = reduce_edge(angular, center, (nodes, angular.shape[-1]))
        directional = directional / divisor_1.unsqueeze(-1)
        moments = torch.cat((scalar, directional), dim=-1)
        masses = torch.stack((divisor_0, divisor_1), dim=-1)
        return moments, masses
