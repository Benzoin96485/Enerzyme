"""Single-hop DPA4C moments, invariant readout and output calibration."""

from __future__ import annotations

import torch
from torch import Tensor

from ..layers import BaseAtomEmbedding, RangeSeparationLayer
from ..mp.basis.envelope import C3CutoffEnvelope
from ..mp.basis.radial import BesselC3RadialBasis
from ..mp.block import Block, BlockState
from ..mp.head import Head
from ..mp.ops import scatter_sum
from .defaults import CALIBRATION_RMS_FLOOR, DISTANCE_EPS, ENVELOPE_EXPONENT, NORM_FLOOR
from .geometry import cartesian_harmonics, derive_degree_channels, invariant_width, require_profile
from .radial import OrderedPairFiLM, RadialAmplitude
from .readout import InvariantReadout


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


class DPA4CBlock(Block):
    """Aggregate pair-conditioned Cartesian moments and form polynomial invariants."""

    def __init__(self, dim_embedding, channels, lmax, radial_modes, n_radial, basis_type, cutoff_sr):
        super().__init__()
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

    def forward(self, state, ctx):
        raw = self.raw_descriptor(state.scalar, ctx.receiver, ctx.sender, ctx.vec)
        return BlockState(scalar=raw)

    def raw_descriptor(
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
        if torch.compiler.is_compiling():
            torch._assert_async(((center >= 0) & (neighbor >= 0)).all(), "edge indices must be non-negative")
            torch._assert_async(((center < nodes) & (neighbor < nodes)).all(),
                                "edge indices must address rows of atom_embedding")
            torch._assert_async((center != neighbor).all(), "DPA4C edges must not contain self-loops")
        elif center.numel():
            # Transfer the validation results together, so valid GPU graphs
            # incur a single synchronization regardless of their edge count.
            i_min, i_max = torch.aminmax(center)
            j_min, j_max = torch.aminmax(neighbor)
            bounds = torch.stack((i_min, j_min, i_max, j_max, (center == neighbor).any()))
            low_i, low_j, high_i, high_j, has_loop = bounds.tolist()
            if min(low_i, low_j) < 0:
                raise ValueError("edge indices must be non-negative")
            if max(high_i, high_j) >= nodes:
                raise ValueError("edge indices must address rows of atom_embedding")
            if has_loop:
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
        powers = torch.stack((chi2, chi2 * chi2), dim=-1)
        masses = torch.sqrt(scatter_sum(powers, center, nodes) + NORM_FLOOR)
        scalar = scatter_sum(enveloped, center, nodes) / masses[:, :1]
        angular_amplitude = enveloped[:, :self.degree_channels[1]] * envelope.unsqueeze(-1)
        harmonics = cartesian_harmonics(direction, self.lmax)
        angular_parts = []
        for degree, width in enumerate(self.degree_channels):
            if degree == 0:
                continue
            block = harmonics[:, degree * degree : (degree + 1) * (degree + 1)]
            message = block.unsqueeze(-1) * angular_amplitude[:, None, :width]
            angular_parts.append(message.flatten(1))
        angular = torch.cat(angular_parts, dim=-1)
        directional = scatter_sum(angular, center, nodes)
        directional = directional / masses[:, 1:]
        moments = torch.cat((scalar, directional), dim=-1)
        return moments, masses


class DPA4CHead(Head):
    output_fields = ("atom_feature",)

    def __init__(self, width, channels):
        super().__init__()
        self.channels = channels
        self.dim_feature_out = width
        self.register_buffer("calib_mean", torch.zeros(width))
        self.register_buffer("calib_scale", torch.ones(width))

    def forward(self, state, ctx, taps, inputs):
        raw = state.scalar
        return {"atom_feature": (raw - self.calib_mean.to(dtype=raw.dtype)) / self.calib_scale.to(dtype=raw.dtype)}

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
        mean = values.new_zeros(self.dim_feature_out)
        scale = values.new_ones(self.dim_feature_out)
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


def build_descriptor_stack(core, built_layers) -> None:
    core.adopt_radius_stack(built_layers)
    for layer in built_layers:
        if isinstance(layer, BaseAtomEmbedding) and int(layer.dim_embedding) != core.channels:
            raise ValueError(
                f"atom embedding width {layer.dim_embedding} must equal "
                f"DPA4C channels {core.channels}"
            )
        if isinstance(layer, RangeSeparationLayer):
            if layer.cutoff_fn is not None:
                raise ValueError(
                    "DPA4C requires RangeSeparation without cutoff_fn. "
                    "The neighbor list is a hard ball and the C³ envelope "
                    "is the only smooth cutoff."
                )
            if float(layer.cutoff_sr) + 1.0e-8 < core.cutoff_sr:
                raise ValueError(
                    f"RangeSeparation cutoff_sr ({layer.cutoff_sr}) must be "
                    f">= DPA4C cutoff_sr ({core.cutoff_sr})"
                )
