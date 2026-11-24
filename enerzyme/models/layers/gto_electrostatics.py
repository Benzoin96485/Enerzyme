"""Open-boundary Gaussian multipole electrostatics.

Real-space l ≤ 1 evaluation used by MACE-POLAR-1 (Batatia et al., 2026).
The pair kernel and the displaced-charge projection follow
``graph_longrange.realspace_electrostatics`` (WillBaldwin0/graph_electrostatics).
Positions are Å and charges are electron charges. The pair energy is then in eV:

    K = 1 / (4 π ε₀) ≈ 14.3996 eV·Å,
    E_ij = K q_i q_j erf(r / (2σ)) / r.

Each directed edge carries half of that pair, so the two directions add up to
``E_ij``. This is the kernel in the paper's real-space implementation (the
``erf(r / (2σ))`` form), with ``σ`` the common Gaussian width. Self-energy is
the on-site overlap and does not depend on the neighbor list. A monopole
depends only on distance. An ``l = 1`` density is four displaced charges along
the laboratory axes (energy offset 0.02 Å, feature offset 0.1 Å, matching
graph_electrostatics). That stencil is equivariant as the offset goes to zero.
The analytic energy and field methods evaluate the isotropic Gaussian kernel
and its derivatives on native Cartesian dipoles. Periodic k-space is not implemented here.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

# ε₀ in e·V⁻¹·Å⁻¹. FIELD_CONSTANT / (4π) is 1/(4πε₀) in eV·Å.
FIELD_CONSTANT = 1.0 / (5.526349406e-3)


def coulomb_prefactor() -> float:
    """``1/(4πε₀)`` in eV·Å when the distance unit is Å."""
    return FIELD_CONSTANT / (4.0 * math.pi)


def _as_width_list(widths: Sequence[float]) -> List[float]:
    values = [float(width) for width in widths]
    if not values:
        raise ValueError("at least one Gaussian width is required")
    if any(width <= 0.0 for width in values):
        raise ValueError(f"Gaussian widths must be positive, got {values}")
    return values


def _check_l(max_l: int, what: str) -> int:
    level = int(max_l)
    if level < 0 or level > 1:
        raise NotImplementedError(
            f"{what} only supports l = 0 and l = 1 in this real-space port, got {level}. "
            "Periodic and higher-order evaluation needs graph_electrostatics "
            "(https://github.com/WillBaldwin0/graph_electrostatics)."
        )
    return level


def _cl_sigma(ell: int, sigma: float, normalize: str) -> float:
    """Reciprocal of the GTO normalization denominator. Computed in float64."""
    if normalize not in ("multipoles", "receiver", "none"):
        raise ValueError(
            "normalize must be 'multipoles', 'receiver', or 'none', "
            f"got {normalize!r}"
        )
    if normalize == "none":
        return 1.0
    sigma_t = torch.tensor(float(sigma), dtype=torch.float64)
    if normalize == "multipoles":
        l_dep = (
            math.sqrt(4.0 * math.pi / (2 * ell + 1))
            * 2.0 ** ((2 * ell + 1) / 2.0)
            * torch.exp(torch.lgamma(torch.tensor((2 * ell + 3) / 2.0, dtype=torch.float64)))
        )
        denom = l_dep * sigma_t ** (2 * ell + 3)
    else:
        l_dep = 2.0 ** ((ell + 1) / 2.0) * torch.exp(
            torch.lgamma(torch.tensor((ell + 3) / 2.0, dtype=torch.float64))
        )
        denom = l_dep * sigma_t ** (ell + 3)
    return float(1.0 / denom)


def _overlap_integral(ell: int, sigma_source: float, sigma_receive: float) -> float:
    """Radial self-overlap on the graph_electrostatics grid, in float64."""
    extent = 10.0 * max(float(sigma_source), float(sigma_receive))
    grid = torch.linspace(1.0e-4, extent, 10000, dtype=torch.float64)
    exponent = (2 * ell + 3) / 2.0
    reduced = 0.5 * grid.square() / (sigma_receive ** 2)
    gamma_full = torch.exp(torch.lgamma(torch.tensor(exponent, dtype=torch.float64)))
    gamma_shift = torch.exp(torch.lgamma(torch.tensor(ell + 0.5, dtype=torch.float64)))
    lower = torch.special.gammainc(
        torch.tensor(exponent, dtype=torch.float64), reduced
    ) * gamma_full
    lower = lower - torch.special.gammainc(
        torch.tensor(ell + 1.5, dtype=torch.float64), torch.zeros_like(reduced)
    ) * gamma_shift
    radial_tail = (
        2.0 ** (ell + 0.5)
        * sigma_receive ** (2 * ell + 3)
        * lower
        * grid ** (-(ell + 1))
    )
    near = (sigma_receive ** 2) * grid ** ell * torch.exp(
        -0.5 * grid.square() / (sigma_receive ** 2)
    )
    integrand = (
        grid ** (ell + 2)
        * torch.exp(-0.5 * grid.square() / (sigma_source ** 2))
        * (radial_tail + near)
    )
    return float(torch.trapezoid(integrand, grid))


def _overlap_tables(
    l_source: int,
    sigma_source: float,
    l_receive: int,
    sigmas_receive: Sequence[float],
    normalize_source: str,
    normalize_receive: str,
) -> Tuple[Tensor, Tensor, int]:
    """Prefix overlap constants and the source multipole each one multiplies."""
    n_radial = len(sigmas_receive)
    feature_dim = n_radial * (l_receive + 1) ** 2
    values: List[float] = []
    source_index: List[int] = []
    prefactor_unit = FIELD_CONSTANT
    for ell in range(min(l_source, l_receive) + 1):
        for sigma_receive in sigmas_receive:
            integral = _overlap_integral(ell, sigma_source, float(sigma_receive))
            scale = (
                _cl_sigma(ell, sigma_source, normalize_source)
                * _cl_sigma(ell, float(sigma_receive), normalize_receive)
                * prefactor_unit
                / (2 * ell + 1)
                * integral
            )
            for component in range(2 * ell + 1):
                values.append(scale)
                source_index.append(ell ** 2 + component)
    overlap = torch.tensor(values, dtype=torch.float64)
    index = torch.tensor(source_index, dtype=torch.long)
    return overlap, index, feature_dim


PairIndices = Tuple[Tensor, Tensor]


def _pair_index(batch: Tensor, n_images: int, pairs: Optional[PairIndices] = None) -> PairIndices:
    """Directed pairs of images whose original atoms differ and share a graph.

    ``n_images=1`` is the complete graph without self-pairs. ``n_images=4`` is
    the displaced-charge graph: the four images of one atom do not interact,
    and that self-energy is added separately.
    """
    if pairs is not None:
        return pairs
    device = batch.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    n_atoms = int(batch.shape[0])
    if n_atoms == 0 or n_images < 1:
        return empty, empty
    batch = batch.long()
    origin = torch.arange(n_atoms, device=device).repeat_interleave(n_images)
    batch_images = batch.repeat_interleave(n_images)
    n_graphs = int(batch.max().item()) + 1
    senders: List[Tensor] = []
    receivers: List[Tensor] = []
    for graph in range(n_graphs):
        nodes = torch.nonzero(batch_images == graph, as_tuple=False).view(-1)
        count = int(nodes.numel())
        if count <= 1:
            continue
        left = nodes.view(-1, 1).expand(count, count).reshape(-1)
        right = nodes.view(1, -1).expand(count, count).reshape(-1)
        keep = origin[left] != origin[right]
        senders.append(left[keep])
        receivers.append(right[keep])
    if not senders:
        return empty, empty
    return torch.cat(senders), torch.cat(receivers)


def _displaced_positions(positions: Tensor, shift_x: Tensor, shift_y: Tensor, shift_z: Tensor) -> Tensor:
    """Four images per atom: center, +x, +y, +z. Gradient flows back to ``positions``."""
    n_atoms = positions.shape[0]
    images = positions.new_zeros(n_atoms, 4, 3)
    images[:, 1] = shift_x.to(dtype=positions.dtype, device=positions.device)
    images[:, 2] = shift_y.to(dtype=positions.dtype, device=positions.device)
    images[:, 3] = shift_z.to(dtype=positions.dtype, device=positions.device)
    return (positions.unsqueeze(1) + images).reshape(n_atoms * 4, 3)


def _pair_node_energy(
    charges: Tensor,
    positions: Tensor,
    sender: Tensor,
    receiver: Tensor,
    sigma: float,
) -> Tensor:
    """Half-pair energies scattered onto the receiver. Summing them is the pair energy."""
    node = charges.new_zeros(charges.shape[0])
    if sender.numel() == 0:
        return node
    displacement = positions[receiver] - positions[sender]
    distance = torch.linalg.norm(displacement, dim=-1)
    smooth = torch.erf(distance * 0.5 / sigma) / (distance.abs() + 1.0e-6)
    edge = (
        0.5
        * FIELD_CONSTANT
        * smooth
        * charges[sender]
        * charges[receiver]
        / (4.0 * math.pi)
    )
    node.index_add_(0, receiver, edge)
    return node


def _scalar_projection(
    charges: Tensor,
    positions: Tensor,
    sender: Tensor,
    receiver: Tensor,
    total_width: Tensor,
) -> Tensor:
    """Smeared Coulomb scalars, one column per receiver width. Shape ``[N, n_radial]``."""
    n_radial = int(total_width.numel())
    out = charges.new_zeros(charges.shape[0], n_radial)
    if sender.numel() == 0:
        return out
    displacement = positions[sender] - positions[receiver]
    distance = torch.linalg.norm(displacement, dim=-1, keepdim=True)
    smooth = torch.erf(0.5 * distance / total_width) / (distance + 1.0e-6)
    contrib = charges[sender].unsqueeze(-1) * smooth
    out.index_add_(0, receiver, contrib)
    return out * (FIELD_CONSTANT / (4.0 * math.pi))


def _gaussian_moments(t: Tensor):
    """Integrals M_n(t) = ∫_0^1 u^(2n) exp(-tu²) du for n=0,1,2."""
    # The recurrence divides by t twice. Use its analytic continuation near
    # zero, including co-located atoms, to keep force derivatives finite.
    small = t < 0.3
    regular = torch.where(small, torch.ones_like(t), t)
    root = regular.sqrt()
    exp = (-regular).exp()
    m0 = (0.5 * math.sqrt(math.pi)) * torch.erf(root) / root
    m1 = (m0 - exp) / (2 * regular)
    m2 = (3 * m1 - exp) / (2 * regular)
    local = torch.where(small, t, torch.zeros_like(t))
    moments = []
    for degree, standard in enumerate((m0, m1, m2)):
        series = 0.0
        for k in range(9, -1, -1):
            coefficient = 1.0 / (math.factorial(k) * (2 * degree + 2 * k + 1))
            series = coefficient - local * series
        moments.append(torch.where(small, series, standard))
    return moments


def _analytic_field(multipoles: Tensor, positions: Tensor, batch: Tensor, widths: Tensor, pairs: Optional[PairIndices] = None):
    """Potential and Cartesian gradient of Gaussian charges and xyz dipoles.

    For receiver minus source displacement x, the source dipole acts as
    -p·∇_x on erf(a|x|)/|x|. A receiver dipole acts as +p·∇_x.
    """
    sender, receiver = _pair_index(batch, 1, pairs)
    r = positions[receiver] - positions[sender]
    a = 0.5 / widths.to(positions).reshape(1, -1)
    t = r.square().sum(-1, keepdim=True) * a.square()
    m0, m1, m2 = _gaussian_moments(t)
    factor = (2 * coulomb_prefactor() / math.sqrt(math.pi)) * a
    kernel = factor * m0
    grad_factor = -2 * factor * a.square() * m1
    hess_factor = 4 * factor * a.pow(4) * m2
    charge = multipoles[sender, :1]
    dipole = multipoles[sender, 1:4] if multipoles.shape[1] == 4 else torch.zeros_like(r)
    radial_dipole = (dipole * r).sum(-1, keepdim=True)
    value = kernel * charge - grad_factor * radial_dipole
    gradient = ((grad_factor * charge - hess_factor * radial_dipole).unsqueeze(-1) * r[:, None]
                - grad_factor.unsqueeze(-1) * dipole[:, None])
    return receiver, value, gradient


class _SelfOverlap(nn.Module):
    def __init__(
        self,
        l_source: int,
        sigma_source: float,
        l_receive: int,
        sigmas_receive: Sequence[float],
        normalize_source: str,
        normalize_receive: str,
    ) -> None:
        super().__init__()
        overlap, index, feature_dim = _overlap_tables(
            l_source,
            sigma_source,
            l_receive,
            sigmas_receive,
            normalize_source,
            normalize_receive,
        )
        self.feature_dim = feature_dim
        self.register_buffer("overlap", overlap)
        self.register_buffer("source_index", index)

    def forward(self, multipoles: Tensor) -> Tensor:
        picked = multipoles.index_select(-1, self.source_index)
        out = multipoles.new_zeros(multipoles.shape[0], self.feature_dim)
        width = self.overlap.to(dtype=multipoles.dtype)
        out[:, : width.numel()] = picked * width
        return out


class GaussianMultipoleEnergy(nn.Module):
    """Per-atom smeared Coulomb energy of one real Gaussian multipole density.

    The displaced-charge method takes multipoles in ``(q, y, z, x)`` order.
    The analytic method uses the native e3nn order ``(q, x, y, z)``. The returned vector sums to the graph energy of
    each molecule, so a later segment sum and ``Fa = -∇E`` stay consistent.
    ``include_self`` adds ``½ q·Γ·q`` on each atom. ``energy_scale`` converts
    eV into the dataset unit (Hartree datasets use ``1/27.211386245988``).
    """

    def __init__(
        self,
        max_l: int = 1,
        sigma: float = 1.5,
        include_self: bool = True,
        offset: float = 0.02,
        energy_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_l = _check_l(max_l, "GaussianMultipoleEnergy")
        self.sigma = float(sigma)
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.include_self = bool(include_self)
        self.offset = float(offset)
        self.energy_scale = float(energy_scale)
        self.self_overlap = _SelfOverlap(
            self.max_l,
            self.sigma,
            self.max_l,
            [self.sigma],
            "multipoles",
            "multipoles",
        )
        self.register_buffer("shift_x", torch.tensor([self.offset, 0.0, 0.0]))
        self.register_buffer("shift_y", torch.tensor([0.0, self.offset, 0.0]))
        self.register_buffer("shift_z", torch.tensor([0.0, 0.0, self.offset]))

    def analytic_energy(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        """Evaluate the isotropic kernel directly on native (q, x, y, z) coefficients."""
        widths = positions.new_tensor([self.sigma])
        receiver, value, gradient = _analytic_field(multipoles, positions, batch, widths, pairs)
        pair = multipoles[receiver, 0] * value[:, 0]
        if self.max_l == 1:
            pair = pair + (multipoles[receiver, 1:4] * gradient[:, 0]).sum(-1)
        node = positions.new_zeros(positions.shape[0]).index_add(0, receiver, 0.5 * pair)
        if self.include_self:
            node = node + 0.5 * (multipoles * self.self_overlap(multipoles)).sum(dim=-1)
        return node * self.energy_scale

    def atomic_energy(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        if multipoles.ndim != 2:
            raise ValueError(f"multipoles must be rank 2, got {tuple(multipoles.shape)}")
        if self.max_l == 0:
            node = self._monopole_energy(multipoles[:, 0], positions, batch, pairs)
        else:
            node = self._dipole_energy(multipoles, positions, batch, pairs)
        if self.include_self:
            field = self.self_overlap(multipoles)
            node = node + 0.5 * (multipoles * field).sum(dim=-1)
        return node * self.energy_scale

    def _monopole_energy(self, charges: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        sender, receiver = _pair_index(batch, 1, pairs)
        return _pair_node_energy(charges, positions, sender, receiver, self.sigma)

    def _dipole_energy(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        if multipoles.shape[-1] < 4:
            raise ValueError(
                f"l = 1 multipoles need 4 components (q, y, z, x), got {multipoles.shape[-1]}"
            )
        extended = _displaced_positions(positions, self.shift_x, self.shift_y, self.shift_z)
        charges = positions.new_zeros(extended.shape[0])
        charges[1::4] = multipoles[:, 3] / self.offset
        charges[2::4] = multipoles[:, 1] / self.offset
        charges[3::4] = multipoles[:, 2] / self.offset
        charges[0::4] = multipoles[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
        sender, receiver = _pair_index(batch, 4, pairs)
        node = _pair_node_energy(charges, extended, sender, receiver, self.sigma)
        return node.view(-1, 4).sum(dim=-1)


class GaussianPotentialFeatures(nn.Module):
    """Project a Gaussian multipole density onto atom-centered Gaussian probes.

    For ``n`` widths and ``l = 1``, scalar channels precede the ``n`` vectors.
    The displaced-charge method orders each vector as yzx; the analytic
    method uses native e3nn xyz. ``include_self`` adds
    the on-site overlap (off in MACE-POLAR-1). The probe offset is 0.1 Å.
    """

    def __init__(
        self,
        density_max_l: int = 1,
        density_sigma: float = 1.5,
        projection_max_l: int = 1,
        projection_sigmas: Optional[Sequence[float]] = None,
        include_self: bool = False,
        offset: float = 0.1,
    ) -> None:
        super().__init__()
        self.density_max_l = _check_l(density_max_l, "GaussianPotentialFeatures")
        self.projection_max_l = _check_l(projection_max_l, "GaussianPotentialFeatures")
        self.density_sigma = float(density_sigma)
        if self.density_sigma <= 0.0:
            raise ValueError(f"density_sigma must be positive, got {density_sigma}")
        if projection_sigmas is None:
            projection_sigmas = (1.5, 2.0)
        self.projection_sigmas = _as_width_list(projection_sigmas)
        self.num_radial = len(self.projection_sigmas)
        self.include_self = bool(include_self)
        self.offset = float(offset)
        self.out_dim = self.num_radial * (self.projection_max_l + 1) ** 2
        widths = torch.tensor(self.projection_sigmas, dtype=torch.float64)
        total = torch.sqrt((self.density_sigma ** 2 + widths.square()) / 2.0)
        self.register_buffer("total_width", total)
        l0 = [
            _cl_sigma(0, sigma, "receiver") / _cl_sigma(0, sigma, "multipoles")
            for sigma in self.projection_sigmas
        ]
        l1 = [
            math.sqrt(3.0)
            * sigma ** 2
            * (_cl_sigma(1, sigma, "receiver") / _cl_sigma(0, sigma, "multipoles"))
            / self.offset
            for sigma in self.projection_sigmas
        ]
        self.register_buffer("l0_factors", torch.tensor(l0, dtype=torch.float64))
        self.register_buffer("l1_factors", torch.tensor(l1, dtype=torch.float64))
        self.self_overlap = _SelfOverlap(
            self.density_max_l,
            self.density_sigma,
            self.projection_max_l,
            self.projection_sigmas,
            "multipoles",
            "receiver",
        )
        self.register_buffer("shift_x", torch.tensor([self.offset, 0.0, 0.0]))
        self.register_buffer("shift_y", torch.tensor([0.0, self.offset, 0.0]))
        self.register_buffer("shift_z", torch.tensor([0.0, 0.0, self.offset]))

    def analytic_features(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        """Isotropic projected fields, in native e3nn (xyz) vector order."""
        receiver, value, gradient = _analytic_field(multipoles, positions, batch, self.total_width, pairs)
        packed = self.l0_factors.to(value) * value
        if self.projection_max_l == 1:
            # The stored factor converts finite potential differences. Its
            # product with the probe distance converts the exact gradient.
            l1 = self.l1_factors.to(gradient) * self.offset
            vector = (gradient * l1[:, None]).flatten(1)
            packed = torch.cat((packed, vector), dim=-1)
        features = packed.new_zeros(positions.shape[0], self.out_dim).index_add(0, receiver, packed)
        if self.include_self:
            features = features + self.self_overlap(multipoles)
        return features

    def forward(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self._monopole_on_monopole(multipoles, positions, batch, pairs)
        else:
            source = multipoles
            if self.density_max_l == 0:
                source = multipoles.new_zeros(multipoles.shape[0], 4)
                source[:, 0] = multipoles[:, 0]
            features = self._dipole_projection(source, positions, batch, pairs)
            if self.projection_max_l == 0:
                features = features[:, : self.num_radial]
        if self.include_self:
            features = features + self.self_overlap(multipoles)
        return features

    def _monopole_on_monopole(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        sender, receiver = _pair_index(batch, 1, pairs)
        scalars = _scalar_projection(
            multipoles[:, 0],
            positions,
            sender,
            receiver,
            self.total_width.to(dtype=positions.dtype),
        )
        return self.l0_factors.to(dtype=scalars.dtype) * scalars

    def _dipole_projection(self, multipoles: Tensor, positions: Tensor, batch: Tensor, pairs: Optional[PairIndices] = None) -> Tensor:
        if multipoles.shape[-1] < 4:
            raise ValueError(
                f"l = 1 multipoles need 4 components (q, y, z, x), got {multipoles.shape[-1]}"
            )
        extended = _displaced_positions(positions, self.shift_x, self.shift_y, self.shift_z)
        charges = positions.new_zeros(extended.shape[0])
        charges[1::4] = multipoles[:, 3] / self.offset
        charges[2::4] = multipoles[:, 1] / self.offset
        charges[3::4] = multipoles[:, 2] / self.offset
        charges[0::4] = multipoles[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
        sender, receiver = _pair_index(batch, 4, pairs)
        scalars = _scalar_projection(
            charges,
            extended,
            sender,
            receiver,
            self.total_width.to(dtype=positions.dtype),
        )
        packed = multipoles.new_zeros(multipoles.shape[0], 4 * self.num_radial)
        l0 = self.l0_factors.to(dtype=scalars.dtype)
        l1 = self.l1_factors.to(dtype=scalars.dtype)
        packed[:, : self.num_radial] = l0 * scalars[0::4]
        packed[:, self.num_radial :: 3] = l1 * (scalars[2::4] - scalars[0::4])
        packed[:, self.num_radial + 1 :: 3] = l1 * (scalars[3::4] - scalars[0::4])
        packed[:, self.num_radial + 2 :: 3] = l1 * (scalars[1::4] - scalars[0::4])
        return packed
