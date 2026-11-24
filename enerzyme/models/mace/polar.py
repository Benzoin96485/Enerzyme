"""MACE-POLAR-1 long-range layer.

The short-range Core is unchanged. This layer reads its per-layer equivariant
features and adds the polarizable electrostatics from Batatia et al. (2026):

1. A linear map of every layer is summed into a spin-resolved Gaussian multipole.
2. Fukui weights from the last layer equilibrate the monopoles to
   ``(Q ± S) / 2``. ``Q`` is the molecular charge and ``S`` is the multiplicity
   minus one, the same convention as :class:`NeuralSpinChargeEquilibrationLayer`.
3. For ``num_update`` iterations the Coulomb potential of each spin density is
   projected onto atom-centered Gaussians, a local equivariant update adds a
   correction to every multipole component, and the monopoles are equilibrated
   again. Higher multipoles are not rescaled.
4. The Coulomb energy uses the final total density ``p↑ + p↓``. The nonlocal
   MLP uses that same density and the potential recomputed from it.

An external applied field is not an input of this layer (``v_app = 0``).
Open boundaries only: a ``pbc`` tensor with any true entry raises
``NotImplementedError``. Real-space molecules do not need graph_electrostatics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..mp.ops import scatter_sum
from ..layers._base_layer import BaseFFLayer
from ..layers.gto_electrostatics import GaussianMultipoleEnergy, GaussianPotentialFeatures
from .blocks import (
    GatedIrrepsMLP,
    InvariantChannelDot,
    ScalarModulation,
    SiLUMLP,
    feature_spec,
    stack_node_features,
    uniform_multiplicity,
)


# Same small message-passing stack as plain MACE, plus the polarizable field.
# Published MACE-POLAR-1-M / L use 512 channels, correlation 3, max_ell 3,
# r_max 6 Å, and 2 (M) or 3 (L) interactions. σ_charge = 1.5 Å, field widths
# 1.5 Å and 2.0 Å, l ≤ 1, two density updates. Energies of the Coulomb term
# are eV when Ra is Å. A Hartree dataset sets energy_scale: 0.0367493
# (1/27.211386245988). Set avg_num_neighbors to the dataset mean.
DEFAULT_BUILD_PARAMS_POLAR = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "avg_num_neighbors": 8.0,
}
DEFAULT_LAYER_PARAMS_POLAR = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "mace", "trainable": False, "polynomial_p": 5},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "max_ell": 2,
            "additional_hidden_irreps": "32x1o",
            "correlation": 2,
            "num_interactions": 2,
            "output_mode": "layers",
            "keep_last_layer_irreps": True,
        },
    },
    {"name": "MACEEnergyReadout"},
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
            "scales": {"Ea": {"values": 1.0, "learnable": True}},
        },
    },
    {"name": "PolarField"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _spin_irreps(max_l: int) -> o3.Irreps:
    spherical = o3.Irreps.spherical_harmonics(max_l)
    # Concatenate, do not simplify: e3nn stores this as two SH blocks, and
    # view(N, 2, (l+1)^2) is then [spin, lm].
    return spherical + spherical


def _field_irreps(max_l: int, n_radial: int) -> o3.Irreps:
    spherical = o3.Irreps.spherical_harmonics(max_l)
    return (spherical * n_radial).sort()[0].simplify()


class _LayerLinearSum(nn.Module):
    def __init__(self, irreps_in, irreps_out, n_layers: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [o3.Linear(irreps_in, irreps_out) for _ in range(n_layers)]
        )

    def forward(self, layers: Tensor) -> Tensor:
        total = self.linears[0](layers[:, :, 0])
        for index, linear in enumerate(self.linears[1:], start=1):
            total = total + linear(layers[:, :, index])
        return total


class OneBodyFieldUpdate(nn.Module):
    """Local equivariant correction of the spin density, plus a new Fukui pair.

    ``V`` is a bias-carrying linear mix of the potential, the mixed node
    features, and the current spin density. An invariant channel dot of the
    node features with ``V`` is concatenated with a linear map of the species
    one-hot and passed through a SiLU MLP. Those scalars modulate the node
    features, and a two-layer gated readout emits ``Δp`` then ``(f↑, f↓)``.
    """

    def __init__(
        self,
        node_irreps: o3.Irreps,
        spin_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        max_Za: int,
        readout_irreps: str = "64x0e+32x1o",
        mlp_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.spin_irreps = o3.Irreps(spin_irreps)
        self.n_channels = uniform_multiplicity(self.node_irreps)
        self.embed_potential = o3.Linear(potential_irreps, self.node_irreps)
        self.embed_features = o3.Linear(self.node_irreps, self.node_irreps)
        self.embed_charges = o3.Linear(self.spin_irreps, self.node_irreps)
        self.species = o3.Linear(
            o3.Irreps(f"{max_Za + 1}x0e"),
            o3.Irreps(f"{self.n_channels}x0e"),
        )
        self.dot = InvariantChannelDot(self.node_irreps)
        hidden = int(mlp_hidden)
        self.mlp = SiLUMLP(
            [2 * self.n_channels, hidden, hidden, hidden, self.n_channels]
        )
        self.modulate = ScalarModulation(self.node_irreps)
        hidden_irreps = o3.Irreps(readout_irreps)
        if self.node_irreps.lmax == 0:
            hidden_irreps = o3.Irreps(
                [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0]
            )
            if hidden_irreps.dim == 0:
                hidden_irreps = o3.Irreps("16x0e")
        self.readout = GatedIrrepsMLP(
            self.node_irreps,
            hidden_irreps,
            self.spin_irreps + o3.Irreps("2x0e"),
        )

    def forward(
        self,
        node_features: Tensor,
        node_attrs: Tensor,
        potential: Tensor,
        spin_flat: Tensor,
    ) -> Tensor:
        mixed = (
            self.embed_potential(potential)
            + self.embed_features(node_features)
            + self.embed_charges(spin_flat)
        )
        invariants = self.dot(node_features, mixed)
        species = self.species(node_attrs)
        scales = self.mlp(torch.cat([invariants, species], dim=-1))
        hidden = self.modulate(node_features, scales)
        return self.readout(hidden)


class NonlocalEnergyHead(nn.Module):
    """Invariant MLP of the final density and the final potential.

    The density argument is the final spin density only. Adding the initial
    density again would count the baseline twice.
    """

    def __init__(
        self,
        node_irreps: o3.Irreps,
        spin_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        n_channels = uniform_multiplicity(self.node_irreps)
        self.embed_density = o3.Linear(spin_irreps, self.node_irreps)
        self.embed_potential = o3.Linear(potential_irreps, self.node_irreps)
        self.dot_density = InvariantChannelDot(self.node_irreps)
        self.dot_potential = InvariantChannelDot(self.node_irreps)
        width = int(hidden)
        self.mlp = SiLUMLP([2 * n_channels, width, width, width, 1])

    def forward(self, node_features: Tensor, spin_flat: Tensor, potential: Tensor) -> Tensor:
        density_invariants = self.dot_density(node_features, self.embed_density(spin_flat))
        potential_invariants = self.dot_potential(
            node_features, self.embed_potential(potential)
        )
        return self.mlp(torch.cat([density_invariants, potential_invariants], dim=-1)).squeeze(-1)


class PolarFieldLayer(BaseFFLayer):
    def __init__(
        self,
        max_Za: int,
        built_layers=None,
        feature_irreps: Optional[str] = None,
        num_output_blocks: Optional[int] = None,
        density_max_l: int = 1,
        density_smearing_width: float = 1.5,
        field_max_l: int = 1,
        field_smearing_widths: Optional[List[float]] = None,
        num_update: int = 2,
        include_self_interaction: bool = True,
        include_feature_self_interaction: bool = False,
        add_nonlocal_energy: bool = True,
        readout_irreps: str = "64x0e+32x1o",
        fukui_irreps: str = "16x0e",
        mlp_hidden: int = 64,
        nonlocal_hidden: int = 128,
        energy_scale: float = 1.0,
        energy_offset: float = 0.02,
        feature_offset: float = 0.1,
        field_feature_norms: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            output_fields={
                "E_nonlocal_a",
                "E_coulomb_a",
                "Qa",
                "Sa",
                "dipole",
                "spin_multipole",
            }
        )
        self.max_Za = int(max_Za)
        self.node_irreps, self.num_output_blocks = feature_spec(
            built_layers, feature_irreps, num_output_blocks
        )
        uniform_multiplicity(self.node_irreps)
        if field_smearing_widths is None:
            field_smearing_widths = [1.5, 2.0]
        self.density_max_l = int(density_max_l)
        self.field_max_l = int(field_max_l)
        self.num_update = int(num_update)
        if self.num_update < 0:
            raise ValueError(f"num_update must be >= 0, got {num_update}")
        self.add_nonlocal_energy = bool(add_nonlocal_energy)
        self.spin_irreps = _spin_irreps(self.density_max_l)
        self.nlm = (self.density_max_l + 1) ** 2
        self.field_irreps = _field_irreps(self.field_max_l, len(field_smearing_widths))
        self.potential_irreps = self.field_irreps + self.field_irreps
        self.sources = _LayerLinearSum(
            self.node_irreps, self.spin_irreps, self.num_output_blocks
        )
        self.mixer = _LayerLinearSum(
            self.node_irreps, self.node_irreps, self.num_output_blocks
        )
        self.fukui = GatedIrrepsMLP(self.node_irreps, fukui_irreps, "2x0e")
        self.updates = nn.ModuleList(
            [
                OneBodyFieldUpdate(
                    self.node_irreps,
                    self.spin_irreps,
                    self.potential_irreps,
                    self.max_Za,
                    readout_irreps=readout_irreps,
                    mlp_hidden=mlp_hidden,
                )
                for _ in range(self.num_update)
            ]
        )
        self.nonlocal_head = NonlocalEnergyHead(
            self.node_irreps,
            self.spin_irreps,
            self.potential_irreps,
            hidden=nonlocal_hidden,
        )
        self.potential = GaussianPotentialFeatures(
            density_max_l=self.density_max_l,
            density_sigma=density_smearing_width,
            projection_max_l=self.field_max_l,
            projection_sigmas=field_smearing_widths,
            include_self=include_feature_self_interaction,
            offset=feature_offset,
        )
        if self.potential.out_dim != self.field_irreps.dim:
            raise RuntimeError(
                "Potential feature dimension "
                f"{self.potential.out_dim} does not match {self.field_irreps}."
            )
        self.coulomb = GaussianMultipoleEnergy(
            max_l=self.density_max_l,
            sigma=density_smearing_width,
            include_self=include_self_interaction,
            offset=energy_offset,
            energy_scale=energy_scale,
        )
        if field_feature_norms is None:
            norms = torch.ones(self.potential_irreps.dim)
        else:
            norms = torch.tensor([float(value) for value in field_feature_norms])
            if int(norms.numel()) != self.potential_irreps.dim:
                raise ValueError(
                    f"field_feature_norms has length {norms.numel()}, "
                    f"expected {self.potential_irreps.dim}."
                )
        self.register_buffer("field_feature_norms", norms)

    def _graph_total(
        self, value: Optional[Tensor], batch: Tensor, n_graphs: int, dtype: torch.dtype
    ) -> Tensor:
        if value is None:
            return torch.zeros(n_graphs, dtype=dtype, device=batch.device)
        flat = torch.as_tensor(value, device=batch.device, dtype=dtype).reshape(-1)
        if flat.numel() == 1:
            return flat.expand(n_graphs)
        if flat.numel() == n_graphs:
            return flat
        if flat.numel() == batch.numel():
            order = torch.arange(batch.numel(), device=batch.device)
            first = torch.full(
                (n_graphs,), batch.numel(), dtype=torch.long, device=batch.device
            )
            first.scatter_reduce_(0, batch, order, reduce="amin", include_self=True)
            return flat[first]
        raise ValueError(
            f"Q and S must be a scalar, one value per molecule, or one value per atom. "
            f"Got {flat.numel()} values for {n_graphs} molecules and {batch.numel()} atoms."
        )

    def _normalize_fukui(self, fukui: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
        totals = scatter_sum(fukui.double(), batch, n_graphs)
        denom = totals[batch].to(dtype=fukui.dtype)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        return fukui / denom

    def _equilibrate(
        self,
        density: Tensor,
        fukui: Tensor,
        charge: Tensor,
        spin: Tensor,
        batch: Tensor,
    ) -> Tensor:
        n_graphs = int(charge.shape[0])
        weights = self._normalize_fukui(fukui, batch, n_graphs)
        predicted = scatter_sum(density[:, :, 0].double(), batch, n_graphs)
        predicted = predicted[batch].to(dtype=density.dtype)
        target_up = (0.5 * (charge + spin))[batch]
        target_down = (0.5 * (charge - spin))[batch]
        updated = density.clone()
        updated[:, 0, 0] = density[:, 0, 0] + weights[:, 0] * (target_up - predicted[:, 0])
        updated[:, 1, 0] = density[:, 1, 0] + weights[:, 1] * (target_down - predicted[:, 1])
        return updated

    def _spin_potential(self, density: Tensor, positions: Tensor, batch: Tensor) -> Tensor:
        up = self.potential(density[:, 0, :], positions, batch)
        down = self.potential(density[:, 1, :], positions, batch)
        return torch.cat((up, down), dim=-1) / self.field_feature_norms.to(dtype=up.dtype)

    def _dipole(self, total_density: Tensor, positions: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
        moment = positions * total_density[:, :1]
        dipole = scatter_sum(moment, batch, n_graphs)
        if total_density.shape[-1] > 1:
            angular = scatter_sum(total_density[:, 1:4], batch, n_graphs)
            dipole = dipole + angular[:, [2, 0, 1]]
        return dipole

    def get_output(
        self,
        atom_feature: Tensor,
        Za: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if pbc is not None and torch.as_tensor(pbc).bool().any():
            raise NotImplementedError(
                "MACE-POLAR-1 periodic evaluation is not implemented. "
                "Open-boundary molecules omit pbc. Periodic k-space needs "
                "graph_electrostatics (https://github.com/WillBaldwin0/graph_electrostatics)."
            )
        n_atoms = int(Ra.shape[0])
        if batch_seg is None:
            batch_seg = torch.zeros(n_atoms, dtype=torch.long, device=Ra.device)
        else:
            batch_seg = batch_seg.to(dtype=torch.long, device=Ra.device)
        if n_atoms == 0:
            empty = Ra.new_zeros(0)
            return {
                "E_nonlocal_a": empty,
                "E_coulomb_a": empty,
                "Qa": empty,
                "Sa": empty,
                "dipole": Ra.new_zeros(0, 3),
                "spin_multipole": Ra.new_zeros(0, 2, self.nlm),
            }
        n_graphs = int(batch_seg.max().item()) + 1
        charge = self._graph_total(Q, batch_seg, n_graphs, Ra.dtype)
        spin = self._graph_total(S, batch_seg, n_graphs, Ra.dtype)
        layers = stack_node_features(atom_feature, self.num_output_blocks)
        if layers.shape[1] != self.node_irreps.dim:
            raise ValueError(
                f"atom_feature width {layers.shape[1]} does not match {self.node_irreps} "
                f"(dim {self.node_irreps.dim})."
            )
        mixed = self.mixer(layers)
        raw = self.sources(layers).view(n_atoms, 2, self.nlm)
        initial_fukui = self.fukui(layers[:, :, -1])
        density = self._equilibrate(raw, initial_fukui, charge, spin, batch_seg)
        one_hot = torch.nn.functional.one_hot(Za.long(), num_classes=self.max_Za + 1)
        one_hot = one_hot.to(dtype=layers.dtype)
        for update in self.updates:
            potential = self._spin_potential(density, Ra, batch_seg)
            correction = update(
                mixed,
                one_hot,
                potential,
                density.reshape(n_atoms, self.spin_irreps.dim),
            )
            delta = correction[:, : self.spin_irreps.dim].view(n_atoms, 2, self.nlm)
            next_fukui = correction[:, self.spin_irreps.dim :]
            density = self._equilibrate(density + delta, next_fukui, charge, spin, batch_seg)
        final_potential = self._spin_potential(density, Ra, batch_seg)
        if self.add_nonlocal_energy:
            nonlocal_energy = self.nonlocal_head(
                layers[:, :, -1],
                density.reshape(n_atoms, self.spin_irreps.dim),
                final_potential,
            )
        else:
            nonlocal_energy = Ra.new_zeros(n_atoms)
        total_density = density.sum(dim=1)
        coulomb = self.coulomb.atomic_energy(total_density, Ra, batch_seg)
        return {
            "E_nonlocal_a": nonlocal_energy,
            "E_coulomb_a": coulomb,
            "Qa": total_density[:, 0],
            "Sa": density[:, 0, 0] - density[:, 1, 0],
            "dipole": self._dipole(total_density, Ra, batch_seg, n_graphs),
            "spin_multipole": density,
        }
