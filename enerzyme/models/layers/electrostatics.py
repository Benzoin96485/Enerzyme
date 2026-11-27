from typing import Dict, List, Literal, Optional, Sequence, Tuple
import math
import torch
from torch import Tensor
from . import BaseFFLayer
from ..mp.ops import batch_size_from_counts, scatter_sum
from ..mp.basis.envelope import CUTOFF_KEY_TYPE, CUTOFF_REGISTER


class ChargeConservationLayer(BaseFFLayer):
    def __init__(self) -> None:
        r"""
        Correct the atomic charges to make their summation equal to the total charge by [1]

        .. math::

            q_i^{\mathrm{corrected}} = q_i -
            \frac{1}{N}\left(\sum_{j=1}^N q_j - Q\right).

        .. rubric:: References

        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        """
        super().__init__()

    def get_output(
        self, Za: Tensor, Qa: Tensor, 
        Q: Optional[Tensor]=None, batch_seg: Optional[Tensor]=None, N=None,
    ) -> Dict[Literal["Qa", "Q"], Tensor]:
        '''
        Correct the atomic charge

        .. rubric:: Parameters

        Za: Long tensor of atomic numbers, shape [N * batch_size]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        Q: Float tensor of total charges, shape [batch_size]

        batch_seg: Long tensor of batch indices, shape [N * batch_size]

        .. rubric:: Returns

        Qa_corrected: Float tensor of corrected atomic charge, shape [N * batch_size]

        raw_Q: Float tensor of total atomic charge before correction, shape [batch_size]
        '''
        if batch_seg is None:
            batch_seg = torch.zeros_like(Za, dtype=torch.long)
        #number of atoms per batch (needed for charge scaling)
        size = batch_size_from_counts(N)
        N_per_batch = scatter_sum(torch.ones_like(batch_seg), batch_seg, size)
        view_shape = (-1, ) if Qa.dim() == 1 else (-1, 1)
        raw_Q = scatter_sum(Qa, batch_seg, size)
        if Q is None: #assume desired total charge zero if not given
            Q = torch.zeros_like(N_per_batch)
        correction = (Q.view(view_shape) - raw_Q) / N_per_batch.view(view_shape).clamp_min(1)
        # One structure has one correction for every atom. Broadcast directly so
        # its backward is a sum rather than an indexed atomic add to a scalar.
        if size != 1:
            correction = correction[batch_seg]
        return {"Qa": Qa + correction, "Q": raw_Q}


class VelocityConservationLayer(BaseFFLayer):
    """Per graph, subtract the mean so each channel sums to zero over atoms."""

    def __init__(self) -> None:
        super().__init__(
            input_fields={"batch_seg", "Q_vel_a", "S_vel_a"},
            output_fields={"Q_vel_a", "S_vel_a"},
        )

    def get_output(
        self, batch_seg: Tensor, Q_vel_a: Tensor, S_vel_a: Tensor
    ) -> Dict[str, Tensor]:
        ones = torch.ones(batch_seg.shape[0], dtype=Q_vel_a.dtype, device=Q_vel_a.device)
        N_per = scatter_sum(ones, batch_seg)
        mean_q = scatter_sum(Q_vel_a, batch_seg) / N_per
        mean_s = scatter_sum(S_vel_a, batch_seg) / N_per
        return {
            "Q_vel_a": Q_vel_a - mean_q[batch_seg],
            "S_vel_a": S_vel_a - mean_s[batch_seg],
        }


class ElectrostaticEnergyLayer(BaseFFLayer):
    def __init__(
        self,
        cutoff_sr: Optional[float] = None,
        cutoff_lr: Optional[float] = None,
        Bohr_in_R: float = 0.5291772108,
        Hartree_in_E: float = 1,
        dielectric_constant: float = 1,
        cutoff_fn: CUTOFF_KEY_TYPE = "smooth",
        flavor: Literal["PhysNet", "SpookyNet", "SO3LR"] = "SpookyNet",
        electrostatic_energy_scale: float = 4.0,
        neighborlist_format_lr: Literal["sparse", "ordered_sparse"] = "sparse",
    ) -> None:
        r"""
        Pairwise electrostatic energy.

        Flavors:

        - ``PhysNet`` / ``SpookyNet``: shielded Coulomb with smooth SR blend [1].
        - ``SO3LR``: erf-damped Coulomb ``erf(r/σ)/r`` (Kabylda et al., JACS 2025);
          ``electrostatic_energy_scale`` is σ (pretrained SO3LR: 4.0). With
          ``cutoff_lr``, blends energy- and force-shifted forms on
          ``[0.45·Rc, Rc]``.

        .. rubric:: Parameters

        Bohr_in_R / Hartree_in_E: unit conversion (``kehalf = 0.5 * Bohr * Hartree``).

        cutoff_sr: short-range blend cutoff (required for PhysNet/SpookyNet).

        cutoff_lr: long-range cutoff; ignored interactions beyond this distance.

        .. rubric:: References

        [1] J. Chem. Theory Comput. 2019, 15, 3678−3693.
        """
        super().__init__(input_fields={"Dij_lr", "Qa", "idx_i", "idx_j"}, output_fields={"E_ele_a"})
        self.flavor = flavor
        self.cutoff_lr = cutoff_lr
        self.dielectric_constant = dielectric_constant
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E

        if flavor == "SO3LR":
            if neighborlist_format_lr not in {"sparse", "ordered_sparse"}:
                raise ValueError(
                    "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
                )
            self.sigma = float(electrostatic_energy_scale)
            # sparse bidirectional list → kehalf; ordered_sparse → full ke = 2*kehalf
            self.pair_kehalf = (
                self.kehalf
                if neighborlist_format_lr == "sparse"
                else 2.0 * self.kehalf
            )
            self._switch = CUTOFF_REGISTER["smooth"]
            if cutoff_lr is not None and cutoff_lr > 0:
                self.cuton = 0.45 * float(cutoff_lr)
            return

        if cutoff_sr is None:
            raise TypeError("cutoff_sr is required for PhysNet/SpookyNet electrostatics")
        if flavor == "PhysNet":
            self.cutoff = cutoff_sr / 2
            self.cuton = 0
        elif flavor == "SpookyNet":
            self.cutoff = cutoff_sr * 0.75
            self.cuton = cutoff_sr * 0.25
        else:
            raise ValueError(f"Unknown electrostatic flavor: {flavor}")
        self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]

        if cutoff_lr is not None and cutoff_lr > 0:
            self.cutoff_lr2 = self.cutoff_lr * self.cutoff_lr
            self.two_div_cut = 2.0 / self.cutoff_lr
            if flavor == "PhysNet":
                self.lr_shield = self._simple_lr_shield
            else:
                self.rcutconstant = self.cutoff_lr / (self.cutoff_lr ** 2 + 1.0) ** 1.5
                self.cutconstant = (2 * self.cutoff_lr ** 2 + 1.0) / (self.cutoff_lr** 2 + 1.0) ** 1.5
                self.lr_shield = self._smooth_lr_shield

    def _lr_ordinary(self, Dij: Tensor) -> Tensor:
        return 1.0 / Dij + Dij / self.cutoff_lr2 - self.two_div_cut

    def _shield(self, Dij: Tensor) -> Tensor:
        return torch.sqrt(Dij * Dij + 1.0)

    def _simple_lr_shield(self, Dij: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Dij_shield = self._shield(Dij)
        zeros = torch.zeros_like(Dij)
        condition = Dij < self.cutoff_lr
        return (
            torch.where(condition, self._lr_ordinary(Dij), zeros), 
            torch.where(condition, self._lr_ordinary(Dij_shield), zeros), condition, zeros
        )

    def _smooth_lr_shield(self, Dij: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Dij_shield = self._shield(Dij)
        zeros = torch.zeros_like(Dij)
        condition = Dij < self.cutoff_lr
        return (
            torch.where(condition, self._lr_ordinary(Dij), zeros), 
            torch.where(condition, 1.0 / Dij_shield + Dij * self.rcutconstant - self.cutconstant, zeros),
            condition, zeros
        )

    @staticmethod
    def _erf_potential(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return torch.erf(r / sigma) / r

    @staticmethod
    def _erf_force(r: Tensor, sigma: float) -> Tensor:
        r = r.clamp_min(1e-12)
        return (
            2.0 * r * torch.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)
            - torch.erf(r / sigma)
        ) / (r ** 2)

    def _get_E_ele_a_so3lr(
        self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor
    ) -> Tensor:
        if Qa.dim() > 1:
            Qa = Qa.squeeze(-1)
        r = Dij_lr.reshape(-1)
        qi = Qa[idx_i]
        qj = Qa[idx_j]
        pairwise = self._erf_potential(r, self.sigma)
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            edge = self.pair_kehalf * qi * qj * pairwise / self.dielectric_constant
        else:
            cutoff = float(self.cutoff_lr)
            f = self._switch(r, cutoff, self.cuton)
            shift = self._erf_potential(
                torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma
            )
            force_shift = self._erf_force(
                torch.tensor(cutoff, dtype=r.dtype, device=r.device), self.sigma
            )
            energy_shifted = pairwise - shift
            force_shifted = pairwise - shift - force_shift * (r - cutoff)
            blended = f * energy_shifted + (1.0 - f) * force_shifted
            edge = self.pair_kehalf * qi * qj * blended / self.dielectric_constant
            edge = torch.where(r < cutoff, edge, torch.zeros_like(edge))
        return scatter_sum(edge, idx_i, len(Qa))

    def get_E_ele_a(self, Dij_lr: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        '''
        Compute the atomic electrostatic energy

        .. rubric:: Parameters

        Dij: Float tensor of pair distances, shape [N_pair * batch_size]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        idx_i: Long tensor of the first indices of pairs, shape [N_pair * batch_size]

        idx_j: Long tensor of the second indices of pairs, shape [N_pair * batch_size]

        .. rubric:: Returns

        Ea: Float tensor of atomic electrostatic energy, shape [N * batch_size]
        '''
        if self.flavor == "SO3LR":
            return self._get_E_ele_a_so3lr(Dij_lr, Qa, idx_i, idx_j)

        if Qa.device.type == "cpu" or Qa.dim() > 1:
            fac = self.kehalf * Qa[idx_i] * Qa[idx_j] / self.dielectric_constant
        else:
            fac = self.kehalf * Qa.gather(0, idx_i) * Qa.gather(0, idx_j) / self.dielectric_constant
        switch = self.cutoff_fn(Dij_lr, self.cutoff, self.cuton)
        cswitch = 1 - switch
        view_shape = (-1, 1) if Qa.dim() > 1 else (-1,)
        if self.cutoff_lr is None or self.cutoff_lr <= 0:
            Eele_ordinary = 1.0 / Dij_lr
            Eele_shielded = 1.0 / self._shield(Dij_lr)
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary).view(view_shape)
        else:
            Eele_ordinary, Eele_shielded, condition, zeros = self.lr_shield(Dij_lr)
            # combine shielded and ordinary interactions and apply prefactors
            Eele = fac * (switch * Eele_shielded + cswitch * Eele_ordinary).view(view_shape)
            Eele = torch.where(condition, Eele, zeros)
        return scatter_sum(Eele, idx_i, len(Qa))


def require_uniform_molecular_charge(values: Tensor, batch_seg: Optional[Tensor]) -> None:
    """Reject a per-atom total that is not constant inside one molecule.

    ``values`` is one charge per atom. ``batch_seg is None`` means every atom
    belongs to the same molecule.
    """
    flat = values.reshape(-1)
    n_atoms = int(flat.numel())
    if n_atoms <= 1:
        return
    if batch_seg is None:
        index = torch.zeros(n_atoms, dtype=torch.long, device=flat.device)
    else:
        index = batch_seg.to(device=flat.device).long().reshape(-1)
        if index.numel() != n_atoms:
            raise ValueError(
                f"batch_seg length {index.numel()} != number of charge values {n_atoms}"
            )
    # Adjacent atoms in molecule order suffice, including sparse or
    # interleaved molecule IDs. All intermediate shapes depend only on N.
    labels, order = torch.sort(index)
    ordered = flat.detach()[order]
    finite = torch.isfinite(flat).all()
    nonnegative = labels[0] >= 0
    uniform = ((labels[1:] != labels[:-1]) | (ordered[1:] == ordered[:-1])).all()
    message = ("Q must be one molecular total per molecule. A length-N Q has to "
               "repeat that total on every atom of the molecule.")
    if torch.compiler.is_compiling():
        torch._assert_async(finite, "Q must be finite")
        torch._assert_async(nonnegative, "batch_seg must be non-negative")
        torch._assert_async(uniform, message)
    else:
        if not finite:
            raise ValueError("Q must be finite")
        if not nonnegative:
            raise ValueError("batch_seg must be non-negative")
        if not uniform:
            raise ValueError(message)


def neutral_charge_equilibration(
    charges: Tensor,
    weights: Tensor,
    total_charge: Tensor,
    batch_seg: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Shift each channel so it sums to the molecular total charge.

    ``charges`` and ``weights`` have shape ``[N, C]`` (a 1-D input is treated
    as one channel). ``weights`` are the non-negative redistribution weights;
    TensorNet2 passes the square of the network output. ``total_charge`` is
    the molecular total already written on every atom, shape ``[N]`` or
    ``[N, 1]``, and it must be the same value on every atom of a molecule.
    ``eps`` is added to a positive weight sum, matching TorchMD-Net. A channel
    whose weights sum to zero inside a molecule is shifted uniformly, because
    the published ratio is otherwise undefined and the charges would not move.
    """
    if charges.shape != weights.shape:
        raise ValueError(
            f"charges and weights must share a shape, got {tuple(charges.shape)} "
            f"and {tuple(weights.shape)}"
        )
    squeezed = charges.ndim == 1
    if squeezed:
        charges = charges.unsqueeze(-1)
        weights = weights.unsqueeze(-1)
    n_atoms = int(charges.shape[0])
    if batch_seg is None:
        batch_seg = torch.zeros(n_atoms, dtype=torch.long, device=charges.device)
    elif batch_seg.shape[0] != n_atoms:
        raise ValueError(
            f"batch_seg length {batch_seg.shape[0]} != number of atoms {n_atoms}"
        )
    target = total_charge.to(device=charges.device, dtype=charges.dtype).reshape(-1)
    if target.numel() == 1:
        target = target.expand(n_atoms)
    elif target.numel() != n_atoms:
        raise ValueError(
            f"total_charge must have 1 or {n_atoms} values, got {target.numel()}"
        )
    require_uniform_molecular_charge(target, batch_seg)
    counts = scatter_sum(torch.ones(n_atoms, 1, dtype=charges.dtype, device=charges.device), batch_seg)
    raw_sum = scatter_sum(weights, batch_seg)
    dead = raw_sum <= 0
    uniform = torch.ones_like(weights) / counts[batch_seg].clamp_min(1)
    effective = torch.where(dead[batch_seg], uniform, weights)
    summed_weights = scatter_sum(effective, batch_seg)
    summed_weights = summed_weights + torch.where(
        dead,
        torch.zeros_like(summed_weights),
        summed_weights.new_full((), float(eps)),
    )
    summed_charges = scatter_sum(charges, batch_seg)
    delta = target.unsqueeze(-1) - summed_charges[batch_seg]
    updated = charges + effective / summed_weights[batch_seg] * delta
    return updated.squeeze(-1) if squeezed else updated


def aimnet_charge_damping(distance: Tensor, damping_cutoff: float = 4.6) -> Tensor:
    """AIMNet2 short-range factor: 0 at ``r = 0``, about 1 for ``r >= cutoff``.

    This removes the Coulomb singularity. It is not the message-passing cutoff.
    The constant ``exp(-1)`` is the AIMNet normalization.
    """
    if float(damping_cutoff) <= 0:
        raise ValueError(f"damping_cutoff must be positive, got {damping_cutoff}")
    reduced = (distance / float(damping_cutoff)).clamp(0, 1.0 - 1e-6)
    screen = torch.exp(-1.0 / (1.0 - reduced.square())) / math.exp(-1.0)
    return 1.0 - screen


def coulomb_kernel(
    distance: Tensor,
    cutoff: Optional[float] = None,
    epsilon_solvent: float = 78.3,
    reaction_field: bool = False,
) -> Tensor:
    """``1/r``, or the OpenMM reaction-field kernel when ``reaction_field`` is set."""
    inverse = 1.0 / distance.clamp_min(1e-12)
    if not reaction_field:
        return inverse
    if cutoff is None or float(cutoff) <= 0:
        raise ValueError("reaction_field requires a positive cutoff")
    if float(epsilon_solvent) <= 0:
        raise ValueError(f"epsilon_solvent must be positive, got {epsilon_solvent}")
    cutoff_f = float(cutoff)
    epsilon = float(epsilon_solvent)
    k_rf = (1.0 / cutoff_f**3) * (epsilon - 1.0) / (2.0 * epsilon + 1.0)
    c_rf = (1.0 / cutoff_f) * (3.0 * epsilon) / (2.0 * epsilon + 1.0)
    return inverse + k_rf * distance.square() - c_rf


def expand_coulomb_weights(
    layer_weights: Optional[Sequence[float]],
    n_channels: int,
    like: Tensor,
) -> Tensor:
    """Broadcast per-stage weights across charge channels and normalize.

    ``None`` gives every channel the same weight. A list whose length equals
    the channel count is used as-is. A shorter list is repeated across
    contiguous stages when it divides ``n_channels``. The result sums to 1.
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    if layer_weights is None:
        weights = like.new_ones(n_channels)
    else:
        values = [float(value) for value in layer_weights]
        if len(values) < 1:
            raise ValueError("layer_weights must contain at least one value")
        if n_channels % len(values) != 0:
            raise ValueError(
                f"layer_weights has length {len(values)}, which does not divide "
                f"{n_channels} charge channels. Pass one weight per stage or one "
                "weight per channel."
            )
        stage = like.new_tensor(values)
        weights = stage.repeat_interleave(n_channels // len(values))
    total = weights.sum()
    if torch.compiler.is_compiling():
        torch._assert_async(total != 0, "layer_weights must not sum to zero")
    elif total == 0:
        raise ValueError("layer_weights must not sum to zero")
    return weights / total


def intramolecular_pair_indices(positions: Tensor, batch_seg: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """Distinct atom pairs in the same molecule, in upper-triangle order."""
    n_atoms = positions.shape[0]
    left, right = torch.triu_indices(n_atoms, n_atoms, offset=1, device=positions.device)
    if batch_seg is not None:
        same_molecule = torch.where(batch_seg[left] == batch_seg[right])[0]
        left, right = left[same_molecule], right[same_molecule]
    return left, right


def intramolecular_upper_pairs(
    positions: Tensor, batch_seg: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Upper-triangle pairs that stay inside one molecule. No periodic images."""
    left, right = intramolecular_pair_indices(positions, batch_seg)
    if positions.shape[0] <= 1:
        return left, right, positions.new_zeros(0)
    distance = torch.linalg.vector_norm(positions[left] - positions[right], dim=-1)
    return left, right, distance


class WeightedCoulombEnergyLayer(BaseFFLayer):
    """Multi-channel Coulomb from AIMNet2 / TensorNet2.

    ``charge_channels`` is ``[N, C]``. Each channel contributes ``q_i q_j``
    times a damped kernel; those energies are then averaged with
    ``layer_weights``. The prefactor is ``kehalf = 0.5 * Bohr_in_R *
    Hartree_in_E``. ``pairs: all_to_all`` builds one undirected pair per
    molecule from ``Ra`` and adds that energy to both atoms, which is the
    vacuum AceFF path (no periodic images). ``pairs: directed`` reads
    ``idx_i``, ``idx_j``, and ``Dij_lr`` and adds ``kehalf`` only onto
    ``idx_i``, so a bidirectional neighbor list totals the full Coulomb.
    A positive ``cutoff`` drops longer pairs. ``reaction_field`` replaces
    ``1/r`` with the OpenMM reaction-field kernel and needs that cutoff.
    ``EnergyReduce`` adds the atomic term because the field is ``E_coul_a``.
    """

    _force_pair_fields = ("Ra", "batch_seg")

    def _prepare_force_pairs(self, Ra, batch_seg):
        if self.pairs == "all_to_all" and self.cutoff is None:
            return intramolecular_pair_indices(Ra, batch_seg)
        return None

    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        damping_cutoff: float = 4.6,
        cutoff: Optional[float] = None,
        reaction_field: bool = False,
        epsilon_solvent: float = 78.3,
        pairs: Literal["all_to_all", "directed"] = "all_to_all",
        Bohr_in_R: float = 0.5291772105638411,
        Hartree_in_E: float = 1.0,
    ) -> None:
        if pairs not in {"all_to_all", "directed"}:
            raise ValueError("pairs must be 'all_to_all' or 'directed', got {0}".format(pairs))
        if float(damping_cutoff) <= 0:
            raise ValueError(f"damping_cutoff must be positive, got {damping_cutoff}")
        if cutoff is not None and float(cutoff) <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if reaction_field and cutoff is None:
            raise ValueError("reaction_field requires cutoff")
        if float(Bohr_in_R) <= 0 or float(Hartree_in_E) <= 0:
            raise ValueError("Bohr_in_R and Hartree_in_E must be positive")
        super().__init__(
            input_fields={"Ra", "charge_channels", "batch_seg", "idx_i", "idx_j", "Dij_lr"},
            output_fields={"E_coul_a"},
        )
        self.layer_weights = None if layer_weights is None else [float(value) for value in layer_weights]
        self.damping_cutoff = float(damping_cutoff)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.reaction_field = bool(reaction_field)
        self.epsilon_solvent = float(epsilon_solvent)
        self.pairs = pairs
        self.kehalf = 0.5 * float(Bohr_in_R) * float(Hartree_in_E)

    def get_E_coul_a(
        self,
        Ra: Tensor,
        charge_channels: Tensor,
        batch_seg: Optional[Tensor] = None,
        idx_i: Optional[Tensor] = None,
        idx_j: Optional[Tensor] = None,
        Dij_lr: Optional[Tensor] = None,
        _pair_indices: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        if charge_channels.ndim == 1:
            charge_channels = charge_channels.unsqueeze(-1)
        if charge_channels.ndim != 2 or charge_channels.shape[0] != Ra.shape[0]:
            raise ValueError(
                "charge_channels must have shape [N, C] aligned with Ra, "
                f"got {tuple(charge_channels.shape)} and {tuple(Ra.shape)}"
            )
        n_atoms = int(Ra.shape[0])
        energy = Ra.new_zeros(n_atoms)
        if n_atoms == 0:
            return energy
        weights = expand_coulomb_weights(
            self.layer_weights, int(charge_channels.shape[-1]), Ra
        )
        if self.pairs == "all_to_all":
            if _pair_indices is None:
                left, right, distance = intramolecular_upper_pairs(Ra, batch_seg)
            else:
                left, right = _pair_indices
                distance = torch.linalg.vector_norm(Ra[left] - Ra[right], dim=-1)
            add_both = True
        else:
            if idx_i is None or idx_j is None or Dij_lr is None:
                raise ValueError("pairs: directed requires idx_i, idx_j, and Dij_lr")
            left = idx_i.reshape(-1).long()
            right = idx_j.reshape(-1).long()
            distance = Dij_lr.reshape(-1).to(dtype=Ra.dtype)
            add_both = False
        if self.cutoff is not None and distance.numel():
            keep = distance < self.cutoff
            left, right, distance = left[keep], right[keep], distance[keep]
        if distance.numel() == 0:
            return energy
        damping = aimnet_charge_damping(distance, self.damping_cutoff)
        kernel = coulomb_kernel(
            distance, self.cutoff, self.epsilon_solvent, self.reaction_field
        )
        pair = self.kehalf * damping * kernel
        channel = pair.unsqueeze(-1) * charge_channels[left] * charge_channels[right]
        edge = (channel * weights.unsqueeze(0)).sum(dim=-1)
        energy = energy.index_add(0, left, edge)
        if add_both:
            energy = energy.index_add(0, right, edge)
        return energy


class EwaldElectrostaticEnergyLayer(BaseFFLayer):
    def __init__(
        self,
        cutoff_real: float,
        k_cutoff: float,
        alpha: Optional[float] = None,
        Bohr_in_R: float = 0.5291772108,
        Hartree_in_E: float = 1,
        dielectric_constant: float = 1,
    ) -> None:
        r"""
        Point-charge Ewald-summation electrostatic energy for periodic (PBC) systems:
        an erfc-damped real-space Coulomb sum over the cutoff-limited pair list, plus
        a Gaussian-screened reciprocal-space sum and a self-energy correction. A
        drop-in alternative to :class:`ElectrostaticEnergyLayer` for periodic
        datasets (same ``E_ele_a`` output field; swap the YAML layer name).

        Real space, self-energy correction and the atomic bookkeeping (spreading the
        per-graph reciprocal energy over atoms weighted by q^2) follow SpookyNet's
        Ewald summation [2], which assumes an orthorhombic cell. Reciprocal space is
        generalized here to an arbitrary triclinic cell using the reciprocal-lattice
        construction (``G = 2*pi*cell^-1^T``) of LES [3]. Each graph in a batch may
        have its own cell, so the reciprocal sum loops once per graph (same reason
        LES's own Ewald module loops per graph); the real-space sum needs no such
        loop since it is a plain scatter-sum over the (already batch-global)
        ``idx_i``/``idx_j`` pair list, like every other layer in this module.

        Params
        -----
        cutoff_real: real-space damping cutoff. Pairs with ``Dij_lr >= cutoff_real``
            are dropped from the real-space sum (their contribution is captured by
            the reciprocal sum instead). Should not exceed the search radius used to
            build ``idx_i``/``idx_j`` (Datahub ``neighbor_list_cutoff``).

        k_cutoff: reciprocal-space cutoff radius (1/Angstrom).

        alpha: real/reciprocal-space split (Ewald damping parameter). Defaults to
            SpookyNet's heuristic ``4 / cutoff_real + 1e-3``.

        References
        -----
        [1] P. P. Ewald, Ann. Phys. 1921, 369, 253-287.
        [2] O. T. Unke et al., Nat. Commun. 2021, 12, 7273 (SpookyNet),
            https://github.com/OUnke/SpookyNet
        [3] B. Cheng et al., "Latent Ewald Summation" (LES),
            https://github.com/ChengUCB/les
        """
        super().__init__(
            input_fields={"Ra", "Qa", "idx_i", "idx_j", "Dij_lr", "cell", "batch_seg"},
            output_fields={"E_ele_a"},
        )
        self.cutoff_real = float(cutoff_real)
        self.k_cutoff = float(k_cutoff)
        self.dielectric_constant = dielectric_constant
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E
        self.ke = Bohr_in_R * Hartree_in_E
        if alpha is None:
            alpha = 4.0 / self.cutoff_real + 1e-3
        self.alpha = float(alpha)
        self.alpha2 = self.alpha * self.alpha
        if self.alpha * self.cutoff_real < 4.0:  # erfc(4.0) ~ 1e-8
            import warnings
            warnings.warn(
                f"EwaldElectrostaticEnergy: alpha={self.alpha} may be too small "
                f"for cutoff_real={self.cutoff_real}; consider alpha >= "
                f"{4.0 / self.cutoff_real:.4g}, or reciprocal-space error may leak "
                "into the damped real-space term."
            )

    def _real_space(self, Qa: Tensor, idx_i: Tensor, idx_j: Tensor, Dij_lr: Tensor) -> Tensor:
        r = Dij_lr.clamp_min(1e-12)
        fac = self.kehalf * Qa[idx_i] * Qa[idx_j] / self.dielectric_constant
        pairwise = fac * torch.erfc(self.alpha * r) / r
        pairwise = torch.where(Dij_lr < self.cutoff_real, pairwise, torch.zeros_like(pairwise))
        return segment_sum_coo(pairwise, idx_i, dim_size=Qa.shape[0])

    def _reciprocal_space(self, Ra: Tensor, Qa: Tensor, cell: Tensor, batch_seg: Tensor) -> Tensor:
        device, dtype = Ra.device, Ra.dtype
        E_recip_a = torch.zeros_like(Qa)
        eps = 1e-8
        num_graphs = int(batch_seg.max().item()) + 1
        for g in range(num_graphs):
            mask = batch_seg == g
            if not torch.any(mask):
                continue
            cell_g = cell[g].to(dtype)
            volume = torch.det(cell_g)
            if volume.abs() < 1e-6:
                continue  # degenerate/absent cell: this graph is not periodic
            r = Ra[mask]
            q = Qa[mask]
            G = 2.0 * math.pi * torch.linalg.inv(cell_g).T
            recip_norms = torch.linalg.norm(G, dim=1).clamp_min(1e-12)
            n_max = torch.clamp(torch.ceil(self.k_cutoff / recip_norms), min=1).long().tolist()
            n1 = torch.arange(-n_max[0], n_max[0] + 1, device=device, dtype=dtype)
            n2 = torch.arange(-n_max[1], n_max[1] + 1, device=device, dtype=dtype)
            n3 = torch.arange(-n_max[2], n_max[2] + 1, device=device, dtype=dtype)
            kvec = torch.cartesian_prod(n1, n2, n3) @ G
            k2 = (kvec * kvec).sum(-1)
            keep = (k2 > 1e-12) & (k2 <= self.k_cutoff ** 2)
            kvec, k2 = kvec[keep], k2[keep]
            if kvec.shape[0] == 0:
                continue
            kr = r @ kvec.T  # [n_atoms_g, M]
            S_real = (q.unsqueeze(-1) * torch.cos(kr)).sum(0)  # [M]
            S_imag = (q.unsqueeze(-1) * torch.sin(kr)).sum(0)
            kfac = torch.exp(-0.25 * k2 / self.alpha2) / k2
            e_total = (2.0 * math.pi / volume) * torch.sum(kfac * (S_real ** 2 + S_imag ** 2))
            # Spread the per-graph reciprocal energy over its atoms, weighted by
            # q^2 (SpookyNet's atomic bookkeeping trick), so the layer's output
            # stays atomic like every other energy layer feeding EnergyReduce.
            w = q * q + eps
            E_recip_a[mask] = (w / w.sum()) * e_total
        e_self = self.alpha / math.sqrt(math.pi) * Qa * Qa
        return self.ke / self.dielectric_constant * (E_recip_a - e_self)

    def get_E_ele_a(
        self, Ra: Tensor, Qa: Tensor, idx_i: Tensor, idx_j: Tensor, Dij_lr: Tensor,
        cell: Optional[Tensor] = None, batch_seg: Optional[Tensor] = None,
    ) -> Tensor:
        '''
        Compute the atomic Ewald electrostatic energy.

        Params
        -----
        Ra: Float tensor of atom positions, shape [N * batch_size, 3]

        Qa: Float tensor of atomic charges, shape [N * batch_size]

        idx_i, idx_j: Long tensors of pair indices, shape [N_pair * batch_size]

        Dij_lr: Float tensor of pair distances, shape [N_pair * batch_size]

        cell: Float tensor of lattice vectors, shape [batch_size, 3, 3]

        batch_seg: Long tensor of batch indices, shape [N * batch_size]

        Returns
        -----
        E_ele_a: Float tensor of atomic electrostatic energy, shape [N * batch_size]
        '''
        if Qa.dim() > 1:
            Qa = Qa.squeeze(-1)
        if cell is None:
            raise ValueError(
                "EwaldElectrostaticEnergy requires a 'cell' input (periodic "
                "systems); for non-periodic data use ElectrostaticEnergy instead."
            )
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        if batch_seg is None:
            batch_seg = torch.zeros(Qa.shape[0], dtype=torch.long, device=Qa.device)
        E_real = self._real_space(Qa, idx_i, idx_j, Dij_lr)
        E_recip = self._reciprocal_space(Ra, Qa, cell, batch_seg)
        return E_real + E_recip


class AtomicCharge2DipoleLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"Qa", "Ra", "batch_seg", "N"}, output_fields={"M2"})

    def get_M2(self, Qa: Tensor, Ra: Tensor, batch_seg: Optional[Tensor]=None, N=None) -> Tensor:
        if batch_seg is None:
            batch_seg = torch.zeros(Qa.shape[0], dtype=torch.long, device=Qa.device)
        Pa = Qa.unsqueeze(1) * Ra.view((-1, 3, 1) if Qa.dim() > 1 else (-1, 3))
        return scatter_sum(Pa, batch_seg, batch_size_from_counts(N))
