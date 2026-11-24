"""Embedded-atom densities built from Cartesian Gaussian orbitals.

Zhang, Xia, and Jiang, J. Chem. Phys. 156, 114801 (2022), Eqs. 1–4, and the
contraction in https://github.com/zhangylch/REANN ``src/density.py``. The
printed cutoff ``[cos(pi r / rc) + 1]^2`` is four times
:func:`enerzyme.models.mp.basis.envelope.cosine_transition` squared. The repository uses
the squared cosine, which is 1 at the origin, and this module follows that.
Higher angular channels are ordered Cartesian products of the raw displacement,
not unique monomials and not divided by the multinomial ``l! / (lx! ly! lz!)``.
The cutoff window is applied once and then multiplied by those products, which
is the released loop rather than an extra factor of ``fc^l``.

EquiREANN (Wu et al., J. Phys. Chem. A 2024) calls
:func:`pair_gaussian_orbitals`, :func:`aggregate_weighted_orbitals`, and
:func:`invariant_orbital_density` again after adding the neighbor orbital.
That addition lives with the EquiREANN Core, not in this invariant density.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..mp.basis.cartesian import cartesian_powers
from ..mp.basis.envelope import cosine_transition


def default_n_orbit(nwave: int, lmax: int) -> int:
    """Descriptor width ``nwave * (nwave + 1) / 2 * (lmax + 1)``.

    The repository increments the user's ``nipsin`` by one before this product,
    so ``nipsin`` there is this ``lmax``.
    """
    if int(nwave) < 1:
        raise ValueError(f"nwave must be >= 1, got {nwave}")
    if int(lmax) < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    return int(nwave) * (int(nwave) + 1) // 2 * (int(lmax) + 1)


def angular_channel_count(lmax: int) -> int:
    """Number of ordered Cartesian channels, ``sum_{l=0}^{L} 3^l``."""
    if int(lmax) < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    return (3 ** (int(lmax) + 1) - 1) // 2


def angular_momentum_index(lmax: int, device: torch.device | None = None) -> Tensor:
    """Shell index of every Cartesian channel. Shape ``[n_angular]``."""
    chunks = []
    count = 1
    for shell in range(int(lmax) + 1):
        chunks.append(torch.full((count,), shell, dtype=torch.long, device=device))
        count *= 3
    return torch.cat(chunks)


def squared_cosine_window(distance: Tensor, cutoff: float) -> Tensor:
    """Repository cutoff ``[0.5 * cos(pi r / rc) + 0.5]^2``."""
    return cosine_transition(distance, float(cutoff)).square()


def cartesian_angular(vectors: Tensor, window: Tensor, lmax: int) -> Tensor:
    """Window, then ``window * r``, then ``window * r ⊗ r``, and so on.

    ``vectors`` is ``[E, 3]`` and points from the receiver to the neighbor.
    The return is ``[n_angular, E]`` in the same channel order as the released
    recursive product.
    """
    n_angular = angular_channel_count(lmax)
    n_edges = vectors.shape[0]
    if n_edges == 0:
        return vectors.new_zeros((n_angular, 0))
    powers = cartesian_powers(vectors, lmax, weight=window)
    return torch.cat([part.reshape(n_edges, -1) for part in powers.values()], dim=1).transpose(0, 1)


def pair_gaussian_orbitals(
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    distance: Tensor,
    species: Tensor,
    rs: Tensor,
    inta: Tensor,
    lmax: int,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Geometric orbitals ``[E, n_angular, nwave]`` and the squared-cosine window.

    The window is the factor already baked into every angular channel. EquiREANN
    multiplies the passed neighbor orbital by this same tensor. ``vij`` points
    from the receiver to the neighbor. Gaussian centers are read for the neighbor.
    """
    if int(lmax) < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    nwave = int(rs.shape[-1])
    n_angular = angular_channel_count(int(lmax))
    if idx_i.numel() == 0:
        empty = vij.new_zeros((0, n_angular, nwave))
        return empty, vij.new_zeros((0,))
    if idx_i.shape != idx_j.shape:
        raise ValueError(
            f"idx_i and idx_j must have the same shape, got {tuple(idx_i.shape)} "
            f"and {tuple(idx_j.shape)}"
        )
    if vij.ndim != 2 or vij.shape[-1] != 3 or vij.shape[0] != idx_i.shape[0]:
        raise ValueError(
            f"vij must have shape [E, 3] with E={idx_i.shape[0]}, got {tuple(vij.shape)}"
        )
    distance = distance.reshape(-1)
    if distance.shape[0] != idx_i.shape[0]:
        raise ValueError(
            f"distance has length {distance.shape[0]} but there are {idx_i.shape[0]} edges"
        )
    if rs.shape[-1] != inta.shape[-1]:
        raise ValueError(
            f"rs and inta must share nwave, got {tuple(rs.shape)} and {tuple(inta.shape)}"
        )
    neighbor = species.reshape(-1).long().index_select(0, idx_j.long())
    if neighbor.numel() and (int(neighbor.min()) < 0 or int(neighbor.max()) >= rs.shape[0]):
        raise ValueError(
            f"neighbor atomic numbers must lie in 0..{rs.shape[0] - 1}, "
            f"got {int(neighbor.min())}..{int(neighbor.max())}"
        )
    dtype = vij.dtype
    centers = rs.index_select(0, neighbor).to(dtype=dtype)
    exponents = inta.index_select(0, neighbor).to(dtype=dtype)
    radial = torch.exp(exponents * (distance.to(dtype=dtype).unsqueeze(-1) - centers).square())
    window = squared_cosine_window(distance, cutoff).to(dtype=dtype)
    angular = cartesian_angular(vij.to(dtype=dtype), window, int(lmax))
    orbital = angular.transpose(0, 1).unsqueeze(-1) * radial.unsqueeze(1)
    return orbital, window


def aggregate_weighted_orbitals(
    pair: Tensor,
    coeff: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    n_atoms: int,
) -> Tensor:
    """Sum ``c_j * pair_ij`` onto the receiver. Shape ``[N, n_angular, nwave]``.

    This is ``ψ`` in EquiREANN. The neighbor coefficient is applied before the
    sum, which is the released contraction, and the square happens later.
    """
    if pair.ndim != 3:
        raise ValueError(f"pair must have shape [E, n_angular, nwave], got {tuple(pair.shape)}")
    if coeff.ndim != 2 or coeff.shape[0] != int(n_atoms) or coeff.shape[-1] != pair.shape[-1]:
        raise ValueError(
            f"coeff must have shape [{n_atoms}, {pair.shape[-1]}], got {tuple(coeff.shape)}"
        )
    if pair.shape[0] != idx_i.shape[0]:
        raise ValueError(
            f"pair has {pair.shape[0]} edges but idx_i has {idx_i.shape[0]}"
        )
    weighted = pair * coeff.to(dtype=pair.dtype).index_select(0, idx_j.long()).unsqueeze(1)
    accumulated = weighted.new_zeros((int(n_atoms), pair.shape[1], pair.shape[2]))
    if weighted.shape[0] == 0:
        return accumulated
    return accumulated.index_add(0, idx_i.long(), weighted)


def invariant_orbital_density(
    accumulated: Tensor,
    hyper: Tensor,
    l_of_channel: Tensor,
) -> Tensor:
    """Square of the shell-shared contraction. Shape ``[N, n_orbit]``.

    ``hyper`` has shape ``[lmax + 1, nwave, n_orbit]``. Every Cartesian channel
    of one shell uses the same matrix, so the sum of squares is invariant.
    """
    if accumulated.ndim != 3:
        raise ValueError(
            f"accumulated orbitals must have shape [N, n_angular, nwave], got {tuple(accumulated.shape)}"
        )
    if hyper.ndim != 3:
        raise ValueError(
            f"hyper must have shape [lmax + 1, nwave, n_orbit], got {tuple(hyper.shape)}"
        )
    if hyper.shape[1] != accumulated.shape[-1]:
        raise ValueError(
            f"hyper nwave {hyper.shape[1]} does not match orbitals {accumulated.shape[-1]}"
        )
    n_shell = int(hyper.shape[0])
    if n_shell < 1 or angular_channel_count(n_shell - 1) != int(l_of_channel.numel()):
        raise ValueError(
            "hyper and l_of_channel do not describe the same angular shells: "
            f"hyper {tuple(hyper.shape)}, channels {int(l_of_channel.numel())}"
        )
    if accumulated.shape[1] != int(l_of_channel.numel()):
        raise ValueError(
            f"orbitals have {accumulated.shape[1]} angular channels, "
            f"l_of_channel has {int(l_of_channel.numel())}"
        )
    shell = hyper.to(dtype=accumulated.dtype, device=accumulated.device).index_select(
        0, l_of_channel.to(device=accumulated.device)
    )
    mixed = torch.einsum("naw,awo->nao", accumulated, shell)
    return mixed.square().sum(dim=1)


def embedded_atom_density(
    coeff: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    distance: Tensor,
    species: Tensor,
    rs: Tensor,
    inta: Tensor,
    hyper: Tensor,
    l_of_channel: Tensor,
    cutoff: float,
) -> Tensor:
    """Invariant density ``[N, n_orbit]`` from orbital coefficients ``[N, nwave]``.

    ``idx_i`` is the receiver and ``idx_j`` is the neighbor. ``species`` is the
    atomic number of every atom; Gaussian centers ``rs`` and exponents ``inta``
    are rows of that table and are read for the neighbor. ``hyper`` has shape
    ``[lmax + 1, nwave, n_orbit]`` and is shared by every Cartesian component
    of one shell, which keeps the sum of squares rotationally invariant.
    Amplitudes are summed over neighbors before the square.
    """
    n_atoms, nwave = coeff.shape
    n_orbit = hyper.shape[-1]
    lmax = int(hyper.shape[0]) - 1
    if lmax < 0 or angular_channel_count(lmax) != int(l_of_channel.numel()):
        raise ValueError(
            "hyper and l_of_channel do not describe the same angular shells: "
            f"hyper {tuple(hyper.shape)}, channels {int(l_of_channel.numel())}"
        )
    if hyper.shape[1] != nwave or rs.shape[-1] != nwave or inta.shape[-1] != nwave:
        raise ValueError(
            f"nwave mismatch: coeff {nwave}, hyper {tuple(hyper.shape)}, rs {tuple(rs.shape)}"
        )
    if idx_i.numel() == 0:
        return coeff.new_zeros((n_atoms, n_orbit))
    pair, _window = pair_gaussian_orbitals(
        idx_i, idx_j, vij, distance, species, rs, inta, lmax, cutoff
    )
    accumulated = aggregate_weighted_orbitals(pair, coeff, idx_i, idx_j, n_atoms)
    return invariant_orbital_density(accumulated, hyper, l_of_channel)
