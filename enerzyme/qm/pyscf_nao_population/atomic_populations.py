"""ASE + GPU4PySCF DFT + PySCF NAO population: per-atom ``Qa`` / ``Sa`` priors.

``atoms.info`` uses:

- ``charge``: total molecular charge (e), same convention as OMol / xTB helpers.
- ``spin``: spin multiplicity M (2S+1), integer ≥ 1 (default 1).

All DFT settings (``xc``, ``basis``, ``conv_tol``, ``density_fit``, ``use_gpu``) are
caller-supplied; there are no hidden functional/basis defaults in this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr
from pyscf import gto

from .gpu_dft import run_dft
from .nao_population import nao_Qa_Sa_from_mf

if TYPE_CHECKING:
    from ase import Atoms
    from pyscf.gto import Mole


def _read_charge_spin(atoms: Atoms) -> tuple[int, int]:
    charge = float(atoms.info.get("charge", 0.0))
    mult = int(atoms.info.get("spin", 1))
    if mult < 1:
        raise ValueError("atoms.info['spin'] (multiplicity) must be >= 1")
    return int(round(charge)), mult


def atoms_to_mol(
    atoms: Atoms,
    *,
    basis: str,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> Mole:
    """Build a PySCF molecule from ASE atoms."""
    if charge is None or multiplicity is None:
        q, m = _read_charge_spin(atoms)
        charge = q if charge is None else charge
        multiplicity = m if multiplicity is None else multiplicity

    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(wrap=False), dtype=float) / Bohr
    atom_lines = [
        f"{sym} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}"
        for sym, pos in zip(symbols, positions, strict=True)
    ]
    spin = max(0, int(multiplicity) - 1)
    return gto.M(
        atom="; ".join(atom_lines),
        basis=basis,
        charge=int(charge),
        spin=spin,
        verbose=0,
    )


def atomic_Q_and_S_from_pyscf_nao(
    atoms: Atoms,
    max_scf_iter: int,
    *,
    xc: str,
    basis: str,
    conv_tol: float,
    density_fit: bool,
    use_gpu: bool,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run finite-step GPU4PySCF (or CPU) DFT and return NAO-based ``Qa``, ``Sa``."""
    mol = atoms_to_mol(atoms, basis=basis)
    mf = run_dft(
        mol,
        xc=xc,
        max_cycle=int(max_scf_iter),
        conv_tol=float(conv_tol),
        density_fit=bool(density_fit),
        verbose=int(verbose),
        use_gpu=bool(use_gpu),
    )
    qa, sa = nao_Qa_Sa_from_mf(mol, mf)
    return np.asarray(qa, dtype=float).reshape(-1), np.asarray(sa, dtype=float).reshape(-1)
