"""PySCF natural atomic orbital (NAO) Mulliken population on a converged or partial SCF density."""
from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from pyscf import scf
from pyscf.lo import orth
from pyscf.scf import hf

if TYPE_CHECKING:
    from pyscf.gto import Mole
    from pyscf.scf.hf import SCF


def _overlap(mf: SCF) -> np.ndarray:
    return mf.get_ovlp()


def _nao_ao_populations_per_atom(mol: Mole, mf: SCF, dm: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Sum NAO-basis Mulliken AO populations per atom for one spin density matrix."""
    orth_coeff = orth.orth_ao(mf, "nao", s=s)
    c_inv = np.dot(orth_coeff.conj().T, s)
    dm_t = reduce(np.dot, (c_inv, dm, c_inv.T.conj()))
    pop, _ = hf.mulliken_pop(mol, dm_t, np.eye(orth_coeff.shape[0]), verbose=0)
    nelec = np.zeros(mol.natm, dtype=float)
    for i, label in enumerate(mol.ao_labels(fmt=None)):
        nelec[label[0]] += float(pop[i])
    return nelec


def nao_Qa_Sa_from_mf(mol: Mole, mf: SCF) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom charge-like ``Qa`` and spin-proxy ``Sa`` from NAO populations.

    Convention (aligned with OMol ``nbo_charges`` / ``nbo_spins`` checks):

    - ``Qa``: formal atomic charge, ``Z_A - (n_alpha + n_beta)`` in the NAO basis.
    - ``Sa``: ``n_alpha - n_beta`` per atom (closed-shell singlet → zeros).
    - ``sum(Qa) ≈ mol.charge``, ``sum(Sa) ≈ mol.spin`` (unpaired electrons).
    """
    s = _overlap(mf)
    nuclear = mol.atom_charges()

    if isinstance(mf, scf.uhf.UHF):
        dm_alpha, dm_beta = mf.make_rdm1()
        n_alpha = _nao_ao_populations_per_atom(mol, mf, dm_alpha, s)
        n_beta = _nao_ao_populations_per_atom(mol, mf, dm_beta, s)
        qa = nuclear - (n_alpha + n_beta)
        sa = n_alpha - n_beta
        return qa, sa

    if isinstance(mf, scf.rohf.ROHF):
        dm = mf.make_rdm1()
        if isinstance(dm, (list, tuple)):
            dm = dm[0] + dm[1]
    else:
        dm = mf.make_rdm1()

    n_total = _nao_ao_populations_per_atom(mol, mf, dm, s)
    qa = nuclear - n_total
    return qa, np.zeros(mol.natm, dtype=float)
