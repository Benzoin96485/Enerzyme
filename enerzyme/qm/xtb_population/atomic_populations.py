"""ASE + tblite xtbml: per-atom Mulliken-type charge Q_a and spin proxy S_a from ``atoms.info``.

``atoms.info`` uses:

- ``charge``: total molecular charge (e).
- ``spin``: spin multiplicity M (2S+1), integer ≥ 1 (default 1 if missing).

xTB uses ``GFN2-xTB``. For M ≠ 1, ``spin-polarization`` is added so xtbml exposes
``q_A_alpha`` / ``q_A_beta``; for M = 1, closed-shell ``q_A`` only.

When ``max_scf_iter`` is small (e.g. 1), SCC may not converge. tblite then raises
``TBLiteRuntimeError``, but the last-cycle ``Result`` (including xtbml
``post-processing-dict``) is still read—this is intentional for one-shot SCC workflows.

Returns:

- **Q_a**: ``q_A_alpha + q_A_beta`` when split exists, else merged ``q_A``.
- **S_a**: ``q_A_alpha - q_A_beta`` when split exists, else zeros (singlet / merged channel).

Requires: ``ase``, ``tblite`` with xtbml (see ``xtbml_charges.py``). Use
:func:`enerzyme.qm.xtb_population.deps.check_xtbml_dependencies` before calling in contexts
where optional deps may be missing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np
from tblite.exceptions import TBLiteRuntimeError
from tblite.interface import Calculator, Result

from .xtbml_charges import register_xtbml

if TYPE_CHECKING:
    from ase import Atoms


def _atoms_to_tblite_inputs(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    from ase.units import Bohr

    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    pos = np.asarray(atoms.get_positions(wrap=False), dtype=float) / Bohr
    return z, pos


def _read_charge_spin(atoms: Atoms) -> tuple[float, int]:
    info = atoms.info
    charge = float(info.get("charge", 0.0))
    mult = int(info.get("spin", 1))
    if mult < 1:
        raise ValueError("atoms.info['spin'] (multiplicity) must be >= 1")
    return charge, mult


def _uhf_from_multiplicity(mult: int) -> int:
    """tblite ``uhf`` = number of unpaired electrons for high-spin open shell (M−1)."""
    return max(0, mult - 1)


def _post_processing_dict(res: Result) -> Mapping[str, Any]:
    raw = res.get("post-processing-dict")
    if not isinstance(raw, Mapping):
        raise TypeError("Result has no post-processing-dict mapping")
    return cast(Mapping[str, Any], raw)


def atomic_Q_and_S_from_xtbml(
    atoms: Atoms,
    max_scf_iter: int = 1,
    *,
    spin_polarization_gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GFN2-xTB + xtbml and return per-atom Q_a and S_a (see module docstring)."""
    numbers, positions = _atoms_to_tblite_inputs(atoms)
    charge, multiplicity = _read_charge_spin(atoms)
    uhf = _uhf_from_multiplicity(multiplicity)

    calc = Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=uhf)
    calc.set("max-iter", int(max_scf_iter))
    calc.set("verbosity", 0)

    # Empirical spin polarization: forces two-spin xtbml density when M ≠ 1
    if multiplicity != 1:
        calc.add("spin-polarization", float(spin_polarization_gamma))

    register_xtbml(calc)
    res = Result()
    try:
        calc.singlepoint(res)
    except TBLiteRuntimeError:
        # Last SCC cycle is still stored on ``res`` (e.g. SCF not converged in 1 cycle).
        pass

    n = int(res.get("natoms"))
    pp = _post_processing_dict(res)

    q_alpha = (
        np.asarray(pp["q_A_alpha"], dtype=float).reshape(-1)
        if "q_A_alpha" in pp
        else None
    )
    q_beta = (
        np.asarray(pp["q_A_beta"], dtype=float).reshape(-1)
        if "q_A_beta" in pp
        else None
    )
    q_merged = (
        np.asarray(pp["q_A"], dtype=float).reshape(-1) if "q_A" in pp else None
    )

    if q_alpha is not None and q_beta is not None:
        if q_alpha.size != n or q_beta.size != n:
            raise ValueError("q_A_alpha / q_A_beta length mismatch with natoms")
        qa = q_alpha + q_beta
        sa = q_alpha - q_beta
        return qa, sa

    if multiplicity != 1:
        raise RuntimeError(
            "Expected xtbml spin-split keys q_A_alpha and q_A_beta for multiplicity != 1; "
            f"got keys containing q_A: {[k for k in pp if 'q_A' in k]!r}"
        )

    if q_merged is None or q_merged.size != n:
        raise RuntimeError(
            "Singlet run expected merged q_A in post-processing-dict; "
            f"q_A-related keys: {[k for k in pp if 'q_A' in k]!r}"
        )
    return q_merged, np.zeros(n, dtype=float)
