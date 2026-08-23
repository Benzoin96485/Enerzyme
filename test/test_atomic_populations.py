"""Unit tests for xtbml atomic population helpers (no tblite runtime)."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enerzyme.qm.xtb_population import atomic_populations as ap


def _fake_atoms(charge: float = 0.0, spin: int = 1, n_atoms: int = 2):
    atoms = MagicMock()
    atoms.info = {"charge": charge, "spin": spin}
    atoms.get_atomic_numbers.return_value = np.array([6] * n_atoms, dtype=int)
    atoms.get_positions.return_value = np.zeros((n_atoms, 3), dtype=float)
    return atoms


def _patch_tblite_run(pp: dict, *, charges: np.ndarray | None = None, n_atoms: int = 2):
    res = MagicMock()

    def _get(key):
        if key == "post-processing-dict":
            return pp
        if key == "charges":
            return charges
        if key == "natoms":
            return n_atoms
        raise KeyError(key)

    res.get.side_effect = _get
    calc = MagicMock()
    calc.singlepoint.return_value = None
    return (
        patch.object(ap, "register_xtbml", return_value="mock"),
        patch.object(ap, "Calculator", return_value=calc),
        patch.object(ap, "Result", return_value=res),
        patch.object(
            ap,
            "_atoms_to_tblite_inputs",
            return_value=(np.array([6] * n_atoms), np.zeros((n_atoms, 3))),
        ),
    )


def _run_with_mocks(pp: dict, *, charges: np.ndarray | None = None, n_atoms: int = 2):
    stack = ExitStack()
    for p in _patch_tblite_run(pp, charges=charges, n_atoms=n_atoms):
        stack.enter_context(p)
    return stack


def test_open_shell_falls_back_to_result_charges():
    atoms = _fake_atoms(spin=3)
    charges = np.array([0.2, -0.2], dtype=float)
    with _run_with_mocks({}, charges=charges):
        qa, sa = ap.atomic_Q_and_S_from_xtbml(atoms, max_scf_iter=1)
    assert np.allclose(qa, charges)
    assert np.allclose(sa, 0.0)


def test_open_shell_uses_merged_q_a_when_present():
    atoms = _fake_atoms(spin=3)
    merged = np.array([0.3, -0.3], dtype=float)
    with _run_with_mocks({"q_A": merged.tolist()}):
        qa, sa = ap.atomic_Q_and_S_from_xtbml(atoms, max_scf_iter=1)
    assert np.allclose(qa, merged)
    assert np.allclose(sa, 0.0)


def test_singlet_still_requires_q_a_or_charges():
    atoms = _fake_atoms(spin=1)
    with _run_with_mocks({}):
        with pytest.raises(RuntimeError, match="Singlet run expected merged q_A"):
            ap.atomic_Q_and_S_from_xtbml(atoms, max_scf_iter=1)
