#!/usr/bin/env python
"""Convert labeled pickle fragments to ASE LMDB for Datahub.data_format=aselmdb.

Accepts either:
- standard Enerzyme keys (``E``, ``Fa``, ``M2``, ``Ra``, ``Za``, ``Q``, ``S``), or
- legacy Enerzymette keys (``energy``, ``grad``, ``dipole``, ``coord``, ``atom_type``, …).

Campaign pickles store energy in Hartree; ASE DB stores eV / eV·Å⁻¹.
Legacy ``grad`` is converted to forces (``Fa = -grad``) before writing.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import ase.units
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect


def _as_symbols(za_or_symbols) -> list[str]:
    from ase.data import chemical_symbols

    arr = list(za_or_symbols)
    if arr and isinstance(arr[0], (int, np.integer)):
        return [chemical_symbols[int(z)] for z in arr]
    return [str(s) for s in arr]


def _record_to_atoms_kwargs(rec: dict, index: int):
    if "Ra" in rec:
        positions = np.asarray(rec["Ra"], dtype=float)
        symbols = _as_symbols(rec["Za"])
        e_ha = float(rec["E"])
        forces_ha = np.asarray(rec["Fa"], dtype=float)
        charge = int(rec.get("Q", 0))
        spin = int(rec.get("S", 0)) + 1
        dipole = rec.get("M2")
        idx = int(rec.get("index", index))
    else:
        positions = np.asarray(rec["coord"], dtype=float)
        symbols = _as_symbols(rec["atom_type"])
        e_ha = float(rec["energy"])
        forces_ha = -np.asarray(rec["grad"], dtype=float)
        charge = int(rec.get("total_chrg", 0))
        spin = int(rec.get("total_spin", 0)) + 1
        dipole = rec.get("dipole")
        idx = int(rec.get("index", index))

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    kwargs = {
        "energy": e_ha * ase.units.Ha,
        "forces": forces_ha * ase.units.Ha,
    }
    if dipole is not None:
        kwargs["dipole"] = np.asarray(dipole, dtype=float)
    info = {"charge": charge, "spin": spin, "index": idx}
    return atoms, kwargs, info


def convert(pickle_path: Path, out_path: Path, limit: int | None = None) -> int:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list of dicts in {pickle_path}, got {type(data)}")
    if limit is not None:
        data = data[:limit]
    if out_path.exists():
        out_path.unlink()
    n = 0
    with connect(str(out_path)) as db:
        for i, rec in enumerate(data):
            atoms, calc_kwargs, info = _record_to_atoms_kwargs(rec, i)
            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
            db.write(atoms, data=info, index=info["index"])
            n += 1
        # Prefer schema over first-row probing when Datahub loads the DB.
        from enerzyme.data.datahub import ASELMDB_METADATA_PROPERTIES_KEY

        meta = dict(db.metadata or {})
        meta[ASELMDB_METADATA_PROPERTIES_KEY] = [
            "Ra", "Za", "N", "Q", "S", "E", "Fa", "M2",
        ]
        db.metadata = meta
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True, help="fragments.pkl")
    p.add_argument("-o", "--output", type=Path, required=True, help="fragments.aselmdb")
    p.add_argument("-n", "--limit", type=int, default=None)
    args = p.parse_args()
    n = convert(args.input, args.output, args.limit)
    print(f"Wrote {n} structures to {args.output}")


if __name__ == "__main__":
    main()
