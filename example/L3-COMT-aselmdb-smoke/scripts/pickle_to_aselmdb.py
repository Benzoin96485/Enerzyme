#!/usr/bin/env python
"""Convert Enerzyme labeled pickle fragments to ASE LMDB for Datahub.data_format=aselmdb.

Campaign pickles store energy/grad in Hartree; ASE DB stores eV / eV·Å⁻¹.
Gradients are converted to forces (Fa = -grad) before writing.
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
            symbols = [str(s) for s in rec["atom_type"]]
            atoms = Atoms(
                symbols=symbols,
                positions=np.asarray(rec["coord"], dtype=float),
                pbc=False,
            )
            e_ev = float(rec["energy"]) * ase.units.Ha
            forces_ev = -np.asarray(rec["grad"], dtype=float) * ase.units.Ha
            kwargs = {"energy": e_ev, "forces": forces_ev}
            if "dipole" in rec and rec["dipole"] is not None:
                # ASE dipole moment unit is e·Å; leave raw dipole in data if uncertain
                pass
            atoms.calc = SinglePointCalculator(atoms, **kwargs)
            info = {
                "charge": int(rec.get("total_chrg", 0)),
                "spin": int(rec.get("total_spin", 0)) + 1,
                "index": int(rec.get("index", i)),
            }
            db.write(atoms, data=info, index=info["index"])
            n += 1
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
