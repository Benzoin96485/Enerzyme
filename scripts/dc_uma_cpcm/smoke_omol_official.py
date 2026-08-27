#!/usr/bin/env python3
"""Official UMA-s-1p2 OMol inference smoke gate.

Pins fairchem via PYTHONPATH, records versions/hashes, and runs
FAIRChemCalculator(task_name="omol", inference_settings="batch") on a
tiny water molecule. Exits non-zero if the pinned source cannot be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON path under experiments/dc_uma_cpcm/reports.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from enerzyme.models.esen.runtime import (
        EXPERIMENTS_ROOT,
        OMOL_TASK_NAME,
        UMA_MODEL_NAME,
        apply_runtime_env,
        collect_runtime_versions,
        ensure_pinned_fairchem,
        resolve_uma_checkpoint,
    )

    apply_runtime_env()
    ensure_pinned_fairchem()
    versions = collect_runtime_versions()
    ckpt = resolve_uma_checkpoint()

    import torch
    from ase.build import molecule
    from fairchem.core import FAIRChemCalculator

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.pbc = False
    calc = FAIRChemCalculator.from_model_checkpoint(
        str(ckpt),
        task_name=OMOL_TASK_NAME,
        inference_settings="batch",
        device=device,
    )
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite official OMol prediction: E={energy}, F={forces}")

    report = {
        "ok": True,
        "fairchem_module": versions["fairchem_file"],
        "uma_model_name": UMA_MODEL_NAME,
        "task_name": OMOL_TASK_NAME,
        "inference_settings": "batch",
        "device": device,
        "water_energy_eV": energy,
        "water_forces_eV_A": forces.tolist(),
        "versions": versions,
    }
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.report is not None:
        report_path = args.report
    else:
        report_path = EXPERIMENTS_ROOT / "reports" / "omol_official_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text + "\n")
    print(f"wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — smoke gate must fail loudly
        print(f"OMOL official smoke gate failed: {exc}", file=sys.stderr)
        raise
