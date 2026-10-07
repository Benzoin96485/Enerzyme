"""GPU4PySCF single-point / gradient jobs for Enerzyme annotate.

Theory, grids, SCF thresholds, optional C-PCM, and the Electrolites plugin
live in a YAML template. Geometry, charge, and multiplicity come from the
ASE atoms (multiplicity is ``atoms.info['spin']``; PySCF receives ``2S``).
A numeric ``atoms.info['epsilon']`` selects the template C-PCM dielectric, or
stock C-PCM when the template has no ``pcm`` block. Vacuum tokens skip PCM.

GPU4PySCF and Electrolites are imported only when a job runs. Electrolites is
applied in the worker process before ``gpu4pyscf.dft`` is used.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

from .electrolites_plugin import ElectrolitesPlugin, apply_electrolites, parse_electrolites

SCHEMA = "enerzyme_gpu4pyscf_job_v1"
VACUUM_TOKENS = frozenset({"vacuum", "vac", "none", "null", ""})

# Same stock cavity as the DC-UMA C-PCM Gate A profile.
STOCK_CPCM = {
    "method": "C-PCM",
    "surface_discretization_method": "SWIG",
    "vdw_scale": 1.2,
    "r_probe": 0.0,
    "lebedev_order": 29,
    "equilibrium_solvation": False,
}

_SETTING_KEYS = frozenset({
    "xc",
    "basis",
    "cart",
    "grids_level",
    "nlcgrids_level",
    "conv_tol",
    "direct_scf_tol",
    "max_cycle",
    "level_shift",
    "init_guess",
    "gradients",
    "verbose",
    "density_fit",
    "pcm",
    "electrolites",
    "precision_env",
})


class Gpu4PyscfError(RuntimeError):
    """Invalid GPU4PySCF template, job spec, or finished calculation."""


@dataclass(frozen=True)
class Gpu4PyscfSettings:
    xc: str
    basis: str
    cart: bool
    grids_level: int
    nlcgrids_level: int
    conv_tol: float
    direct_scf_tol: float
    max_cycle: int
    level_shift: float
    init_guess: str
    gradients: bool
    verbose: int
    density_fit: bool
    pcm: dict[str, Any] | None
    electrolites: ElectrolitesPlugin
    precision_env: dict[str, str]


def _env_str(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return str(value)


def is_vacuum(epsilon: Any) -> bool:
    if epsilon is None:
        return True
    if isinstance(epsilon, str) and epsilon.strip().lower() in VACUUM_TOKENS:
        return True
    return False


def validate_pcm(pcm: Mapping[str, Any] | None, *, eps: Any = None) -> dict[str, Any] | None:
    if pcm is None and is_vacuum(eps):
        return None
    merged = dict(STOCK_CPCM if pcm is None else pcm)
    if eps is not None and not is_vacuum(eps):
        merged["eps"] = float(eps)
    if is_vacuum(merged.get("eps")):
        return None
    method = str(merged.get("method") or "")
    if method.upper() not in {"C-PCM", "CPCM"}:
        raise Gpu4PyscfError(f"PCM method must be C-PCM, got {method!r}")
    dielectric = float(merged["eps"])
    if not math.isfinite(dielectric) or dielectric < 1.0:
        raise Gpu4PyscfError(f"PCM eps must be finite and >= 1, got {merged.get('eps')!r}")
    surface = str(merged.get("surface_discretization_method") or "")
    if surface.upper() != "SWIG":
        raise Gpu4PyscfError(f"PCM surface must be SWIG, got {surface!r}")
    return {
        "method": "C-PCM",
        "eps": dielectric,
        "lebedev_order": int(merged["lebedev_order"]),
        "vdw_scale": float(merged["vdw_scale"]),
        "r_probe": float(merged.get("r_probe", 0.0)),
        "surface_discretization_method": "SWIG",
        "equilibrium_solvation": bool(merged.get("equilibrium_solvation", False)),
    }


def load_gpu4pyscf_settings(path: Path | str) -> Gpu4PyscfSettings:
    template = Path(path)
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Gpu4PyscfError(f"GPU4PySCF template must be a mapping: {template}")
    unknown = sorted(set(payload) - _SETTING_KEYS)
    if unknown:
        raise Gpu4PyscfError(f"unrecognized GPU4PySCF template keys: {unknown}")
    pcm = payload.get("pcm")
    if pcm in (None, False) or (isinstance(pcm, str) and is_vacuum(pcm)):
        pcm_profile = None
    elif isinstance(pcm, dict):
        pcm_profile = validate_pcm(pcm)
    else:
        raise Gpu4PyscfError(f"pcm must be a mapping or null, got {pcm!r}")
    env_raw = payload.get("precision_env") or {}
    if not isinstance(env_raw, dict):
        raise Gpu4PyscfError("precision_env must be a mapping")
    return Gpu4PyscfSettings(
        xc=str(payload.get("xc") or ""),
        basis=str(payload.get("basis") or ""),
        cart=bool(payload.get("cart", False)),
        grids_level=int(payload.get("grids_level", 3)),
        nlcgrids_level=int(payload.get("nlcgrids_level", 3)),
        conv_tol=float(payload.get("conv_tol", 1.0e-10)),
        direct_scf_tol=float(payload.get("direct_scf_tol", 1.0e-13)),
        max_cycle=int(payload.get("max_cycle", 128)),
        level_shift=float(payload.get("level_shift", 0.0)),
        init_guess=str(payload.get("init_guess", "minao")),
        gradients=bool(payload.get("gradients", True)),
        verbose=int(payload.get("verbose", 0)),
        density_fit=bool(payload.get("density_fit", False)),
        pcm=pcm_profile,
        electrolites=parse_electrolites(payload.get("electrolites", False)),
        precision_env={str(key): _env_str(value) for key, value in env_raw.items()},
    )


def _pcm_for_atoms(settings: Gpu4PyscfSettings, atoms: Atoms) -> dict[str, Any] | None:
    if "epsilon" not in atoms.info:
        return None if settings.pcm is None else dict(settings.pcm)
    epsilon = atoms.info.get("epsilon")
    if is_vacuum(epsilon):
        return None
    return validate_pcm(settings.pcm, eps=epsilon)


def build_job_spec(
    atoms: Atoms,
    settings: Gpu4PyscfSettings,
    plugin: ElectrolitesPlugin,
) -> dict[str, Any]:
    if not settings.xc or not settings.basis:
        raise Gpu4PyscfError("GPU4PySCF template requires xc and basis")
    multiplicity = int(atoms.info.get("spin", 1))
    if multiplicity < 1:
        raise Gpu4PyscfError(f"multiplicity (atoms.info['spin']) must be >= 1, got {multiplicity}")
    numbers = [int(z) for z in atoms.get_atomic_numbers()]
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    return {
        "schema": SCHEMA,
        "index": int(atoms.info["index"]),
        "numbers": numbers,
        "positions": positions.tolist(),
        "charge": int(atoms.info.get("charge", 0)),
        "multiplicity": multiplicity,
        "pyscf_spin": multiplicity - 1,
        "xc": settings.xc,
        "basis": settings.basis,
        "cart": settings.cart,
        "grids_level": settings.grids_level,
        "nlcgrids_level": settings.nlcgrids_level,
        "conv_tol": settings.conv_tol,
        "direct_scf_tol": settings.direct_scf_tol,
        "max_cycle": settings.max_cycle,
        "level_shift": settings.level_shift,
        "init_guess": settings.init_guess,
        "gradients": settings.gradients,
        "verbose": settings.verbose,
        "density_fit": settings.density_fit,
        "pcm": _pcm_for_atoms(settings, atoms),
        "electrolites": plugin.to_dict(),
        "precision_env": dict(settings.precision_env),
    }


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


def _synchronize() -> None:
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        return


def _mole(spec: Mapping[str, Any]):
    from pyscf import gto

    numbers = [int(z) for z in spec["numbers"]]
    positions = np.asarray(spec["positions"], dtype=np.float64)
    atom = [(chemical_symbols[z], positions[i].tolist()) for i, z in enumerate(numbers)]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = spec["basis"]
    mol.charge = int(spec["charge"])
    mol.spin = int(spec["pyscf_spin"])
    mol.cart = bool(spec["cart"])
    mol.unit = "Angstrom"
    mol.verbose = int(spec.get("verbose") or 0)
    mol.build()
    return mol


def _attach_pcm(mf, pcm: Mapping[str, Any] | None):
    if not pcm:
        return mf
    mf = mf.PCM()
    solvent = mf.with_solvent
    solvent.method = pcm["method"]
    solvent.eps = float(pcm["eps"])
    solvent.lebedev_order = int(pcm["lebedev_order"])
    solvent.vdw_scale = float(pcm["vdw_scale"])
    solvent.r_probe = float(pcm["r_probe"])
    solvent.surface_discretization_method = pcm["surface_discretization_method"]
    solvent.equilibrium_solvation = bool(pcm["equilibrium_solvation"])
    return mf


def run_gpu4pyscf_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one RKS/UKS energy (and analytic gradient) from a job spec."""
    plugin = parse_electrolites(spec.get("electrolites") or False)
    for key, value in (spec.get("precision_env") or {}).items():
        os.environ[str(key)] = _env_str(value)
    patched = apply_electrolites(plugin)
    from gpu4pyscf import dft

    mol = _mole(spec)
    spin = int(spec["pyscf_spin"])
    if spin == 0:
        mf = dft.RKS(mol, xc=str(spec["xc"]))
    else:
        mf = dft.UKS(mol, xc=str(spec["xc"]))
    if spec.get("density_fit"):
        mf = mf.density_fit()
    mf.conv_tol = float(spec["conv_tol"])
    mf.direct_scf_tol = float(spec["direct_scf_tol"])
    mf.max_cycle = int(spec["max_cycle"])
    mf.level_shift = float(spec.get("level_shift") or 0.0)
    mf.init_guess = str(spec.get("init_guess") or "minao")
    mf.verbose = int(spec.get("verbose") or 0)
    if getattr(mf, "grids", None) is not None:
        mf.grids.level = int(spec["grids_level"])
    if getattr(mf, "nlcgrids", None) is not None:
        mf.nlcgrids.level = int(spec["nlcgrids_level"])
    mf = _attach_pcm(mf, spec.get("pcm"))
    _synchronize()
    energy = mf.kernel()
    _synchronize()
    converged = bool(mf.converged)
    payload: dict[str, Any] = {
        "ok": converged and energy is not None,
        "scf_converged": converged,
        "scf_cycles": int(getattr(mf, "cycles", 0) or 0),
        "energy_ha": None if energy is None else float(energy),
        "grad_ha_bohr": None,
        "dipole_au": None,
        "electrolites_patched": list(patched),
        "pcm": None if not spec.get("pcm") else dict(spec["pcm"]),
        "error": None if converged else "SCF did not converge",
    }
    if not payload["ok"] or not spec.get("gradients", True):
        return payload
    _synchronize()
    grad = mf.nuc_grad_method().kernel()
    _synchronize()
    payload["grad_ha_bohr"] = _as_numpy(grad).astype(np.float64).tolist()
    try:
        dipole = mf.dip_moment(unit="AU", verbose=0)
        payload["dipole_au"] = _as_numpy(dipole).reshape(-1)[:3].astype(np.float64).tolist()
    except Exception:
        payload["dipole_au"] = None
    return payload


def energy_ha_to_ev(energy_ha: float) -> float:
    return float(energy_ha) * Hartree


def grad_ha_bohr_to_forces_ev_a(grad_ha_bohr: Sequence[Sequence[float]]) -> np.ndarray:
    """``F = -dE/dR`` from a Hartree/Bohr gradient."""
    return -np.asarray(grad_ha_bohr, dtype=np.float64) * Hartree / Bohr


def dipole_au_to_e_angstrom(dipole_au: Sequence[float]) -> np.ndarray:
    return np.asarray(dipole_au, dtype=np.float64) * Bohr


def result_log(spec: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines = [
        "enerzyme gpu4pyscf",
        f"index {spec.get('index')}",
        f"xc {spec.get('xc')} basis {spec.get('basis')}",
        f"electrolites {spec.get('electrolites')}",
        f"pcm {spec.get('pcm')}",
        f"scf_converged {result.get('scf_converged')} cycles {result.get('scf_cycles')}",
        f"energy_ha {result.get('energy_ha')}",
        f"patched {result.get('electrolites_patched')}",
    ]
    if result.get("error"):
        lines.append(f"error {result['error']}")
    return "\n".join(lines) + "\n"
