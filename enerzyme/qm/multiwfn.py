"""Multiwfn atomic-charge post-processing for ``enerzyme annotate``.

TeraChem (GPU) writes a molden; a separate CPU process pool runs Multiwfn so
charge jobs never occupy a TeraChem worker. Workers only write ``.chg`` files;
the parent merges ``Qa`` into ASE LMDB / pickle after the pool joins.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy, which
from typing import Any, Dict, List, Mapping, Optional, Sequence
import os
import subprocess

import numpy as np

from ..utils import logger


# Methods that ask "how to obtain atomic densities?" after the population option.
# Answer ``1`` = built-in sphericalized atomic densities, then ``y`` to write .chg.
_DENSITY_THEN_CHG: Dict[str, int] = {
    "hirshfeld": 1,
    "vdd": 2,
    "adch": 11,
    "cm5": 16,
    "1.2cm5": -16,
}

# Löwdin asks for a population-print path before the .chg prompt (blank = screen).
_LOWDIN_OPTION = 6

# Mulliken has a nested submenu: ``1`` outputs charges, then ``y`` writes .chg,
# then ``0`` leaves the submenu and ``0`` leaves the population menu.
_MULLIKEN_OPTION = 5

# SCPA writes .chg immediately after the option (no density prompt).
_SCPA_OPTION = 7

# MBIS: extra submenu ``1`` = start calculation, then ``y`` writes .chg.
_MBIS_OPTION = 20

CHARGE_METHODS: Dict[str, str] = {
    "hirshfeld": "Hirshfeld atomic charge (menu 1)",
    "vdd": "Voronoi deformation density (menu 2)",
    "mulliken": "Mulliken (menu 5)",
    "lowdin": "Löwdin (menu 6)",
    "scpa": "Modified Mulliken / SCPA (menu 7)",
    "adch": "ADCH (menu 11)",
    "cm5": "CM5 (menu 16)",
    "1.2cm5": "1.2×CM5 (menu -16)",
    "mbis": "MBIS (menu 20)",
}

_SENTINEL = None


def normalize_charge_method(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "1.2*cm5": "1.2cm5",
        "12cm5": "1.2cm5",
        "löwdin": "lowdin",
        "hirschfeld": "hirshfeld",
    }
    return aliases.get(key, key)


def build_multiwfn_stdin(
    charge_method: str = "1.2cm5",
    menu_inputs: Optional[Sequence[str]] = None,
) -> str:
    """Build Multiwfn stdin **after** the wavefunction is already loaded.

    Pass the molden path as argv (``Multiwfn_noGUI molden -nt N``) so this string
    starts at the main menu, not at the file-path prompt.
    """
    if menu_inputs:
        lines = [str(x) for x in menu_inputs]
        return "\n".join(lines) + "\n"

    method = normalize_charge_method(charge_method)
    if method in _DENSITY_THEN_CHG:
        option = _DENSITY_THEN_CHG[method]
        lines = ["7", str(option), "1", "y", "0", "q"]
    elif method == "lowdin":
        lines = ["7", str(_LOWDIN_OPTION), "", "y", "0", "q"]
    elif method == "mulliken":
        lines = ["7", str(_MULLIKEN_OPTION), "1", "y", "0", "0", "q"]
    elif method == "scpa":
        lines = ["7", str(_SCPA_OPTION), "y", "0", "q"]
    elif method == "mbis":
        lines = ["7", str(_MBIS_OPTION), "1", "y", "0", "q"]
    else:
        known = ", ".join(sorted(CHARGE_METHODS))
        raise ValueError(
            f"Unknown Multiwfn charge_method {charge_method!r}. "
            f"Known methods: {known}. Or pass menu_inputs for a custom recipe "
            "(Hirshfeld-I / RESP / CHELPG / AIM need extra prompts)."
        )
    return "\n".join(lines) + "\n"


def parse_chg_file(chg_path: Path, n_atoms: Optional[int] = None) -> np.ndarray:
    """Read Multiwfn ``.chg`` column 5 (atomic charge)."""
    path = Path(chg_path)
    if not path.is_file():
        raise FileNotFoundError(f"Multiwfn .chg not found: {path}")
    qa = np.loadtxt(path, usecols=4, ndmin=1, dtype=float)
    if n_atoms is not None and qa.size != int(n_atoms):
        raise ValueError(
            f"{path} has {qa.size} charges but structure has {n_atoms} atoms"
        )
    return qa


def resolve_multiwfn_executable(executable: str) -> str:
    path = Path(executable).expanduser()
    if path.is_file():
        return str(path.resolve())
    found = which(executable)
    if found:
        return found
    raise FileNotFoundError(
        f"Multiwfn executable {executable!r} not found on PATH. "
        "Load a noGUI module (e.g. multiwfn/v260410-noGui) in the job script."
    )


def invoke_multiwfn(
    molden: Path,
    workdir: Path,
    n_threads: int,
    executable: str = "Multiwfn_noGUI",
    timeout: Optional[float] = None,
    settings_ini: Optional[Path] = None,
    charge_method: str = "1.2cm5",
    menu_inputs: Optional[Sequence[str]] = None,
) -> Path:
    """Run Multiwfn on one molden in ``workdir``; return the written ``.chg`` path."""
    molden = Path(molden).resolve()
    if not molden.is_file() or molden.stat().st_size <= 0:
        raise FileNotFoundError(f"Invalid molden for Multiwfn: {molden}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if settings_ini is not None:
        src = Path(settings_ini).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Multiwfn settings_ini not found: {src}")
        copy(src, workdir / "settings.ini")

    exe = resolve_multiwfn_executable(executable)
    n_threads = max(1, int(n_threads))
    cmd = [exe, str(molden), "-nt", str(n_threads)]
    stdin = build_multiwfn_stdin(charge_method=charge_method, menu_inputs=menu_inputs)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    log_path = workdir / "multiwfn.log"
    with open(log_path, "w") as log:
        try:
            subprocess.run(
                cmd,
                input=stdin,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                cwd=str(workdir),
                env=env,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"Multiwfn timed out after {timeout}s on {molden}"
            ) from e

    chg = workdir / f"{molden.stem}.chg"
    if not chg.is_file():
        raise FileNotFoundError(
            f"Multiwfn did not write {chg}. See {log_path}"
        )
    return chg


@dataclass
class MultiwfnConfig:
    enabled: bool = False
    executable: str = "Multiwfn_noGUI"
    charge_method: str = "1.2cm5"
    n_threads: int = 12
    n_processes: int = 1
    timeout: Optional[float] = None
    keep_chg: bool = True
    keep_molden: bool = False
    menu_inputs: Optional[List[str]] = None
    settings_ini: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "executable": self.executable,
            "charge_method": self.charge_method,
            "n_threads": int(self.n_threads),
            "n_processes": int(self.n_processes),
            "timeout": self.timeout,
            "keep_chg": bool(self.keep_chg),
            "keep_molden": bool(self.keep_molden),
            "menu_inputs": list(self.menu_inputs) if self.menu_inputs else None,
            "settings_ini": self.settings_ini,
        }


def parse_multiwfn_config(
    raw: Optional[Mapping[str, Any]],
    *,
    keep_molden: bool = False,
) -> MultiwfnConfig:
    if not raw:
        return MultiwfnConfig(enabled=False, keep_molden=keep_molden)
    data = dict(raw)
    enabled = bool(data.get("enabled", True))
    menu_inputs = data.get("menu_inputs")
    if menu_inputs is not None:
        menu_inputs = [str(x) for x in menu_inputs]
    n_proc = int(data.get("n_processes", 1) or 1)
    n_threads = int(data.get("n_threads", 12) or 1)
    timeout = data.get("timeout", None)
    if timeout is not None:
        timeout = float(timeout)
    method = str(data.get("charge_method", "1.2cm5"))
    cfg = MultiwfnConfig(
        enabled=enabled,
        executable=str(data.get("executable", "Multiwfn_noGUI")),
        charge_method=method,
        n_threads=max(1, n_threads),
        n_processes=max(1, n_proc),
        timeout=timeout,
        keep_chg=bool(data.get("keep_chg", True)),
        keep_molden=keep_molden,
        menu_inputs=menu_inputs,
        settings_ini=data.get("settings_ini"),
    )
    if cfg.enabled and not menu_inputs:
        # Validate method early so annotate fails before QC, not mid-queue.
        build_multiwfn_stdin(charge_method=cfg.charge_method)
    return cfg


def chg_dirname(charge_method: str) -> str:
    method = normalize_charge_method(charge_method)
    if method == "1.2cm5":
        return "chrg-12CM5"
    return f"chrg-{method}"


def _multiwfn_process_main(
    job_queue,
    config_dict: Dict[str, Any],
    chg_dir: str,
    tmp_base: str,
) -> None:
    """Long-lived Multiwfn worker: pull jobs until a sentinel (``None``)."""
    chg_root = Path(chg_dir)
    chg_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tmp_base)
    tmp_root.mkdir(parents=True, exist_ok=True)
    settings = config_dict.get("settings_ini")
    settings_path = Path(settings) if settings else None
    while True:
        job = job_queue.get()
        if job is _SENTINEL:
            break
        index = int(job["index"])
        dest = chg_root / f"{index}.chg"
        if dest.is_file() and dest.stat().st_size > 0:
            logger.info(f"Multiwfn .chg for {index} already exists at {dest}; skipping")
            continue
        workdir = tmp_root / str(index)
        try:
            chg = invoke_multiwfn(
                molden=Path(job["molden"]),
                workdir=workdir,
                n_threads=int(config_dict["n_threads"]),
                executable=str(config_dict["executable"]),
                timeout=config_dict.get("timeout"),
                settings_ini=settings_path,
                charge_method=str(config_dict["charge_method"]),
                menu_inputs=config_dict.get("menu_inputs"),
            )
            n_atoms = job.get("n_atoms")
            parse_chg_file(chg, n_atoms=n_atoms)
            if chg.resolve() != dest.resolve():
                copy(chg, dest)
            logger.info(f"Multiwfn charges written: {dest}")
            if not config_dict.get("keep_molden"):
                molden = Path(job["molden"])
                try:
                    if molden.is_file():
                        molden.unlink()
                except OSError as e:
                    logger.warning(f"Could not delete staged molden {molden}: {e}")
        except Exception as e:
            logger.warning(f"Multiwfn failed for structure {index}: {e}")
