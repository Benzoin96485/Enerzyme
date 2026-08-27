"""ORCA 6.1.1 C-PCM single-point EnGrad jobs for the DC-UMA PoC.

Input keywords are copied from ``fairchem.data.omol.orca.calc``
(commit ``de5db01588da57665bde96966091d268a9b6b8f7``): wB97M-V / def2-TZVPD,
EnGrad, RIJCOSX, DEFGRID3, Tight SCF, ALLPOP. The only solvent addition is an
electrostatic ``%cpcm epsilon ...`` block. SMD, DRACO, and non-electrostatic
CDS terms are never written.

Vacuum jobs omit the C-PCM block entirely (they are not ``epsilon=1``).
Failure retries may only switch to SlowConv, raise MaxIter to 600, and drop
the MO guess. Theory, grid, and cavity are never changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

from enerzyme.data.dc_uma_cpcm.constants import (
    DEFAULT_EXPERIMENTS,
    DEF2_TZVPD_N_BASIS,
    N_DIAG_JOBS,
    N_PILOT_JOBS,
    N_TOTAL_LABELS,
    PILOT_EPSILONS,
)
from enerzyme.data.dc_uma_cpcm.index import is_closed_shell, parse_s_squared
from enerzyme.data.dc_uma_cpcm.sample import size_bin

# --- OMol / fairchem.data.omol.orca.calc constants (do not import fairchem) ---
ORCA_FUNCTIONAL = "wB97M-V"
ORCA_BASIS = "def2-TZVPD"
ORCA_SIMPLE_INPUT = [
    "EnGrad",
    "RIJCOSX",
    "def2/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID3",
    "ALLPOP",
]
ORCA_SCF_CONVERGENCE = "Tight"
ORCA_SCF_MAXITER = 300
ORCA_RETRY_MAXITER = 600
ORCA_SCF_THRESH = "1e-12"
ORCA_SCF_TCUT = "1e-13"
NBO_FLAGS = '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end'

PINNED_FAIRCHEM_ORCA_BASIS = (
    Path("/global/cfs/cdirs/m4787/src/fairchem/src/fairchem/data/omol/orca/basis")
    / "def2-tzvpd.bas"
)
DEFAULT_ORCA_ROOT = Path(DEFAULT_EXPERIMENTS) / "orca"
DEFAULT_MANIFEST = Path(DEFAULT_EXPERIMENTS) / "manifests" / "dc_uma_cpcm_manifest.json"
ARRAY_MAP_NAME = "array_map.json"
STATUS_NAME = "status.json"
META_NAME = "meta.json"
INPUT_NAME = "orca.inp"
OUTPUT_NAME = "orca.out"
ENGRAD_NAME = "orca.engrad"
HASH_NAME = "input.sha256"
BASIS_COPY_NAME = "def2-tzvpd.bas"

FORBIDDEN_SOLVENT_TOKENS = (
    "smd",
    "draco",
    "cds",
    "smdcds",
    "cpcmx",
    "cpcm(water)",
    "cpcm(ethanol)",
    "cpcm(dmso)",
    "cpcm(methanol)",
    "cpcm(acetone)",
    "cpcm(acetonitrile)",
    "cpcm(thf)",
)
VACUUM_TOKENS = frozenset({"vacuum", "vac", "none", ""})
RETRYABLE_HINTS = ("scf not converged", "not converged even after", "maxiter", "slowconv")
# MPI / node transients: segfault, bus error, or abort before a normal EnGrad finish.
INFRA_RETRYABLE_HINTS = (
    "segmentation fault",
    "bus error",
    "mpi_abort",
    "exited on signal",
    "termination in startup",
    "termination in propint",
    "termination in guess",
    "termination in leanscf",
    "termination in properties",
    "termination in scf",
)
NON_RETRYABLE_HINTS = (
    "unknown keyword",
    "error in the input",
    "cannot open",
    "basis set",
    "illegal",
    "syntax error",
)
MAX_ATTEMPTS = 2
FD_DELTA_ANG = 1.0e-3
# Vacuum recompute vs OMol historical: ~1 mHa is above typical SCF noise.
VACUUM_NOISE_HA = 1.0e-3
VACUUM_STATE_CHANGE_EV = 0.50
FORCE_NOISE_EV_A = 0.05
VACUUM_PAIR_FORCE_RMSE_EV_A = 0.10
VACUUM_PAIR_ENERGY_EV = 0.05
FIRST_PASS_MIN = 0.90

ExecuteFn = Callable[[Path, str], subprocess.CompletedProcess]


class OrcaCpcmError(RuntimeError):
    """Invalid C-PCM / ORCA job specification."""


def is_vacuum(epsilon: Any) -> bool:
    if epsilon is None:
        return True
    if isinstance(epsilon, str) and epsilon.strip().lower() in VACUUM_TOKENS:
        return True
    return False


def normalize_epsilon(epsilon: Any) -> float | None:
    """Return a dielectric > 1, or None for a true vacuum job."""
    if is_vacuum(epsilon):
        return None
    value = float(epsilon)
    if not np.isfinite(value) or value < 1.0:
        raise OrcaCpcmError(f"dielectric must be finite and >= 1, got {epsilon!r}")
    return value


def epsilon_tag(epsilon: Any) -> str:
    if is_vacuum(epsilon):
        return "vacuum"
    value = float(epsilon)
    text = f"{value:g}"
    return text.replace(".", "p")


def job_dirname(index: int, epsilon: Any, width: int = 2) -> str:
    return f"{int(index):0{int(width)}d}_eps-{epsilon_tag(epsilon)}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def get_n_basis(numbers: Iterable[int]) -> int:
    total = 0
    for atom_z in numbers:
        sym = chemical_symbols[int(atom_z)]
        nbf = DEF2_TZVPD_N_BASIS.get(sym)
        if nbf is None:
            raise KeyError(f"No def2-TZVPD basis count for element {sym}")
        total += int(nbf)
    return total


def get_mem_estimate_mb(numbers: Sequence[int], multiplicity: int = 1) -> int:
    """Upper-bound MB copied from fairchem ``get_mem_estimate`` (RKS singlet)."""
    nbasis = get_n_basis(numbers)
    if int(multiplicity) == 1:
        a, b = 0.0076739752343756434, 361.4745947062764
    else:
        a, b = 0.016460518374501867, -320.38502508802776
    return int(max(a * (nbasis**1.5) + b, 1000))


def resolve_orca_binary() -> str:
    env = os.environ.get("ORCA_PATH") or os.environ.get("ORCA_BIN")
    if env:
        path = Path(env)
        if path.is_file():
            return str(path)
        nested = path / "orca"
        if nested.is_file():
            return str(nested)
    found = shutil.which("orca")
    if not found:
        raise FileNotFoundError("orca binary not found; load orca/6.1.1 or set ORCA_PATH")
    return found


def orca_version_text(orca_bin: str | None = None) -> str | None:
    try:
        binary = orca_bin or resolve_orca_binary()
    except FileNotFoundError:
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return binary
    text = (proc.stdout or proc.stderr or "").strip()
    return text.splitlines()[0] if text else binary


def _strip_orca_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cut = line.find("#")
        lines.append(line if cut < 0 else line[:cut])
    return "\n".join(lines)


def _assert_no_forbidden_solvent(text: str) -> None:
    lowered = _strip_orca_comments(text).lower()
    for token in FORBIDDEN_SOLVENT_TOKENS:
        if token in lowered:
            raise OrcaCpcmError(f"forbidden solvent keyword {token!r} in ORCA input")


def cpcm_block(epsilon: Any) -> str:
    """Electrostatic C-PCM only. Empty string for vacuum."""
    eps = normalize_epsilon(epsilon)
    if eps is None:
        return ""
    block = f"%cpcm\n  epsilon {eps:g}\nend\n"
    _assert_no_forbidden_solvent(block)
    return block


def simple_input_line(*, energy_only: bool = False, slowconv: bool = False) -> str:
    tokens = [ORCA_FUNCTIONAL, ORCA_BASIS]
    tokens.extend(ORCA_SIMPLE_INPUT)
    if energy_only:
        tokens = [tok for tok in tokens if tok.lower() != "engrad"]
    if slowconv:
        tokens = ["SlowConv" if tok == "NormalConv" else tok for tok in tokens]
    return "! " + " ".join(tokens)


def scf_block(*, maxiter: int = ORCA_SCF_MAXITER, fresh_guess: bool = False) -> str:
    lines = [
        "%scf",
        f"  Convergence {ORCA_SCF_CONVERGENCE}",
        f"  maxiter {int(maxiter)}",
        f"  THRESH {ORCA_SCF_THRESH}",
        f"  TCUT {ORCA_SCF_TCUT}",
    ]
    if fresh_guess:
        lines.append("  Guess PModel")
    lines.append("end")
    return "\n".join(lines) + "\n"


def omol_blocks(
    *,
    nprocs: int,
    maxcore_mb: int,
    maxiter: int = ORCA_SCF_MAXITER,
    fresh_guess: bool = False,
    nbo: bool = False,
) -> str:
    parts = [
        f"%pal\n  nprocs {int(nprocs)}\nend\n",
        f"%maxcore {int(maxcore_mb)}\n",
        scf_block(maxiter=maxiter, fresh_guess=fresh_guess),
        "%elprop\n  Dipole true\n  Quadrupole true\nend\n",
        "%output\n"
        "  Print[P_ReducedOrbPopMO_L] 1\n"
        "  Print[P_ReducedOrbPopMO_M] 1\n"
        "  Print[P_BondOrder_L] 1\n"
        "  Print[P_BondOrder_M] 1\n"
        "  Print[P_Fockian] 1\n"
        "  Print[P_OrbEn] 2\n"
        "end\n",
        f'%basis\n  GTOName "{BASIS_COPY_NAME}"\nend\n',
    ]
    if nbo:
        parts.append(NBO_FLAGS + "\n")
    else:
        # ALLPOP still prints Loewdin; skip external NBO (not on this module).
        pass
    return "\n".join(parts)


def xyz_block(numbers: Sequence[int], positions: Sequence[Sequence[float]], charge: int, multiplicity: int) -> str:
    lines = [f"* xyz {int(charge)} {int(multiplicity)}"]
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] != len(list(numbers)) or pos.shape[1] != 3:
        raise OrcaCpcmError(f"positions shape {pos.shape} does not match {len(list(numbers))} atoms")
    for atom_z, xyz in zip(numbers, pos, strict=True):
        sym = chemical_symbols[int(atom_z)]
        lines.append(f"  {sym:<3s} {xyz[0]: .14f} {xyz[1]: .14f} {xyz[2]: .14f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def render_orca_input(
    *,
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    charge: int,
    multiplicity: int,
    epsilon: Any,
    nprocs: int = 32,
    maxcore_mb: int | None = None,
    maxiter: int = ORCA_SCF_MAXITER,
    slowconv: bool = False,
    fresh_guess: bool = False,
    energy_only: bool = False,
    nbo: bool = False,
    comment: str = "",
) -> str:
    if maxcore_mb is None:
        maxcore_mb = get_mem_estimate_mb(numbers, multiplicity)
    header = []
    if comment:
        for line in comment.strip().splitlines():
            header.append(f"# {line}")
    header.append("# Electrostatic continuum only; extra non-electrostatic solvent models stay off.")
    chunks = [
        "\n".join(header) + "\n",
        simple_input_line(energy_only=energy_only, slowconv=slowconv) + "\n\n",
        omol_blocks(
            nprocs=nprocs,
            maxcore_mb=maxcore_mb,
            maxiter=maxiter,
            fresh_guess=fresh_guess,
            nbo=nbo,
        ),
        cpcm_block(epsilon),
        xyz_block(numbers, positions, charge, multiplicity),
    ]
    text = "\n".join(chunk for chunk in chunks if chunk)
    _assert_no_forbidden_solvent(text)
    if not energy_only and "EnGrad" not in text:
        raise OrcaCpcmError("EnGrad missing from gradient job input")
    if is_vacuum(epsilon) and "%cpcm" in text.lower():
        raise OrcaCpcmError("vacuum jobs must not contain a %cpcm block")
    if not is_vacuum(epsilon) and "%cpcm" not in text.lower():
        raise OrcaCpcmError("C-PCM jobs must contain a %cpcm epsilon block")
    return text


def copy_omol_basis(dest_dir: Path, basis_src: Path | None = None) -> Path:
    src = Path(basis_src) if basis_src is not None else PINNED_FAIRCHEM_ORCA_BASIS
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / BASIS_COPY_NAME
    if not src.is_file():
        raise FileNotFoundError(f"OMol def2-TZVPD basis file missing: {src}")
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    shutil.copy2(src, dest)
    return dest


@dataclass
class ParsedOrca:
    energy_ha: float | None = None
    forces_ha_bohr: list[list[float]] | None = None
    lowdin_charges: list[float] | None = None
    dipole_au: list[float] | None = None
    dipole_debye: float | None = None
    scf_converged: bool | None = None
    scf_cycles: int | None = None
    terminated_normally: bool = False
    orca_version: str | None = None
    cpcm_epsilon: float | None = None
    cpcm_surface: str | None = None
    cpcm_feps: str | None = None
    cpcm_smd: bool = False
    cpcm_draco: bool = False
    has_cartesian_gradient: bool = False
    numerical_gradient: bool = False
    cpcm_gradient_mentioned: bool = False
    retryable: bool = False
    infra_failure: bool = False
    error: str | None = None
    raw_hints: list[str] = field(default_factory=list)

    @property
    def energy_ev(self) -> float | None:
        if self.energy_ha is None:
            return None
        return float(self.energy_ha * Hartree)

    @property
    def forces_ev_a(self) -> list[list[float]] | None:
        if self.forces_ha_bohr is None:
            return None
        arr = -np.asarray(self.forces_ha_bohr, dtype=np.float64) * Hartree / Bohr
        return arr.tolist()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["energy_ev"] = self.energy_ev
        payload["forces_ev_a"] = self.forces_ev_a
        return payload


_RE_VERSION = re.compile(r"Program Version\s+([0-9][0-9.]+)", re.I)
_RE_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+([-+0-9.Ee]+)")
_RE_SCF_OK = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", re.I)
_RE_SCF_FAIL = re.compile(r"SCF NOT CONVERGED", re.I)
_RE_LOEWDIN_BLOCK = re.compile(
    r"LOEWDIN ATOMIC CHARGES\s*-+\s*(.*?)\n\s*\n",
    re.I | re.S,
)
_RE_LOEWDIN_LINE = re.compile(r"^\s*\d+\s+[A-Za-z]+\s*:\s*([-+0-9.Ee]+)", re.M)
_RE_DIPOLE = re.compile(
    r"Total Dipole Moment\s*:\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
    re.I,
)
_RE_DIPOLE_DEBYE = re.compile(r"Magnitude \(Debye\)\s*:\s*([-+0-9.Ee]+)", re.I)
_RE_EPS = re.compile(r"^\s*Epsilon\s+\.+\s+([0-9.]+)", re.I | re.M)
_RE_SURFACE = re.compile(r"Surface type\s+\.+\s+(.+)$", re.I | re.M)
_RE_FEPS = re.compile(r"Epsilon function type\s+\.+\s+(\S+)", re.I)
_RE_CART_GRAD = re.compile(r"CARTESIAN GRADIENT", re.I)
_RE_NUM_GRAD = re.compile(r"NUMERICAL GRADIENT", re.I)
_RE_CPCM_GRAD = re.compile(r"CPCM[^\n]{0,80}GRADIENT|solvation contribution to the gradient", re.I)
_RE_NORMAL = re.compile(r"ORCA TERMINATED NORMALLY", re.I)
_RE_NATOMS_ENGRAD = re.compile(r"# Number of atoms\s+#\s+(\d+)", re.I | re.S)
_RE_ENERGY_ENGRAD = re.compile(r"# The current total energy in Eh\s+#\s+([-+0-9.Ee]+)", re.I | re.S)


def _cpcm_parameters_block(text: str) -> str:
    match = re.search(
        r"CPCM parameters:(.*?)(?:\n\s*\n|CARTESIAN GRADIENT|SCF CONVERGED|FINAL SINGLE POINT)",
        text,
        re.I | re.S,
    )
    return match.group(1) if match else ""


def _solvent_flag_on(text: str, name: str) -> bool:
    """True only when the CPCM parameter table reports the extra as enabled."""
    block = _cpcm_parameters_block(text)
    if not block:
        return False
    match = re.search(rf"^\s*{name}\b[^\n]*$", block, re.I | re.M)
    if match is None:
        return False
    snippet = match.group(0).lower()
    if re.search(r"\b(true|yes)\b", snippet):
        return True
    if re.search(r"\b(false|no)\b", snippet):
        return False
    num = re.search(r"([-+]?\d+(?:\.\d+)?)\s*$", snippet)
    if num is None:
        return False
    return abs(float(num.group(1))) > 0.0


def parse_engrad(text: str) -> tuple[float, np.ndarray]:
    n_match = _RE_NATOMS_ENGRAD.search(text)
    e_match = _RE_ENERGY_ENGRAD.search(text)
    if n_match is None or e_match is None:
        raise OrcaCpcmError("could not parse ORCA .engrad header")
    n_atoms = int(n_match.group(1))
    energy = float(e_match.group(1))
    grad_match = re.search(
        r"# The current gradient in Eh/bohr\s+#\s*(.*?)(?:\n\s*#|\Z)",
        text,
        re.I | re.S,
    )
    if grad_match is None:
        raise OrcaCpcmError("could not parse ORCA .engrad gradient block")
    values = [float(tok) for tok in grad_match.group(1).split()]
    if len(values) != 3 * n_atoms:
        raise OrcaCpcmError(f".engrad has {len(values)} gradient entries, expected {3 * n_atoms}")
    return energy, np.asarray(values, dtype=np.float64).reshape(n_atoms, 3)


def parse_orca_output(text: str, engrad_text: str | None = None) -> ParsedOrca:
    parsed = ParsedOrca()
    ver = _RE_VERSION.search(text)
    if ver:
        parsed.orca_version = ver.group(1)
    energies = _RE_ENERGY.findall(text)
    if energies:
        parsed.energy_ha = float(energies[-1])
    scf_ok = _RE_SCF_OK.findall(text)
    if scf_ok:
        parsed.scf_converged = True
        parsed.scf_cycles = int(scf_ok[-1])
    elif _RE_SCF_FAIL.search(text):
        parsed.scf_converged = False
    parsed.terminated_normally = bool(_RE_NORMAL.search(text))
    if re.search(r"ORCA finished by error", text, re.I):
        parsed.terminated_normally = False
        parsed.error = "ORCA finished by error"
    loewdin_block = _RE_LOEWDIN_BLOCK.search(text)
    if loewdin_block:
        charges = [float(x) for x in _RE_LOEWDIN_LINE.findall(loewdin_block.group(1))]
        if charges:
            parsed.lowdin_charges = charges
    else:
        # Some builds print the header without a blank-line terminator.
        alt = re.search(r"LOEWDIN ATOMIC CHARGES\s*-+\s*(.*)", text, re.I | re.S)
        if alt:
            charges = []
            for line in alt.group(1).splitlines():
                if not line.strip():
                    if charges:
                        break
                    continue
                match = _RE_LOEWDIN_LINE.match(line)
                if match:
                    charges.append(float(match.group(1)))
                elif charges:
                    break
            if charges:
                parsed.lowdin_charges = charges
    dip = _RE_DIPOLE.search(text)
    if dip:
        parsed.dipole_au = [float(dip.group(i)) for i in range(1, 4)]
    debye = _RE_DIPOLE_DEBYE.search(text)
    if debye:
        parsed.dipole_debye = float(debye.group(1))
    eps = _RE_EPS.search(text)
    if eps:
        parsed.cpcm_epsilon = float(eps.group(1))
    surface = _RE_SURFACE.search(text)
    if surface:
        parsed.cpcm_surface = surface.group(1).strip()
    feps = _RE_FEPS.search(text)
    if feps:
        parsed.cpcm_feps = feps.group(1).strip()
    lowered = text.lower()
    parsed.cpcm_smd = _solvent_flag_on(text, "smd")
    parsed.cpcm_draco = _solvent_flag_on(text, "draco")
    parsed.has_cartesian_gradient = bool(_RE_CART_GRAD.search(text))
    parsed.numerical_gradient = bool(_RE_NUM_GRAD.search(text))
    parsed.cpcm_gradient_mentioned = bool(_RE_CPCM_GRAD.search(text))
    if engrad_text:
        try:
            energy, grad = parse_engrad(engrad_text)
            parsed.energy_ha = energy
            parsed.forces_ha_bohr = grad.tolist()
            parsed.has_cartesian_gradient = True
        except OrcaCpcmError as exc:
            parsed.error = str(exc)
    hints = []
    if parsed.scf_converged is False:
        hints.append("SCF NOT CONVERGED")
    for hint in RETRYABLE_HINTS:
        if hint in lowered:
            hints.append(hint)
    for hint in INFRA_RETRYABLE_HINTS:
        if hint in lowered:
            hints.append(hint)
    parsed.raw_hints = hints
    parsed.infra_failure = any(h in lowered for h in INFRA_RETRYABLE_HINTS)
    non_retry = any(token in lowered for token in NON_RETRYABLE_HINTS)
    parsed.retryable = (not non_retry) and (
        parsed.scf_converged is False
        or any(h in lowered for h in RETRYABLE_HINTS)
        or parsed.infra_failure
    )
    if parsed.error is None and not parsed.terminated_normally:
        parsed.error = "ORCA did not print TERMINATED NORMALLY"
    if parsed.scf_converged is False:
        parsed.error = "SCF not converged"
    return parsed


def classify_retry(parsed: ParsedOrca) -> str | None:
    if parsed.terminated_normally and parsed.scf_converged and parsed.energy_ha is not None:
        return None
    if not parsed.retryable:
        return None
    lowered = " ".join(parsed.raw_hints).lower()
    if parsed.infra_failure and parsed.scf_converged is not False:
        return "InfraRetry"
    if "slowconv" in lowered:
        return "SlowConv"
    if "maxiter" in lowered or parsed.scf_converged is False:
        return "MaxIter"
    return "SlowConv"


def energy_to_ev(energy_ha: float) -> float:
    return float(energy_ha * Hartree)


def forces_ha_bohr_to_ev_a(grad_ha_bohr: Sequence[Sequence[float]]) -> np.ndarray:
    return -np.asarray(grad_ha_bohr, dtype=np.float64) * Hartree / Bohr


def finite_difference_force(energy_plus_ha: float, energy_minus_ha: float, delta_ang: float) -> float:
    """Central-difference force (eV/Å) from ± displacements in Ångström."""
    return float(-(energy_plus_ha - energy_minus_ha) * Hartree / (2.0 * float(delta_ang)))


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    return load_json(manifest_path)


def geometry_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {geom["geometry_id"]: geom for geom in manifest["geometries"]}


def pilot_job_specs(
    manifest: Mapping[str, Any],
    *,
    require_n: int | None = N_PILOT_JOBS,
) -> list[dict[str, Any]]:
    geoms = geometry_index(manifest)
    jobs = []
    for raw in manifest.get("pilot", {}).get("jobs", []):
        gid = raw["geometry_id"]
        geom = geoms[gid]
        spec = dict(raw)
        spec["geometry"] = geom
        if not is_closed_shell(geom.get("s_squared")):
            raise OrcaCpcmError(
                f"pilot geometry {gid} is open-shell (S^2={geom.get('s_squared')}); "
                "this stage is closed-shell only"
            )
        jobs.append(spec)
    if require_n is not None and len(jobs) != require_n:
        raise OrcaCpcmError(f"expected {require_n} pilot jobs, found {len(jobs)}")
    return jobs


def _fd_specs_for_index(index: int, geom: Mapping[str, Any], epsilon: Any) -> list[dict[str, Any]]:
    """Attach two Cartesian FD probes to the first vacuum job of a 6–15 atom system."""
    if index != 0 or not is_vacuum(epsilon):
        return []
    n_atoms = int(geom["n_atoms"])
    if n_atoms < 2:
        return []
    return [
        {"atom": 0, "axis": 0, "delta_ang": FD_DELTA_ANG},
        {"atom": 1, "axis": 1, "delta_ang": FD_DELTA_ANG},
    ]


def _index_width(n_jobs: int) -> int:
    return max(2, len(str(max(int(n_jobs) - 1, 0))))


def prepare_orca_jobs(
    specs: Sequence[Mapping[str, Any]],
    root: Path | str,
    *,
    nprocs: int = 32,
    nbo: bool = False,
    basis_src: Path | None = None,
    subset: str = "pilot",
    schema: str = "dc_uma_cpcm_orca_array_v1",
) -> dict[str, Any]:
    root = Path(root)
    jobs_dir = root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    width = _index_width(len(specs))
    array_jobs = []
    for index, spec in enumerate(specs):
        geom = spec["geometry"]
        if not is_closed_shell(geom.get("s_squared")):
            raise OrcaCpcmError(
                f"geometry {geom.get('geometry_id')} is open-shell (S^2={geom.get('s_squared')}); "
                "this stage is closed-shell only"
            )
        epsilon = spec["epsilon"]
        rel = Path("jobs") / job_dirname(index, epsilon, width=width)
        job_dir = root / rel
        job_dir.mkdir(parents=True, exist_ok=True)
        copy_omol_basis(job_dir, basis_src=basis_src)
        maxcore = get_mem_estimate_mb(geom["numbers"], int(geom["spin"]))
        comment = (
            f"job_id={spec['job_id']}\n"
            f"geometry_id={geom['geometry_id']}\n"
            f"charge={geom['charge']} multiplicity={geom['spin']} epsilon={epsilon_tag(epsilon)}"
        )
        text = render_orca_input(
            numbers=geom["numbers"],
            positions=geom["positions"],
            charge=int(geom["charge"]),
            multiplicity=int(geom["spin"]),
            epsilon=epsilon,
            nprocs=nprocs,
            maxcore_mb=maxcore,
            nbo=nbo,
            comment=comment,
        )
        inp_path = job_dir / INPUT_NAME
        existing_status = None
        if (job_dir / STATUS_NAME).is_file():
            existing_status = load_json(job_dir / STATUS_NAME)
        if existing_status and existing_status.get("status") == "success":
            digest = existing_status.get("input_sha256") or sha256_text(text)
        else:
            inp_path.write_text(text, encoding="utf-8")
            digest = sha256_text(text)
            (job_dir / HASH_NAME).write_text(digest + "\n", encoding="utf-8")
        fd_specs = _fd_specs_for_index(index, geom, epsilon) if subset == "pilot" else []
        meta = {
            "index": index,
            "job_id": spec["job_id"],
            "geometry_id": geom["geometry_id"],
            "parent_id": geom["parent_id"],
            "role": spec.get("role"),
            "split": spec.get("split") or geom.get("split"),
            "subset": subset,
            "epsilon": "vacuum" if is_vacuum(epsilon) else float(epsilon),
            "charge": int(geom["charge"]),
            "spin": int(geom["spin"]),
            "n_atoms": int(geom["n_atoms"]),
            "size_bin": size_bin(geom["n_atoms"]),
            "composition": geom.get("composition"),
            "n_basis": geom.get("n_basis") or get_n_basis(geom["numbers"]),
            "vacuum_energy_ev": geom.get("vacuum_energy_ev"),
            "vacuum_forces_ev_a": geom.get("vacuum_forces_ev_a"),
            "s_squared": geom.get("s_squared"),
            "unrestricted": geom.get("unrestricted"),
            "closed_shell": is_closed_shell(geom.get("s_squared")),
            "nprocs": int(nprocs),
            "maxcore_mb": int(maxcore),
            "nbo": bool(nbo),
            "input_sha256": digest,
            "finite_difference": fd_specs,
            "numbers": geom["numbers"],
            "positions": geom["positions"],
        }
        dump_json(job_dir / META_NAME, meta)
        if not existing_status or existing_status.get("status") != "success":
            dump_json(
                job_dir / STATUS_NAME,
                {
                    "job_id": spec["job_id"],
                    "index": index,
                    "status": "pending",
                    "attempt": 0,
                    "input_sha256": digest,
                },
            )
        array_jobs.append(
            {
                "index": index,
                "job_id": spec["job_id"],
                "relpath": str(rel),
                "geometry_id": geom["geometry_id"],
                "parent_id": geom["parent_id"],
                "role": spec.get("role"),
                "epsilon": meta["epsilon"],
                "n_atoms": meta["n_atoms"],
                "charge": meta["charge"],
                "finite_difference": fd_specs,
            }
        )
    array_map = {
        "schema": schema,
        "subset": subset,
        "n_jobs": len(array_jobs),
        "root": str(root),
        "jobs": array_jobs,
        "retry_policy": {
            "allowed": ["SlowConv", "MaxIter 600", "fresh guess"],
            "forbidden": ["theory", "grid", "cavity", "SMD", "DRACO"],
            "max_attempts": MAX_ATTEMPTS,
        },
    }
    dump_json(root / ARRAY_MAP_NAME, array_map)
    return array_map


def prepare_pilot_jobs(
    manifest: Mapping[str, Any],
    root: Path | str,
    *,
    nprocs: int = 32,
    nbo: bool = False,
    basis_src: Path | None = None,
    require_n: int | None = N_PILOT_JOBS,
) -> dict[str, Any]:
    specs = pilot_job_specs(manifest, require_n=require_n)
    return prepare_orca_jobs(
        specs,
        root,
        nprocs=nprocs,
        nbo=nbo,
        basis_src=basis_src,
        subset="pilot",
        schema="dc_uma_cpcm_orca_array_v1",
    )


def prepare_diagnostic_jobs(
    manifest: Mapping[str, Any],
    root: Path | str,
    *,
    nprocs: int = 32,
    nbo: bool = False,
    basis_src: Path | None = None,
    require_n: int | None = N_DIAG_JOBS,
) -> dict[str, Any]:
    from enerzyme.data.dc_uma_cpcm.diagnostic import diagnostic_job_specs

    specs = diagnostic_job_specs(manifest, require_n=require_n)
    return prepare_orca_jobs(
        specs,
        root,
        nprocs=nprocs,
        nbo=nbo,
        basis_src=basis_src,
        subset="diagnostic",
        schema="dc_uma_cpcm_orca_diagnostic_array_v1",
    )


def full_label_job_specs(
    manifest: Mapping[str, Any],
    *,
    require_n: int | None = N_TOTAL_LABELS,
) -> list[dict[str, Any]]:
    """All C-PCM label jobs from the frozen 2,000-label matrix (no vacuum)."""
    geoms = geometry_index(manifest)
    specs = []
    for lab in manifest.get("labels") or []:
        gid = lab["geometry_id"]
        if gid not in geoms:
            raise OrcaCpcmError(f"label geometry {gid} missing from manifest")
        geom = geoms[gid]
        if not is_closed_shell(geom.get("s_squared")):
            raise OrcaCpcmError(
                f"full-matrix geometry {gid} is open-shell (S^2={geom.get('s_squared')}); "
                "this stage is closed-shell only"
            )
        spec = dict(lab)
        spec["job_id"] = lab.get("label_id") or lab.get("job_id")
        if not spec["job_id"]:
            raise OrcaCpcmError(f"label missing label_id: {lab}")
        spec["geometry"] = geom
        specs.append(spec)
    if require_n is not None and len(specs) != require_n:
        raise OrcaCpcmError(f"expected {require_n} full-matrix jobs, found {len(specs)}")
    return specs


def prepare_full_jobs(
    manifest: Mapping[str, Any],
    root: Path | str,
    *,
    nprocs: int = 32,
    nbo: bool = False,
    basis_src: Path | None = None,
    require_n: int | None = N_TOTAL_LABELS,
) -> dict[str, Any]:
    specs = full_label_job_specs(manifest, require_n=require_n)
    return prepare_orca_jobs(
        specs,
        root,
        nprocs=nprocs,
        nbo=nbo,
        basis_src=basis_src,
        subset="full",
        schema="dc_uma_cpcm_orca_full_array_v1",
    )


def import_successes_from(
    dest_root: Path | str,
    src_root: Path | str,
) -> dict[str, int]:
    """Copy successful status (+ engrad/out when present) into dest by matching job_id.

    Lets the 2,000-label array skip the 240 diagnostic jobs already completed.
    """
    dest_root = Path(dest_root)
    src_root = Path(src_root)
    dest_map = load_json(dest_root / ARRAY_MAP_NAME)
    src_map = load_json(src_root / ARRAY_MAP_NAME)
    src_by_id = {item["job_id"]: item for item in src_map["jobs"]}
    n_copied = 0
    n_skipped = 0
    n_missing = 0
    for item in dest_map["jobs"]:
        src_item = src_by_id.get(item["job_id"])
        if src_item is None:
            n_missing += 1
            continue
        src_dir = src_root / src_item["relpath"]
        src_status_path = src_dir / STATUS_NAME
        if not src_status_path.is_file():
            n_missing += 1
            continue
        src_status = load_json(src_status_path)
        if src_status.get("status") != "success":
            n_skipped += 1
            continue
        dest_dir = dest_root / item["relpath"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_status_path, dest_dir / STATUS_NAME)
        for name in (OUTPUT_NAME, ENGRAD_NAME, HASH_NAME):
            src_file = src_dir / name
            if src_file.is_file():
                shutil.copy2(src_file, dest_dir / name)
        n_copied += 1
    return {
        "n_copied": n_copied,
        "n_skipped_non_success": n_skipped,
        "n_no_source": n_missing,
        "n_dest": len(dest_map["jobs"]),
        "n_src": len(src_map["jobs"]),
    }


def _archive_attempt(job_dir: Path, attempt: int) -> None:
    dest = job_dir / "attempts" / f"{int(attempt):02d}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in (INPUT_NAME, OUTPUT_NAME, ENGRAD_NAME, STATUS_NAME, HASH_NAME):
        src = job_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for leftover in job_dir.glob("orca.*"):
        if leftover.name in {INPUT_NAME}:
            continue
        if leftover.suffix in {".gbw", ".ges", ".densities", ".cpcm", ".pc", ".tmp"}:
            try:
                leftover.unlink()
            except OSError:
                pass


def rewrite_infra_retry_input(job_dir: Path, *, reason: str) -> str:
    """Re-run with a fresh guess; keep theory, grid, and cavity unchanged."""
    meta = load_json(job_dir / META_NAME)
    text = render_orca_input(
        numbers=meta["numbers"],
        positions=meta["positions"],
        charge=int(meta["charge"]),
        multiplicity=int(meta["spin"]),
        epsilon=meta["epsilon"],
        nprocs=int(meta.get("nprocs", 32)),
        maxcore_mb=int(meta.get("maxcore_mb") or get_mem_estimate_mb(meta["numbers"], int(meta["spin"]))),
        fresh_guess=True,
        nbo=bool(meta.get("nbo", False)),
        comment=f"retry reason={reason} fresh guess",
    )
    if "Guess PModel" not in text:
        raise OrcaCpcmError("infra retry rewrite failed to apply fresh guess")
    if ORCA_FUNCTIONAL not in text or "DEFGRID3" not in text:
        raise OrcaCpcmError("infra retry rewrite must not change theory or grid")
    (job_dir / INPUT_NAME).write_text(text, encoding="utf-8")
    digest = sha256_text(text)
    (job_dir / HASH_NAME).write_text(digest + "\n", encoding="utf-8")
    return digest


def rewrite_retry_input(job_dir: Path, *, reason: str) -> str:
    meta = load_json(job_dir / META_NAME)
    text = render_orca_input(
        numbers=meta["numbers"],
        positions=meta["positions"],
        charge=int(meta["charge"]),
        multiplicity=int(meta["spin"]),
        epsilon=meta["epsilon"],
        nprocs=int(meta.get("nprocs", 32)),
        maxcore_mb=int(meta.get("maxcore_mb") or get_mem_estimate_mb(meta["numbers"], int(meta["spin"]))),
        maxiter=ORCA_RETRY_MAXITER,
        slowconv=True,
        fresh_guess=True,
        nbo=bool(meta.get("nbo", False)),
        comment=f"retry reason={reason} SlowConv MaxIter {ORCA_RETRY_MAXITER} fresh guess",
    )
    if "SlowConv" not in text or f"maxiter {ORCA_RETRY_MAXITER}" not in text:
        raise OrcaCpcmError("retry rewrite failed to apply SlowConv/MaxIter 600")
    if ORCA_FUNCTIONAL not in text or "DEFGRID3" not in text:
        raise OrcaCpcmError("retry rewrite must not change theory or grid")
    (job_dir / INPUT_NAME).write_text(text, encoding="utf-8")
    digest = sha256_text(text)
    (job_dir / HASH_NAME).write_text(digest + "\n", encoding="utf-8")
    return digest


def _clear_mo_guess(job_dir: Path) -> None:
    for pattern in ("*.gbw", "*.ges", "*.uno", "*.qro"):
        for path in job_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def default_execute(job_dir: Path, orca_bin: str) -> subprocess.CompletedProcess:
    inp = job_dir / INPUT_NAME
    out = job_dir / OUTPUT_NAME
    env = os.environ.copy()
    orca_dir = str(Path(orca_bin).parent)
    env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("OMPI_MCA_btl", "^openib")
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    env.setdefault("OMPI_MCA_opal_warn_on_missing_libcuda", "0")
    with out.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            [orca_bin, inp.name],
            cwd=str(job_dir),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )


def _write_running_status(job_dir: Path, meta: Mapping[str, Any], attempt: int, digest: str) -> None:
    dump_json(
        job_dir / STATUS_NAME,
        {
            "job_id": meta["job_id"],
            "index": meta.get("index"),
            "status": "running",
            "attempt": attempt,
            "input_sha256": digest,
            "started_unix": time.time(),
        },
    )


def _finalize_status(
    job_dir: Path,
    meta: Mapping[str, Any],
    parsed: ParsedOrca,
    *,
    attempt: int,
    digest: str,
    orca_bin: str,
    wall_s: float,
    returncode: int,
    retry_reason: str | None,
) -> dict[str, Any]:
    success = bool(
        parsed.terminated_normally
        and parsed.scf_converged
        and parsed.energy_ha is not None
        and returncode == 0
    )
    if not is_vacuum(meta["epsilon"]):
        requested = float(meta["epsilon"])
        if parsed.cpcm_epsilon is None:
            success = False
            parsed.error = parsed.error or "C-PCM epsilon missing from ORCA output"
        elif abs(parsed.cpcm_epsilon - requested) > 1.0e-3:
            success = False
            parsed.error = (
                f"C-PCM epsilon {parsed.cpcm_epsilon} != requested {requested}"
            )
        if parsed.cpcm_smd or parsed.cpcm_draco:
            success = False
            parsed.error = "SMD/DRACO unexpectedly enabled"
    status = {
        "job_id": meta["job_id"],
        "index": meta.get("index"),
        "geometry_id": meta["geometry_id"],
        "epsilon": meta["epsilon"],
        "charge": meta["charge"],
        "n_atoms": meta["n_atoms"],
        "size_bin": meta.get("size_bin"),
        "status": "success" if success else "failed",
        "attempt": attempt,
        "retry_reason": retry_reason,
        "input_sha256": digest,
        "orca_bin": orca_bin,
        "orca_version": parsed.orca_version or orca_version_text(orca_bin),
        "returncode": returncode,
        "wall_s": wall_s,
        "energy_ha": parsed.energy_ha,
        "energy_ev": parsed.energy_ev,
        "forces_ev_a": parsed.forces_ev_a,
        "lowdin_charges": parsed.lowdin_charges,
        "dipole_au": parsed.dipole_au,
        "dipole_debye": parsed.dipole_debye,
        "scf_converged": parsed.scf_converged,
        "scf_cycles": parsed.scf_cycles,
        "terminated_normally": parsed.terminated_normally,
        "cpcm_epsilon": parsed.cpcm_epsilon,
        "cpcm_surface": parsed.cpcm_surface,
        "cpcm_feps": parsed.cpcm_feps,
        "cpcm_smd": parsed.cpcm_smd,
        "cpcm_draco": parsed.cpcm_draco,
        "has_cartesian_gradient": parsed.has_cartesian_gradient,
        "numerical_gradient": parsed.numerical_gradient,
        "cpcm_gradient_mentioned": parsed.cpcm_gradient_mentioned,
        "analytic_gradient": bool(parsed.has_cartesian_gradient and not parsed.numerical_gradient),
        "retryable": parsed.retryable,
        "error": None if success else parsed.error,
    }
    dump_json(job_dir / STATUS_NAME, status)
    return status


def _run_displaced_energy(
    job_dir: Path,
    meta: Mapping[str, Any],
    atom: int,
    axis: int,
    sign: int,
    delta_ang: float,
    execute: ExecuteFn,
    orca_bin: str,
) -> float:
    positions = np.asarray(meta["positions"], dtype=np.float64).copy()
    positions[int(atom), int(axis)] += float(sign) * float(delta_ang)
    work = job_dir / "fd" / f"a{atom}_{'xyz'[int(axis)]}_{'p' if sign > 0 else 'm'}"
    work.mkdir(parents=True, exist_ok=True)
    copy_omol_basis(work, basis_src=job_dir / BASIS_COPY_NAME)
    text = render_orca_input(
        numbers=meta["numbers"],
        positions=positions.tolist(),
        charge=int(meta["charge"]),
        multiplicity=int(meta["spin"]),
        epsilon=meta["epsilon"],
        nprocs=int(meta.get("nprocs", 32)),
        maxcore_mb=int(meta.get("maxcore_mb") or 1000),
        energy_only=True,
        nbo=bool(meta.get("nbo", False)),
        comment=f"finite-difference energy-only atom={atom} axis={axis} sign={sign}",
    )
    (work / INPUT_NAME).write_text(text, encoding="utf-8")
    proc = execute(work, orca_bin)
    out_text = (work / OUTPUT_NAME).read_text(encoding="utf-8", errors="replace") if (work / OUTPUT_NAME).is_file() else ""
    parsed = parse_orca_output(out_text)
    if parsed.energy_ha is None:
        raise OrcaCpcmError(
            f"FD energy missing for atom={atom} axis={axis} sign={sign} rc={proc.returncode}"
        )
    return float(parsed.energy_ha)


def run_finite_difference(
    job_dir: Path,
    status: Mapping[str, Any],
    execute: ExecuteFn,
    orca_bin: str,
) -> list[dict[str, Any]]:
    meta = load_json(job_dir / META_NAME)
    specs = meta.get("finite_difference") or []
    if not specs:
        return []
    forces = np.asarray(status.get("forces_ev_a"), dtype=np.float64)
    results = []
    for spec in specs:
        atom = int(spec["atom"])
        axis = int(spec["axis"])
        delta = float(spec.get("delta_ang", FD_DELTA_ANG))
        e_plus = _run_displaced_energy(job_dir, meta, atom, axis, +1, delta, execute, orca_bin)
        e_minus = _run_displaced_energy(job_dir, meta, atom, axis, -1, delta, execute, orca_bin)
        fd = finite_difference_force(e_plus, e_minus, delta)
        analytic = float(forces[atom, axis]) if forces.size else None
        results.append(
            {
                "atom": atom,
                "axis": axis,
                "delta_ang": delta,
                "energy_plus_ha": e_plus,
                "energy_minus_ha": e_minus,
                "force_fd_ev_a": fd,
                "force_analytic_ev_a": analytic,
                "abs_err": None if analytic is None else abs(fd - analytic),
            }
        )
    return results


def run_job(
    job_dir: Path | str,
    *,
    execute: ExecuteFn | None = None,
    orca_bin: str | None = None,
    skip_fd: bool = False,
) -> dict[str, Any]:
    job_dir = Path(job_dir)
    meta = load_json(job_dir / META_NAME)
    status_path = job_dir / STATUS_NAME
    if status_path.is_file():
        current = load_json(status_path)
        if current.get("status") == "success":
            return current
        attempt = int(current.get("attempt") or 0)
        # Slurm array resubmit after in-process attempts are exhausted.
        if current.get("status") == "failed" and attempt >= MAX_ATTEMPTS:
            attempt = 0
    else:
        current = {}
        attempt = 0
    execute = execute or default_execute
    binary = orca_bin or resolve_orca_binary()
    digest = (job_dir / HASH_NAME).read_text(encoding="utf-8").strip() if (job_dir / HASH_NAME).is_file() else sha256_file(job_dir / INPUT_NAME)

    retry_reason = None
    while attempt < MAX_ATTEMPTS:
        if attempt > 0:
            prev_parsed = parse_orca_output(
                (job_dir / OUTPUT_NAME).read_text(encoding="utf-8", errors="replace")
                if (job_dir / OUTPUT_NAME).is_file()
                else ""
            )
            retry_reason = current.get("retry_reason") or classify_retry(prev_parsed) or "SlowConv"
            _archive_attempt(job_dir, attempt)
            if retry_reason == "InfraRetry":
                digest = rewrite_infra_retry_input(job_dir, reason=retry_reason)
            else:
                digest = rewrite_retry_input(job_dir, reason=retry_reason)
            _clear_mo_guess(job_dir)
        attempt += 1
        _write_running_status(job_dir, meta, attempt, digest)
        t0 = time.time()
        proc = execute(job_dir, binary)
        wall = time.time() - t0
        out_text = (job_dir / OUTPUT_NAME).read_text(encoding="utf-8", errors="replace") if (job_dir / OUTPUT_NAME).is_file() else ""
        engrad_text = (job_dir / ENGRAD_NAME).read_text(encoding="utf-8", errors="replace") if (job_dir / ENGRAD_NAME).is_file() else None
        parsed = parse_orca_output(out_text, engrad_text)
        status = _finalize_status(
            job_dir,
            meta,
            parsed,
            attempt=attempt,
            digest=digest,
            orca_bin=binary,
            wall_s=wall,
            returncode=int(proc.returncode),
            retry_reason=retry_reason,
        )
        if status["status"] == "success":
            if not skip_fd:
                try:
                    status["finite_difference"] = run_finite_difference(job_dir, status, execute, binary)
                    dump_json(job_dir / STATUS_NAME, status)
                except OrcaCpcmError as exc:
                    status["finite_difference_error"] = str(exc)
                    dump_json(job_dir / STATUS_NAME, status)
            return status
        current = status
        if not parsed.retryable or attempt >= MAX_ATTEMPTS:
            return status
        current["retry_reason"] = classify_retry(parsed)
    return current


def _fill_s_squared_on_meta(meta: dict[str, Any], src: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Best-effort S² for already-written job meta that predates the closed-shell field."""
    out = dict(meta)
    if out.get("s_squared") is not None:
        out["closed_shell"] = is_closed_shell(out.get("s_squared"))
        return out
    if not src:
        return out
    gid = str(out.get("geometry_id") or "")
    parts = gid.split(":")
    if len(parts) < 3:
        return out
    path = Path(src) / parts[1]
    if not path.is_file():
        return out
    try:
        from enerzyme.data.dc_uma_cpcm.index import open_aselmdb

        db = open_aselmdb(str(path))
        try:
            row = db._get_row(int(parts[2]))
            payload = row.data if isinstance(row.data, dict) else {}
            s2 = parse_s_squared(payload.get("s_squared"))
            if s2 is not None:
                out["s_squared"] = s2
                out["unrestricted"] = bool(payload.get("unrestricted"))
                out["closed_shell"] = is_closed_shell(s2)
        finally:
            db.close()
    except Exception:
        return out
    return out


def compare_vacuum_to_omol(status: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any] | None:
    if not is_vacuum(meta.get("epsilon", status.get("epsilon"))):
        return None
    hist_e = meta.get("vacuum_energy_ev")
    hist_f = meta.get("vacuum_forces_ev_a")
    fresh_e = status.get("energy_ev")
    fresh_f = status.get("forces_ev_a")
    if hist_e is None or fresh_e is None or hist_f is None or fresh_f is None:
        return {"error": "missing vacuum energy/forces for comparison"}
    hist_f_arr = np.asarray(hist_f, dtype=np.float64)
    fresh_f_arr = np.asarray(fresh_f, dtype=np.float64)
    dE = float(fresh_e) - float(hist_e)
    dF = fresh_f_arr - hist_f_arr
    s2 = parse_s_squared(meta.get("s_squared"))
    if s2 is not None:
        closed = is_closed_shell(s2)
    elif meta.get("closed_shell") is False:
        closed = False
    else:
        closed = True
    return {
        "geometry_id": meta.get("geometry_id"),
        "omol_energy_ev": float(hist_e),
        "fresh_energy_ev": float(fresh_e),
        "delta_energy_ev": dE,
        "delta_energy_ha": dE / float(Hartree),
        "force_rmse_ev_a": float(np.sqrt(np.mean(dF * dF))),
        "force_max_ev_a": float(np.max(np.abs(dF))),
        "s_squared": s2,
        "closed_shell": closed,
        "possible_state_change": abs(dE) > VACUUM_STATE_CHANGE_EV,
        "above_recompute_noise": abs(dE) / float(Hartree) > VACUUM_NOISE_HA
        or float(np.sqrt(np.mean(dF * dF))) > FORCE_NOISE_EV_A,
        "inconsistent_with_omol_history": abs(dE) > VACUUM_PAIR_ENERGY_EV
        or float(np.sqrt(np.mean(dF * dF))) > VACUUM_PAIR_FORCE_RMSE_EV_A,
    }


def vacuum_gate_from_comparisons(vacuum_cmp: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Closed-shell-only vacuum recompute / state-change decisions."""
    closed_cmp = [row for row in vacuum_cmp if row.get("closed_shell", True) and "error" not in row]
    open_cmp = [row for row in vacuum_cmp if row.get("closed_shell") is False]
    signed_e = [row["delta_energy_ha"] for row in closed_cmp if "delta_energy_ha" in row]
    systematic = False
    if signed_e:
        arr = np.asarray(signed_e, dtype=np.float64)
        systematic = bool(np.all(np.sign(arr) == np.sign(arr[0])) and abs(float(arr.mean())) > VACUUM_NOISE_HA)
    state_change = any(row.get("possible_state_change") for row in closed_cmp)
    inconsistent_hist = [row for row in closed_cmp if row.get("inconsistent_with_omol_history")]
    recommend = systematic or state_change or bool(inconsistent_hist)
    return {
        "vacuum_vs_omol_closed_shell": list(closed_cmp),
        "vacuum_vs_omol_open_shell": list(open_cmp),
        "vacuum_systematic_bias": systematic,
        "possible_electronic_state_switch": state_change,
        "vacuum_inconsistent_with_omol_history": inconsistent_hist,
        "recommend_recompute_all_vacuum": recommend,
        "use_fresh_vacuum_for_residuals": recommend,
    }


def summarize_pilot(
    root: Path | str,
    *,
    report_path: Path | str | None = None,
    write_default_report: bool = True,
    omol_src: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    array_map = load_json(root / ARRAY_MAP_NAME)
    rows = []
    vacuum_cmp = []
    fd_rows = []
    n_success = 0
    n_first_pass = 0
    by_charge: dict[str, list[bool]] = {}
    by_eps: dict[str, list[bool]] = {}
    by_size: dict[str, list[bool]] = {}
    for item in array_map["jobs"]:
        job_dir = root / item["relpath"]
        status = load_json(job_dir / STATUS_NAME) if (job_dir / STATUS_NAME).is_file() else {"status": "missing"}
        meta = _fill_s_squared_on_meta(load_json(job_dir / META_NAME), omol_src)
        ok = status.get("status") == "success"
        first = ok and int(status.get("attempt") or 0) <= 1
        n_success += int(ok)
        n_first_pass += int(first)
        charge_key = str(meta.get("charge"))
        eps_key = epsilon_tag(meta.get("epsilon"))
        size_key = str(meta.get("size_bin"))
        by_charge.setdefault(charge_key, []).append(ok)
        by_eps.setdefault(eps_key, []).append(ok)
        by_size.setdefault(size_key, []).append(ok)
        cmp = compare_vacuum_to_omol(status, meta)
        if cmp is not None:
            vacuum_cmp.append(cmp)
        if status.get("finite_difference"):
            fd_rows.extend(status["finite_difference"])
        rows.append(
            {
                "index": item["index"],
                "job_id": item["job_id"],
                "status": status.get("status"),
                "attempt": status.get("attempt"),
                "epsilon": meta.get("epsilon"),
                "charge": meta.get("charge"),
                "n_atoms": meta.get("n_atoms"),
                "energy_ev": status.get("energy_ev"),
                "scf_converged": status.get("scf_converged"),
                "cpcm_epsilon": status.get("cpcm_epsilon"),
                "analytic_gradient": status.get("analytic_gradient"),
                "lowdin_n": None if status.get("lowdin_charges") is None else len(status["lowdin_charges"]),
                "dipole_au": status.get("dipole_au"),
                "orca_version": status.get("orca_version"),
                "error": status.get("error"),
                "wall_s": status.get("wall_s"),
            }
        )
    n_jobs = len(array_map["jobs"])
    first_pass_rate = n_first_pass / n_jobs if n_jobs else 0.0
    success_rate = n_success / n_jobs if n_jobs else 0.0

    def _rate(flags: list[bool]) -> float:
        return float(sum(flags) / len(flags)) if flags else 0.0

    gate = vacuum_gate_from_comparisons(vacuum_cmp)
    closed_cmp = gate["vacuum_vs_omol_closed_shell"]
    open_cmp = gate["vacuum_vs_omol_open_shell"]
    systematic = gate["vacuum_systematic_bias"]
    state_change = gate["possible_electronic_state_switch"]
    inconsistent_hist = gate["vacuum_inconsistent_with_omol_history"]
    recommend_recompute_vacuum = gate["recommend_recompute_all_vacuum"]
    fd_ok = True
    if fd_rows:
        fd_ok = all(item.get("abs_err") is not None and item["abs_err"] < 2.0e-2 for item in fd_rows)
    stop_expand = first_pass_rate < FIRST_PASS_MIN or state_change
    rates = {
        "by_charge": {k: _rate(v) for k, v in sorted(by_charge.items())},
        "by_epsilon": {k: _rate(v) for k, v in sorted(by_eps.items())},
        "by_size": {k: _rate(v) for k, v in sorted(by_size.items())},
    }
    stratified_gap = 0.0
    all_rates = [r for group in rates.values() for r in group.values()]
    if all_rates:
        stratified_gap = float(max(all_rates) - min(all_rates))
    if stratified_gap > 0.25:
        stop_expand = True
    report = {
        "schema": "dc_uma_cpcm_orca_pilot_v1",
        "root": str(root),
        "n_jobs": n_jobs,
        "n_success": n_success,
        "n_first_pass": n_first_pass,
        "success_rate": success_rate,
        "first_pass_rate": first_pass_rate,
        "first_pass_min": FIRST_PASS_MIN,
        "failure_rates": rates,
        "stratified_success_gap": stratified_gap,
        "jobs": rows,
        "vacuum_vs_omol": vacuum_cmp,
        "vacuum_vs_omol_closed_shell": closed_cmp,
        "vacuum_vs_omol_open_shell": open_cmp,
        "vacuum_systematic_bias": systematic,
        "possible_electronic_state_switch": state_change,
        "vacuum_inconsistent_with_omol_history": inconsistent_hist,
        "recommend_recompute_all_vacuum": recommend_recompute_vacuum,
        "use_fresh_vacuum_for_residuals": bool(inconsistent_hist) or recommend_recompute_vacuum,
        "finite_difference": fd_rows,
        "finite_difference_ok": fd_ok,
        "stop_expand": stop_expand,
        "notes": [
            "Reports may only call held-out parents adapter-unseen parent.",
            "Vacuum jobs omit %cpcm; they are not C-PCM with epsilon=1.",
            "Retries allowed: SlowConv, MaxIter 600, fresh guess only.",
            "Vacuum gates use closed-shell (S^2 <= 0.05) comparisons only; open-shell rows are recorded but ignored.",
        ],
    }
    dump_json(root / "summary.json", report)
    if report_path is not None:
        dump_json(report_path, report)
    elif write_default_report:
        reports_dir = Path(DEFAULT_EXPERIMENTS) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        dump_json(reports_dir / "orca_pilot.json", report)
    return report


def summarize_diagnostic(
    root: Path | str,
    *,
    report_path: Path | str | None = None,
    write_default_report: bool = True,
    omol_src: Path | str | None = None,
) -> dict[str, Any]:
    report = summarize_pilot(
        root, report_path=None, write_default_report=False, omol_src=omol_src
    )
    report["schema"] = "dc_uma_cpcm_orca_diagnostic_v1"
    report["subset"] = "diagnostic"
    report["notes"] = [
        "240-job diagnostic C-PCM labels (16 train parents × 4 ε + 8 val parents × 7 ε).",
        "Vacuum residuals use OMol historical labels only if the 24-job closed-shell vacuum gate passed.",
        "This stage excludes open-shell molecules from diagnostic and production labels.",
        "Reports may only call held-out parents adapter-unseen parent.",
        "Retries allowed: SlowConv, MaxIter 600, fresh guess only.",
    ]
    root = Path(root)
    dump_json(root / "summary.json", report)
    if report_path is not None:
        dump_json(report_path, report)
    elif write_default_report:
        reports_dir = Path(DEFAULT_EXPERIMENTS) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        dump_json(reports_dir / "orca_diagnostic.json", report)
    return report


def summarize_full(
    root: Path | str,
    *,
    report_path: Path | str | None = None,
    write_default_report: bool = True,
    omol_src: Path | str | None = None,
) -> dict[str, Any]:
    report = summarize_pilot(
        root, report_path=None, write_default_report=False, omol_src=omol_src
    )
    report["schema"] = "dc_uma_cpcm_orca_full_v1"
    report["subset"] = "full"
    report["notes"] = [
        "2,000-label C-PCM matrix (train/probe/validation/test; no vacuum jobs).",
        "This stage excludes open-shell molecules from diagnostic and production labels.",
        "Reports may only call held-out parents adapter-unseen parent.",
        "Retries allowed: SlowConv, MaxIter 600, fresh guess only.",
    ]
    root = Path(root)
    dump_json(root / "summary.json", report)
    if report_path is not None:
        dump_json(report_path, report)
    elif write_default_report:
        reports_dir = Path(DEFAULT_EXPERIMENTS) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        dump_json(reports_dir / "orca_full.json", report)
    return report
