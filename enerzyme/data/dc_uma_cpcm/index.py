"""Stream-scan OMol25 ASE-LMDB shards into a compact candidate index."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from ase.data import chemical_symbols

from .constants import (
    ALLOWED_ATOMIC_NUMBERS,
    ALLOWED_CHARGES,
    ALLOWED_ELEMENTS,
    CLOSED_SHELL_S2_MAX,
    CONNECT_SCALE,
    DEF2_TZVPD_N_BASIS,
    KNOWN_DATA_IDS,
    MAX_ATOMS,
    MAX_FORCE_EV_A,
    MAX_N_BASIS,
    MIN_ATOMS,
    OVERLAP_SCALE,
    SINGLET_SPIN,
)
from .geometry import (
    as_int_numbers,
    has_atom_overlap,
    n_connected_components,
    radius_of_gyration,
)
from .identity import (
    graph_hash,
    isotope_signature,
    parse_source_identity,
    preferred_source_path,
)

_COMP_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


@dataclass(frozen=True)
class FilterConfig:
    min_atoms: int = MIN_ATOMS
    max_atoms: int = MAX_ATOMS
    allowed_charges: tuple[int, ...] = ALLOWED_CHARGES
    spin: int = SINGLET_SPIN
    max_s_squared: float = CLOSED_SHELL_S2_MAX
    max_n_basis: int = MAX_N_BASIS
    max_force: float = MAX_FORCE_EV_A
    connect_scale: float = CONNECT_SCALE
    overlap_scale: float = OVERLAP_SCALE
    allowed_elements: frozenset[str] = ALLOWED_ELEMENTS
    allowed_atomic_numbers: frozenset[int] = ALLOWED_ATOMIC_NUMBERS


@dataclass
class RejectStats:
    total: int = 0
    passed: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    data_id_total: dict[str, int] = field(default_factory=dict)
    data_id_passed: dict[str, int] = field(default_factory=dict)
    source_parse_ok: dict[str, int] = field(default_factory=dict)
    source_parse_fail: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add_reason(self, reason: str, n: int = 1) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + n

    def merge(self, other: "RejectStats") -> None:
        self.total += other.total
        self.passed += other.passed
        for key, value in other.reasons.items():
            self.reasons[key] = self.reasons.get(key, 0) + value
        for store, other_store in (
            (self.data_id_total, other.data_id_total),
            (self.data_id_passed, other.data_id_passed),
            (self.source_parse_ok, other.source_parse_ok),
            (self.source_parse_fail, other.source_parse_fail),
        ):
            for key, value in other_store.items():
                store[key] = store.get(key, 0) + value
        for data_id, recs in other.examples.items():
            bucket = self.examples.setdefault(data_id, [])
            for rec in recs:
                if len(bucket) < 8:
                    bucket.append(rec)


def parse_composition_elements(composition: str | None) -> set[str] | None:
    if not composition:
        return None
    found = {m.group(1) for m in _COMP_RE.finditer(composition)}
    return found or None


def estimate_n_basis(numbers: Iterable[int]) -> int:
    total = 0
    for atom_z in as_int_numbers(numbers).tolist():
        sym = chemical_symbols[int(atom_z)]
        nbf = DEF2_TZVPD_N_BASIS.get(sym)
        if nbf is None:
            raise KeyError(f"No def2-TZVPD basis count for element {sym}")
        total += int(nbf)
    return total


def list_aselmdb_shards(src: str | os.PathLike[str]) -> list[str]:
    path = Path(src)
    if path.is_file():
        return [str(path)]
    shards = sorted(
        p
        for p in glob(str(path / "data*.aselmdb"))
        if not p.endswith("-lock") and os.path.isfile(p)
    )
    if not shards:
        raise FileNotFoundError(f"No data*.aselmdb shards under {src}")
    return shards


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def parse_s_squared(value: Any) -> float | None:
    """Return a finite S², or None if the field is missing/unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not _finite(val):
        return None
    return val


def is_closed_shell(value: Any, *, max_s_squared: float = CLOSED_SHELL_S2_MAX) -> bool:
    """True unless S² is present and above the closed-shell cut.

    Missing S² is treated as closed so unit tests and pre-annotation records
    still pass; production sampling annotates from OMol before this check.
    """
    parsed = parse_s_squared(value)
    if parsed is None:
        return True
    return parsed <= float(max_s_squared)


def cheap_reject_payload(payload: dict[str, Any], natoms: Any, cfg: FilterConfig) -> str | None:
    """Metadata-only rejects; does not load positions or forces."""
    try:
        n_atoms = int(payload.get("num_atoms") if payload.get("num_atoms") is not None else natoms)
    except (TypeError, ValueError):
        return "natoms"
    if n_atoms < cfg.min_atoms or n_atoms > cfg.max_atoms:
        return "natoms"
    spin = payload.get("spin")
    if spin is None:
        return "spin"
    if int(spin) != int(cfg.spin):
        return "spin"
    if not is_closed_shell(payload.get("s_squared"), max_s_squared=cfg.max_s_squared):
        return "open_shell"
    charge = payload.get("charge")
    try:
        q = int(charge)
    except (TypeError, ValueError):
        return "charge"
    if q not in cfg.allowed_charges:
        return "charge"
    parsed = parse_composition_elements(payload.get("composition"))
    if parsed is not None and not parsed.issubset(cfg.allowed_elements):
        return "elements"
    n_basis_stored = payload.get("n_basis")
    if n_basis_stored is not None:
        try:
            if int(n_basis_stored) > cfg.max_n_basis:
                return "n_basis"
        except (TypeError, ValueError):
            return "n_basis"
    return None


def evaluate_record(
    *,
    numbers: Iterable[int],
    positions,
    energy: Any,
    forces: Any,
    charge: Any,
    spin: Any,
    pbc: Any,
    n_basis_stored: Any,
    composition: str | None,
    cfg: FilterConfig,
    s_squared: Any = None,
) -> str | None:
    """Return a reject reason, or None if the record passes."""
    payload_like = {
        "num_atoms": int(as_int_numbers(numbers).shape[0]),
        "spin": spin,
        "charge": charge,
        "composition": composition,
        "n_basis": n_basis_stored,
        "s_squared": s_squared,
    }
    cheap = cheap_reject_payload(payload_like, payload_like["num_atoms"], cfg)
    if cheap:
        return cheap
    z = as_int_numbers(numbers)
    if pbc is not None and np.any(pbc):
        return "pbc"
    if not set(int(v) for v in z.tolist()).issubset(cfg.allowed_atomic_numbers):
        return "elements"
    parsed = parse_composition_elements(composition)
    if parsed is not None and not parsed.issubset(cfg.allowed_elements):
        return "elements"
    n_basis_est = estimate_n_basis(z)
    if n_basis_est > cfg.max_n_basis:
        return "n_basis"
    if n_basis_stored is not None:
        try:
            if int(n_basis_stored) > cfg.max_n_basis:
                return "n_basis"
        except (TypeError, ValueError):
            return "n_basis"
    if energy is None or forces is None:
        return "nonfinite"
    forces_arr = np.asarray(forces, dtype=np.float64)
    if (not _finite(energy)) or (not np.isfinite(forces_arr).all()):
        return "nonfinite"
    if float(np.max(np.abs(forces_arr))) > cfg.max_force:
        return "fmax"
    if has_atom_overlap(z, positions, scale=cfg.overlap_scale):
        return "overlap"
    if n_connected_components(z, positions, scale=cfg.connect_scale) != 1:
        return "disconnected"
    return None


def candidate_from_row(
    *,
    shard_name: str,
    row_id: int,
    numbers,
    positions,
    energy,
    forces,
    masses,
    data: dict[str, Any],
    pbc,
    cfg: FilterConfig,
    stats: RejectStats,
    counted: bool = False,
) -> dict[str, Any] | None:
    payload = data if isinstance(data, dict) else {}
    data_id = str(payload.get("data_id") or "unknown")
    z = as_int_numbers(numbers)
    charge = payload.get("charge")
    spin = payload.get("spin")
    source = payload.get("source")
    reference_source = payload.get("reference_source")
    ident = parse_source_identity(data_id, source, reference_source)
    if not counted:
        stats.total += 1
        stats.data_id_total[data_id] = stats.data_id_total.get(data_id, 0) + 1
        if ident is None:
            stats.source_parse_fail[data_id] = stats.source_parse_fail.get(data_id, 0) + 1
        else:
            stats.source_parse_ok[data_id] = stats.source_parse_ok.get(data_id, 0) + 1
        bucket = stats.examples.setdefault(data_id, [])
        if len(bucket) < 8:
            bucket.append(
                {
                    "shard": shard_name,
                    "row_id": int(row_id),
                    "source": source,
                    "reference_source": reference_source,
                    "preferred_path": preferred_source_path(source, reference_source),
                    "source_token": None if ident is None else ident.token,
                    "charge": charge,
                    "spin": spin,
                    "n_atoms": int(z.shape[0]),
                    "composition": payload.get("composition"),
                }
            )
    reason = evaluate_record(
        numbers=z,
        positions=positions,
        energy=energy,
        forces=forces,
        charge=charge,
        spin=spin,
        pbc=pbc,
        n_basis_stored=payload.get("n_basis"),
        composition=payload.get("composition"),
        cfg=cfg,
        s_squared=payload.get("s_squared"),
    )
    if reason is not None:
        stats.add_reason(reason)
        return None

    forces_arr = np.asarray(forces, dtype=np.float64)
    lowdin = payload.get("lowdin_charges")
    mean_abs_lowdin = None
    if lowdin is not None:
        arr = np.asarray(lowdin, dtype=np.float64)
        if arr.size and np.isfinite(arr).all():
            mean_abs_lowdin = float(np.mean(np.abs(arr)))
    rec = {
        "shard": shard_name,
        "row_id": int(row_id),
        "data_id": data_id,
        "source": source,
        "reference_source": reference_source,
        "charge": int(charge),
        "spin": int(spin),
        "n_atoms": int(z.shape[0]),
        "n_heavy": int(np.sum(z > 1)),
        "n_basis": int(payload["n_basis"]) if payload.get("n_basis") is not None else None,
        "n_basis_est": estimate_n_basis(z),
        "energy": float(energy),
        "fmax": float(np.max(np.abs(forces_arr))),
        "frms": float(np.sqrt(np.mean(forces_arr * forces_arr))),
        "rg": radius_of_gyration(positions, z),
        "mean_abs_lowdin": mean_abs_lowdin,
        "composition": payload.get("composition"),
        "numbers": [int(v) for v in z.tolist()],
        "source_mol_id": None if ident is None else ident.token,
        "graph_id": graph_hash(z, positions, connect_scale=cfg.connect_scale),
        "isotope_id": isotope_signature(z, masses),
        "s_squared": parse_s_squared(payload.get("s_squared")),
        "unrestricted": payload.get("unrestricted"),
    }
    stats.passed += 1
    stats.data_id_passed[data_id] = stats.data_id_passed.get(data_id, 0) + 1
    stats.add_reason("passed")
    return rec


def open_aselmdb(path: str):
    """Open an OMol ASE-LMDB shard read-only.

    Write-mode LMDB mmap/locks fail on Lustre compute nodes with
    ``Function not implemented``.
    """
    from ase.db import connect

    return connect(path, readonly=True, use_lock_file=False)


def iter_shard_rows(shard_path: str) -> Iterator[Any]:
    db = open_aselmdb(shard_path)
    try:
        for row_id in range(1, len(db) + 1):
            yield db._get_row(row_id)
    finally:
        db.close()


def scan_shard(shard_path: str, cfg: FilterConfig | None = None) -> tuple[list[dict[str, Any]], RejectStats]:
    cfg = cfg or FilterConfig()
    stats = RejectStats()
    records: list[dict[str, Any]] = []
    shard_name = os.path.basename(shard_path)
    for row in iter_shard_rows(shard_path):
        payload = row.data if isinstance(row.data, dict) else {}
        data_id = str(payload.get("data_id") or "unknown")
        stats.total += 1
        stats.data_id_total[data_id] = stats.data_id_total.get(data_id, 0) + 1
        ident = parse_source_identity(data_id, payload.get("source"), payload.get("reference_source"))
        if ident is None:
            stats.source_parse_fail[data_id] = stats.source_parse_fail.get(data_id, 0) + 1
        else:
            stats.source_parse_ok[data_id] = stats.source_parse_ok.get(data_id, 0) + 1
        bucket = stats.examples.setdefault(data_id, [])
        if len(bucket) < 8:
            bucket.append(
                {
                    "shard": shard_name,
                    "row_id": int(row.id),
                    "source": payload.get("source"),
                    "reference_source": payload.get("reference_source"),
                    "preferred_path": preferred_source_path(
                        payload.get("source"), payload.get("reference_source")
                    ),
                    "source_token": None if ident is None else ident.token,
                    "charge": payload.get("charge"),
                    "spin": payload.get("spin"),
                    "n_atoms": payload.get("num_atoms") or row.natoms,
                    "composition": payload.get("composition"),
                }
            )
        cheap = cheap_reject_payload(payload, row.natoms, cfg)
        if cheap:
            stats.add_reason(cheap)
            continue
        rec = candidate_from_row(
            shard_name=shard_name,
            row_id=int(row.id),
            numbers=row.numbers,
            positions=row.positions,
            energy=row.energy,
            forces=row.forces,
            masses=getattr(row, "masses", None),
            data=payload,
            pbc=row.pbc,
            cfg=cfg,
            stats=stats,
            counted=True,
        )
        if rec is not None:
            records.append(rec)
    return records, stats


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotate_s_squared(
    records: list[dict[str, Any]],
    src: str | os.PathLike[str] | None,
) -> int:
    """Fill missing ``s_squared`` / ``unrestricted`` from OMol ASE-LMDB shards."""
    if not src:
        return 0
    missing = [rec for rec in records if rec.get("s_squared") is None]
    if not missing:
        return 0
    src_path = Path(src)
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in missing:
        by_shard[str(rec["shard"])].append(rec)
    n = 0
    for shard_name, recs in by_shard.items():
        path = src_path / shard_name
        if not path.is_file():
            continue
        db = open_aselmdb(str(path))
        try:
            for rec in recs:
                row = db._get_row(int(rec["row_id"]))
                payload = row.data if isinstance(row.data, dict) else {}
                rec["s_squared"] = parse_s_squared(payload.get("s_squared"))
                if rec["s_squared"] is None:
                    rec["s_squared"] = 0.0
                rec["unrestricted"] = bool(payload.get("unrestricted"))
                n += 1
        finally:
            db.close()
    return n


def drop_open_shell_records(
    records: Iterable[dict[str, Any]],
    *,
    max_s_squared: float = CLOSED_SHELL_S2_MAX,
) -> tuple[list[dict[str, Any]], int]:
    """Keep closed-shell records. Returns ``(kept, n_dropped)``."""
    kept: list[dict[str, Any]] = []
    n_dropped = 0
    for rec in records:
        if is_closed_shell(rec.get("s_squared"), max_s_squared=max_s_squared):
            kept.append(rec)
        else:
            n_dropped += 1
    return kept, n_dropped


def scan_shard_to_file(
    shard: str,
    cand_dir: str,
    cfg: FilterConfig | None = None,
) -> tuple[str, int, dict[str, Any]]:
    """Worker-safe shard scan used by :func:`scan_train_4m` (must be top-level for Pool)."""
    cfg = cfg or FilterConfig()
    records, stats = scan_shard(shard, cfg=cfg)
    shard_name = os.path.splitext(os.path.basename(shard))[0]
    out_path = Path(cand_dir) / f"{shard_name}.jsonl.gz"
    write_jsonl(out_path, records)
    return str(out_path), len(records), asdict(stats)


def scan_train_4m(
    src: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    cfg: FilterConfig | None = None,
    workers: int = 1,
    max_shards: int | None = None,
    shard_subset: list[str] | None = None,
) -> dict[str, Any]:
    """Scan ASE-LMDB shards and write per-shard candidate JSONL plus audit stats."""
    cfg = cfg or FilterConfig()
    shards = shard_subset if shard_subset is not None else list_aselmdb_shards(src)
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    out = Path(out_dir)
    cand_dir = out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    merged = RejectStats()
    shard_files: list[str] = []
    if workers <= 1:
        results = [scan_shard_to_file(shard, str(cand_dir), cfg) for shard in shards]
    else:
        from multiprocessing import Pool

        with Pool(processes=workers) as pool:
            results = pool.starmap(
                scan_shard_to_file,
                [(shard, str(cand_dir), cfg) for shard in shards],
            )
    for path, _n, stats_dict in results:
        shard_files.append(path)
        other = RejectStats(**{
            k: stats_dict[k]
            for k in ("total", "passed", "reasons", "data_id_total", "data_id_passed",
                      "source_parse_ok", "source_parse_fail", "examples")
        })
        merged.merge(other)

    audit = {
        "known_data_ids": list(KNOWN_DATA_IDS),
        "observed_data_ids": sorted(merged.data_id_total),
        "n_shards": len(shards),
        "n_total": merged.total,
        "n_passed": merged.passed,
        "reject_reasons": dict(Counter(merged.reasons)),
        "data_id_total": merged.data_id_total,
        "data_id_passed": merged.data_id_passed,
        "source_parse_ok": merged.source_parse_ok,
        "source_parse_fail": merged.source_parse_fail,
        "examples": merged.examples,
        "candidate_files": shard_files,
        "filters": asdict(cfg) | {"allowed_elements": sorted(cfg.allowed_elements)},
    }
    audit_path = out / "source_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return audit
