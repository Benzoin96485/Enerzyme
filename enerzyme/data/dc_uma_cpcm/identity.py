"""Parent identity: source-path molecule IDs plus covalent Weisfeiler–Lehman graphs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase.data import atomic_masses, chemical_symbols

from .constants import CONNECT_SCALE, WL_ITERATIONS
from .geometry import adjacency_matrix, as_int_numbers, as_positions

_JOB_HASH = re.compile(r"(?:^|/)job_[0-9]+_[0-9a-f]+(?:/|$)", re.I)
_CHEMBL = re.compile(r"(CHEMBL\d+)", re.I)
_ZINC = re.compile(r"(ZINC[0-9]+)", re.I)
_ANI2X = re.compile(r"ani2x_(\d+)_(\d+)_(-?\d+)_(\d+)")
_ANIBB = re.compile(r"aniBB_(\d+)_(\d+)_(-?\d+)_(\d+)", re.I)
_GEOM = re.compile(r"geom_(\d+)_(-?\d+)_(\d+)")
_SPF = re.compile(r"spf_(\d+)_(-?\d+)_(\d+)")
_RGD = re.compile(r"MR_(\d+)_(\d+)_(\d+)_(-?\d+)_(\d+)")
_QM7B = re.compile(r"orbnet_qm7b_(\d+)_(-?\d+)_")
_SMALL_CAN = re.compile(r"orbnet_small_can_(\d+)_")
_PROTOMER = re.compile(r"(CHEMBL\d+)_protomer_(\d+)", re.I)
_SPICE_PUBCHEM = re.compile(r"SPICE_PubChem.*?_spice_(\d+)_(\d+)_(-?\d+)_(\d+)")
_SPICE_AAL = re.compile(r"SPICE_Amino_Acid_Ligand.*?_spice_([^/]+)_(\d+)_(-?\d+)_(\d+)")
_SPICE_DES = re.compile(r"SPICE_DES370K")
_ELYTE_UNSOLV = re.compile(r"outputs_unsolvated[^/]*/([^/]+)_mol(\d+)_(-?\d+)_(\d+)")
_ELYTE_SOLV = re.compile(r"([^/]+)_mol(\d+)_(?:solv\d+|dimer\d+)_(-?\d+)_(\d+)")
_PDB_LIGSTATE = re.compile(r"ligstate(\d+)")
_CORE = re.compile(r"([0-9a-z]+)_core([A-Za-z0-9]+)_(-?\d+)_(\d+)", re.I)
_FRAME = re.compile(r"_frame\d+", re.I)


@dataclass(frozen=True)
class SourceIdentity:
    kind: str
    mol_id: str
    extra: str = ""
    raw: str = ""

    @property
    def token(self) -> str:
        if self.extra:
            return f"{self.kind}:{self.mol_id}:{self.extra}"
        return f"{self.kind}:{self.mol_id}"


def _strip_job_path(path: str) -> str:
    text = (path or "").strip()
    text = text.replace("s3://", "")
    text = re.sub(r"/orca\.tar\.zst$", "", text)
    return text


def preferred_source_path(source: str | None, reference_source: str | None) -> str:
    """Prefer ``reference_source`` when ``source`` is a restart/job-hash path."""
    src = source or ""
    ref = reference_source or ""
    if ref and (_JOB_HASH.search(src) or "restart" in src.lower()):
        return ref
    return src or ref


def _leaf(path: str) -> str:
    parts = [p for p in _strip_job_path(path).split("/") if p]
    return parts[-1] if parts else ""


def _parse_ani2x(path: str) -> SourceIdentity | None:
    m = _ANI2X.search(path)
    if not m:
        return None
    return SourceIdentity("ani2x", m.group(2), raw=m.group(0))


def _parse_anibb(path: str) -> SourceIdentity | None:
    m = _ANIBB.search(path)
    if not m:
        return None
    return SourceIdentity("anibb", m.group(2), raw=m.group(0))


def _parse_geom(path: str) -> SourceIdentity | None:
    m = _GEOM.search(path)
    if not m:
        return None
    return SourceIdentity("geom", m.group(1), raw=m.group(0))


def _parse_orbnet(path: str) -> SourceIdentity | None:
    if "saltcomplex" in path:
        # Multi-component salts: connectivity filter should reject; no molecule ID.
        return None
    m = _PROTOMER.search(path)
    if m:
        return SourceIdentity("chembl", m.group(1).upper(), extra=f"protomer{m.group(2)}", raw=m.group(0))
    m = _CHEMBL.search(path)
    if m:
        return SourceIdentity("chembl", m.group(1).upper(), raw=m.group(0))
    m = _QM7B.search(path)
    if m:
        return SourceIdentity("qm7b", m.group(1), extra=f"q{m.group(2)}", raw=m.group(0))
    m = _SMALL_CAN.search(path)
    if m:
        return SourceIdentity("small_can", m.group(1), raw=m.group(0))
    return None


def _parse_spice(path: str) -> SourceIdentity | None:
    m = _SPICE_PUBCHEM.search(path)
    if m:
        return SourceIdentity("spice_pubchem", m.group(1), raw=m.group(0))
    m = _SPICE_AAL.search(path)
    if m:
        lig_aa = m.group(1)
        return SourceIdentity("spice_aal", lig_aa, raw=m.group(0))
    if _SPICE_DES.search(path):
        # Dimers / complexes: leave identity to the bond graph (usually rejected).
        return None
    return None


def _parse_biomolecule(path: str) -> SourceIdentity | None:
    m = _SPF.search(path)
    if m:
        return SourceIdentity("spf", m.group(1), raw=m.group(0))
    m = _CORE.search(_leaf(path))
    if m:
        return SourceIdentity("protcore", f"{m.group(1).lower()}:{m.group(2)}", raw=m.group(0))
    lig = _CHEMBL.search(path) or _ZINC.search(path)
    ligstate = _PDB_LIGSTATE.search(path)
    if lig:
        extra = f"ligstate{ligstate.group(1)}" if ligstate else ""
        kind = "chembl" if lig.lastindex is not None and lig.re is _CHEMBL else "zinc"
        if lig.re is _ZINC:
            kind = "zinc"
        elif lig.re is _CHEMBL:
            kind = "chembl"
        return SourceIdentity(kind, lig.group(1).upper(), extra=extra, raw=path)
    leaf = _FRAME.sub("", _leaf(path))
    if path.startswith("rna/") or "/rna/" in path or leaf.startswith("rna"):
        return SourceIdentity("rna", leaf, raw=path)
    if path.startswith("dna/") or "/dna/" in path:
        return SourceIdentity("dna", leaf, raw=path)
    if "pdb_pockets" in path:
        return SourceIdentity("pdbpocket", leaf, raw=path)
    if "pdb_fragments" in path:
        return SourceIdentity("pdbfrag", leaf, raw=path)
    if "torsion_profiles" in path:
        return SourceIdentity("torsion", leaf, raw=path)
    return None


def _parse_elyte(path: str) -> SourceIdentity | None:
    m = _ELYTE_UNSOLV.search(path)
    if m:
        return SourceIdentity("elyte_unsolv", m.group(2), extra=m.group(1), raw=m.group(0))
    if "outputs_unsolvated" in path:
        leaf = _leaf(path)
        m2 = re.search(r"_mol(\d+)_", leaf)
        if m2:
            return SourceIdentity("elyte_unsolv", m2.group(1), extra=leaf, raw=leaf)
    # Solvated droplets/shells are multi-molecule; no stable monomer ID.
    return None


def _parse_rgd(path: str) -> SourceIdentity | None:
    m = _RGD.search(path)
    if not m:
        return None
    # Species index distinguishes fragments along a reaction; frames of that species are confs.
    return SourceIdentity("rgd", m.group(1), extra=f"species{m.group(2)}", raw=m.group(0))


def _parse_metal(path: str) -> SourceIdentity | None:
    m = re.search(r"/(\d+)_(\d+)_(-?\d+)_(\d+)(?:/step\d+)?$", _strip_job_path(path))
    if m:
        return SourceIdentity("metal", m.group(1), extra=f"{m.group(2)}:{m.group(3)}:{m.group(4)}", raw=m.group(0))
    return None


_DATA_ID_PARSERS = {
    "ani2x": _parse_ani2x,
    "geom_orca6": _parse_geom,
    "orbnet_denali": _parse_orbnet,
    "spice": _parse_spice,
    "biomolecules": _parse_biomolecule,
    "elytes": _parse_elyte,
    "rgd": _parse_rgd,
    "reactivity": lambda path: _parse_anibb(path),  # pmechdb returns None → graph
    "metal_complexes": _parse_metal,
    "trans1x": lambda _path: None,  # reaction frames change bonding
}


def parse_source_identity(
    data_id: str | None,
    source: str | None,
    reference_source: str | None = None,
) -> SourceIdentity | None:
    path = preferred_source_path(source, reference_source)
    if not path:
        return None
    parser = _DATA_ID_PARSERS.get(data_id or "")
    if parser is not None:
        # Dedicated parsers may return None (e.g. salt complexes, reaction frames).
        return parser(path)
    # Generic fallback used by the source audit when data_id is unexpected.
    m = _CHEMBL.search(path)
    if m:
        return SourceIdentity("chembl", m.group(1).upper(), raw=m.group(0))
    m = _GEOM.search(path)
    if m:
        return SourceIdentity("geom", m.group(1), raw=m.group(0))
    m = _SPF.search(path)
    if m:
        return SourceIdentity("spf", m.group(1), raw=m.group(0))
    return None


def isotope_signature(numbers: Iterable[int], masses: Iterable[float] | None) -> str:
    z = as_int_numbers(numbers)
    if masses is None:
        return "natural"
    mass = np.asarray(list(masses), dtype=np.float64)
    default = atomic_masses[z]
    if mass.shape != z.shape:
        return "natural"
    if np.all(np.abs(mass - default) < 0.2):
        return "natural"
    parts = []
    for atom_z, atom_m in zip(z.tolist(), mass.tolist()):
        if abs(atom_m - atomic_masses[atom_z]) >= 0.2:
            parts.append(f"{chemical_symbols[atom_z]}:{atom_m:.3f}")
    return ",".join(parts) if parts else "natural"


def _digest(payload: str) -> str:
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()


def graph_hash(
    numbers: Iterable[int],
    positions,
    n_iter: int = WL_ITERATIONS,
    connect_scale: float = CONNECT_SCALE,
) -> str:
    """Weisfeiler–Lehman hash of the covalent adjacency graph labeled by Z."""
    z = as_int_numbers(numbers)
    pos = as_positions(positions)
    adj = adjacency_matrix(z, pos, scale=connect_scale)
    colors = [int(atom_z) for atom_z in z.tolist()]
    for _ in range(n_iter):
        nxt = []
        for i, color in enumerate(colors):
            neigh = tuple(sorted(int(colors[j]) for j in np.flatnonzero(adj[i])))
            # Deterministic 64-bit mix; avoid PYTHONHASHSEED.
            acc = (1469598103934665603 ^ (color & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
            acc = (acc * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            for item in neigh:
                acc ^= item & 0xFFFFFFFFFFFFFFFF
                acc = (acc * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            nxt.append(acc)
        colors = nxt
    return _digest("|".join(str(c) for c in sorted(colors)))


def parent_id_from_parts(
    *,
    graph_id: str,
    charge: int,
    spin: int,
    isotope_id: str,
    source_token: str | None,
    identity_kind: str,
) -> str:
    if identity_kind == "source" and source_token:
        core = f"src:{source_token}|graph:{graph_id}"
    else:
        core = f"graph:{graph_id}"
    return f"{core}|iso:{isotope_id}|Q{int(charge)}|S{int(spin)}"
