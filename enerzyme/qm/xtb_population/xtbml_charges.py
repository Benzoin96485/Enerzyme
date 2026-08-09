#!/usr/bin/env python3
"""xtbml density-class atomic charges: q_A (RHF) and q_A_alpha / q_A_beta vs Result.charges.

**Alpha/beta channel charges** (``q_A_alpha``, ``q_A_beta``): on a closed-shell
geometry, ``Calculator.add("spin-polarization", γ)`` makes the xtbml density block
use ``wfn%nspin > 1``, which emits density-class keys with ``_alpha``/``_beta`` suffixes.

For pure UHF (e.g. OH), if xtbml still represents a single density channel, only the
merged ``q_A`` may be present; orbital-energy keys can still appear as ``*_alpha``.
See the ``spin_label`` branch in Fortran ``density.f90``.

Requires tblite with xtbml post-processing: ``Calculator.add("xtbml")`` (tblite >= 0.5.0 and
libtblite built with xtbml). If ``add("xtbml")`` is missing, this module falls back to
``xtbml.toml`` next to this file via the C API. tblite 0.4.0 has neither and cannot provide q_A.

Bundled with Enerzyme under ``enerzyme.qm.xtb_population``. Run ``main()`` from this module
in an environment with tblite >= 0.5 for a self-test.
"""
from __future__ import annotations

import importlib.metadata as im
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict, cast

ChargeMode = Literal["rhf_merged", "spin_split", "uhf"]

import numpy as np

from tblite.exceptions import TBLiteValueError
from tblite.interface import Calculator, Result
from tblite import library as _tblite_library


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def register_xtbml(calc: Calculator) -> str:
    """Register xtbml post-processing. Returns a short label describing how it was done."""
    try:
        calc.add("xtbml")
        return "add('xtbml')"
    except TBLiteValueError:
        toml = _this_dir() / "xtbml.toml"
        if not toml.is_file():
            raise
        # Same path the broken early 0.4.x toml branch intended: pass path string to the library
        _tblite_library.post_processing_push_back(calc._ctx, calc._calc, str(toml))
        return f"post_processing_push_back({toml.name!r})"


def _as1d(
    d: Mapping[str, Any], key: str, *, n_expected: int | None = None
) -> np.ndarray:
    v = d[key]
    a = np.asarray(v, dtype=float).reshape(-1)
    if n_expected is not None and a.size != n_expected:
        raise ValueError(f"{key}: expected size {n_expected}, got {a.size}")
    return a


def extract_xtbml_charges(
    res: Result,
    *,
    n_atoms: int,
    mode: ChargeMode,
) -> dict[str, np.ndarray | None]:
    """Read xtbml partial charges from ``post-processing-dict`` (keys depend on wf / tblite).

    - **rhf_merged**: closed shell w/o ``spin-polarization`` → expect ``q_A``.
    - **spin_split**: e.g. ``Calculator.add('spin-polarization', γ)`` with closed shell → expect
      ``q_A_alpha`` and ``q_A_beta`` (no merged ``q_A`` in typical builds).
    - **uhf**: open shell → either spin-split or merged ``q_A`` (see module docstring).
    """
    # Result.get for post-processing may return a dict-like or plain dict
    raw = res.get("post-processing-dict")
    if not isinstance(raw, Mapping):
        raise TypeError("post-processing-dict is missing or not a mapping")
    pp: Mapping[str, Any] = cast(Mapping[str, Any], raw)

    out: dict[str, np.ndarray | None] = {
        "q_A": None,
        "q_A_alpha": None,
        "q_A_beta": None,
    }
    if "q_A" in pp:
        out["q_A"] = _as1d(pp, "q_A", n_expected=n_atoms)
    if "q_A_alpha" in pp:
        out["q_A_alpha"] = _as1d(pp, "q_A_alpha", n_expected=n_atoms)
    if "q_A_beta" in pp:
        out["q_A_beta"] = _as1d(pp, "q_A_beta", n_expected=n_atoms)

    has_split = out["q_A_alpha"] is not None and out["q_A_beta"] is not None
    has_merged = out["q_A"] is not None

    if mode == "rhf_merged":
        if out["q_A"] is None:
            raise KeyError(
                "rhf_merged: expected key 'q_A'; "
                f"sample keys: {sorted(pp.keys())[:40]}..."
            )
    elif mode == "spin_split":
        if not has_split:
            raise KeyError(
                "spin_split: expected 'q_A_alpha' and 'q_A_beta' (use spin-polarization); "
                f"sample keys: {sorted(pp.keys())[:40]}..."
            )
    else:  # uhf
        if not has_split and not has_merged:
            raise KeyError(
                "uhf: expected spin-split (q_A_alpha, q_A_beta) or merged q_A; "
                f"sample keys: {sorted(pp.keys())[:50]}..."
            )
    return out


class CaseReport(TypedDict):
    label: str
    register: str
    charges: list[float]
    q_A: list[float] | None
    q_A_alpha: list[float] | None
    q_A_beta: list[float] | None
    max_abs_diff_qA_vs_charges: float | None
    max_abs_diff_sumab_vs_charges: float | None
    n_post_processing_keys: int
    post_processing_keys_sample: list[str]


def _run_case(
    label: str,
    numbers: np.ndarray,
    positions: np.ndarray,
    *,
    charge: float | None = None,
    uhf: int | None = None,
    mode: ChargeMode,
    spin_polarization: float | None = None,
) -> CaseReport:
    calc = Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=uhf)
    if spin_polarization is not None:
        calc.add("spin-polarization", float(spin_polarization))
    how = register_xtbml(calc)
    res = calc.singlepoint()
    n = int(res.get("natoms"))
    ch = np.asarray(res.get("charges"), dtype=float).reshape(-1)
    xq = extract_xtbml_charges(res, n_atoms=n, mode=mode)
    raw = res.get("post-processing-dict")
    pp = cast(Mapping[str, Any], raw) if isinstance(raw, Mapping) else {}
    keys = sorted(pp.keys())
    n_keys = len(keys)

    has_ab = xq["q_A_alpha"] is not None and xq["q_A_beta"] is not None

    # Primary comparison: spin-split totals vs Mulliken charges, or merged q_A vs charges
    d_qa: float | None = None
    d_sum: float | None = None
    if has_ab:
        qa_part = cast(np.ndarray, xq["q_A_alpha"])
        qb_part = cast(np.ndarray, xq["q_A_beta"])
        q_sum = qa_part + qb_part
        d_sum = float(np.max(np.abs(q_sum - ch)))
        if xq["q_A"] is not None:
            d_qa = float(np.max(np.abs(xq["q_A"] - ch)))
    elif xq["q_A"] is not None:
        d_qa = float(np.max(np.abs(xq["q_A"] - ch)))

    return {
        "label": label,
        "register": how,
        "charges": ch.tolist(),
        "q_A": xq["q_A"].tolist() if xq["q_A"] is not None else None,
        "q_A_alpha": xq["q_A_alpha"].tolist() if xq["q_A_alpha"] is not None else None,
        "q_A_beta": xq["q_A_beta"].tolist() if xq["q_A_beta"] is not None else None,
        "max_abs_diff_qA_vs_charges": d_qa,
        "max_abs_diff_sumab_vs_charges": d_sum,
        "n_post_processing_keys": n_keys,
        "post_processing_keys_sample": keys[:30],
    }


def _print_report(r: CaseReport) -> None:
    print(f"=== {r['label']} ===")
    print(f"  xtbml registration: {r['register']}")
    print(f"  Result.get('charges') (Mulliken, e): {r['charges']!r}")
    if r["q_A"] is not None:
        print(f"  xtbml q_A: {r['q_A']!r}")
    if r["q_A_alpha"] is not None:
        print(f"  xtbml q_A_alpha: {r['q_A_alpha']!r}")
    if r["q_A_beta"] is not None:
        print(f"  xtbml q_A_beta: {r['q_A_beta']!r}")
    if r["max_abs_diff_qA_vs_charges"] is not None:
        print(
            "  max |q_A - charges| (if q_A present vs charges): "
            f"{r['max_abs_diff_qA_vs_charges']:.2e}"
        )
    if r["max_abs_diff_sumab_vs_charges"] is not None:
        print(
            "  max |q_A_alpha + q_A_beta - charges|: "
            f"{r['max_abs_diff_sumab_vs_charges']:.2e}"
        )
    print(
        f"  post-processing-dict: {r['n_post_processing_keys']} keys "
        f"(first <=30: {r['post_processing_keys_sample']!r})"
    )
    print()


def _tblite_version_tuple() -> tuple[int, int, int]:
    v = im.version("tblite")
    parts = v.split(".")
    ns: list[int] = []
    for p in parts[:3]:
        try:
            ns.append(int("".join(ch for ch in p if ch.isdigit()) or "0"))
        except ValueError:
            ns.append(0)
    while len(ns) < 3:
        ns.append(0)
    return (ns[0], ns[1], ns[2])


def main() -> int:
    try:
        v = im.version("tblite")
    except Exception:
        v = "unknown"
    print("tblite (metadata):", v, flush=True)
    print(flush=True)

    if v != "unknown" and _tblite_version_tuple() < (0, 5, 0):
        print(
            "This tblite is older than 0.5.0: xtbml is not available via "
            "Calculator.add('xtbml') and the bundled libtblite has no xtbml "
            "post-processing, so q_A will not appear. Upgrade tblite and "
            "libtblite, then re-run.",
            file=sys.stderr,
        )
        return 1

    # H2O, closed shell (Bohr), same as env_check / scf_steps_probe
    h2o_numbers = np.array([8, 1, 1], dtype=int)
    h2o_pos = np.array(
        [
            [0.0, 0.0, 0.119262],
            [0.0, 0.763239, -0.477257],
            [0.0, -0.763239, -0.477257],
        ],
        dtype=float,
    )
    # OH doublet: 9 electrons, one unpaired; linear along z, ~1.8 Bohr O–H
    oh_numbers = np.array([8, 1], dtype=int)
    oh_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8324],
        ],
        dtype=float,
    )

    try:
        r_rhf = _run_case(
            "H2O closed-shell (RHF) — q_A",
            h2o_numbers,
            h2o_pos,
            mode="rhf_merged",
        )
        # Two-spin density (wfn%nspin>1 in xtbml): closed shell + empirical spin-polarization
        r_sp = _run_case(
            "H2O + spin-polarization (γ=1) — q_A_alpha, q_A_beta",
            h2o_numbers,
            h2o_pos,
            mode="spin_split",
            spin_polarization=1.0,
        )
        r_uhf = _run_case(
            "OH doublet (UHF) — q_A splits or merged q_A",
            oh_numbers,
            oh_pos,
            charge=0.0,
            uhf=1,
            mode="uhf",
        )
    except TBLiteValueError as exc:
        print("Could not register xtbml post-processing.", file=sys.stderr)
        print(
            "Install tblite >= 0.5.0 (and a matching libtblite with xtbml), or see xtbml.toml.",
            file=sys.stderr,
        )
        print(f"Exception: {exc!r}", file=sys.stderr)
        return 1
    except (KeyError, TypeError) as exc:
        print(
            "Single-point finished but xtbml charges are missing from post-processing-dict.",
            file=sys.stderr,
        )
        print(
            "Either xtbml is missing or density keys failed to populate.",
            file=sys.stderr,
        )
        print(f"Exception: {exc!r}", file=sys.stderr)
        return 1

    _print_report(r_rhf)
    _print_report(r_sp)
    _print_report(r_uhf)

    # Soft validation against Mulliken `charges` (same xTB density; definitions still aligned)
    atol = 5e-4

    def _sum_ab_tolerance_ok(case: CaseReport) -> tuple[bool, str]:
        if case["max_abs_diff_sumab_vs_charges"] is None:
            return (False, "no sum α+β comparand")
        d = case["max_abs_diff_sumab_vs_charges"]
        return (d <= atol, "max |q_A_alpha+q_A_beta-charges|")

    def _uhf_tolerance_ok(case: CaseReport) -> tuple[bool, str]:
        if case["max_abs_diff_sumab_vs_charges"] is not None:
            d = case["max_abs_diff_sumab_vs_charges"]
            return (d <= atol, "max |q_A_alpha+q_A_beta-charges|")
        if case["max_abs_diff_qA_vs_charges"] is not None:
            d = case["max_abs_diff_qA_vs_charges"]
            return (
                d <= atol,
                "max |q_A-charges| (merged density block)",
            )
        return (False, "no comparand")

    if r_rhf["max_abs_diff_qA_vs_charges"] is not None:
        if r_rhf["max_abs_diff_qA_vs_charges"] > atol:
            print(
                f"WARNING: RHF |q_A - charges|_max = {r_rhf['max_abs_diff_qA_vs_charges']:.2e} "
                f"exceeds {atol!r} (tune tolerance if your tblite build differs).",
                file=sys.stderr,
            )
            return 1
    ok_sp, label_sp = _sum_ab_tolerance_ok(r_sp)
    if not ok_sp:
        print(
            f"WARNING: spin-polarization test {label_sp} exceeds {atol!r} "
            f"(sumab={r_sp['max_abs_diff_sumab_vs_charges']!r}).",
            file=sys.stderr,
        )
        return 1
    ok_u, label_u = _uhf_tolerance_ok(r_uhf)
    if not ok_u:
        print(
            f"WARNING: UHF {label_u} exceeds {atol!r} "
            f"(sumab={r_uhf['max_abs_diff_sumab_vs_charges']!r}, "
            f"qa={r_uhf['max_abs_diff_qA_vs_charges']!r}).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
