"""Deterministic parent split, conformer pair selection, and frozen label matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import (
    CHARGE_RESERVE,
    CHARGE_SPLIT,
    CONFS_PER_PARENT,
    DIAG_TRAIN_BY_CHARGE,
    DIAG_VAL_BY_CHARGE,
    EVAL_EPSILONS,
    MAX_CONFS_FOR_RMSD,
    NEAR_DUP_RMSD_A,
    N_PILOT_GEOMS,
    N_PROBE_PARENTS,
    PILOT_EPSILONS,
    PROBE_EPSILONS,
    SCHEMA_VERSION,
    SEED,
    SIZE_BINS,
    TRAIN_EPSILONS,
    UNSEEN_CLAIM,
    UNSEEN_CLAIM_NOTE,
)
from .geometry import formula_from_numbers, heavy_atom_rmsd
from .identity import parent_id_from_parts
from .index import annotate_s_squared, drop_open_shell_records, is_closed_shell, read_jsonl

ELEMENT_ORDER = ("H", "C", "N", "O", "F", "P", "S", "Cl")
Z_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def size_bin(n_atoms: float) -> str:
    n = float(n_atoms)
    if n <= 15:
        return "6-15"
    if n <= 30:
        return "16-30"
    return "31-60"


def assign_parents(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer source molecule IDs; fall back to bond-graph groups with ≥2 confs."""
    closed, _n_open = drop_open_shell_records(records)
    by_source: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    leftovers: list[dict[str, Any]] = []
    for rec in closed:
        token = rec.get("source_mol_id")
        if token:
            key = (token, int(rec["charge"]), int(rec["spin"]), rec.get("isotope_id") or "natural")
            by_source[key].append(rec)
        else:
            leftovers.append(rec)

    assigned: list[dict[str, Any]] = []
    for (token, charge, spin, isotope_id), group in by_source.items():
        by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in group:
            by_graph[rec["graph_id"]].append(rec)
        if len(group) < 2:
            leftovers.extend(group)
            continue
        for graph_id, sub in by_graph.items():
            if len(sub) < 2:
                leftovers.extend(sub)
                continue
            parent_id = parent_id_from_parts(
                graph_id=graph_id,
                charge=charge,
                spin=spin,
                isotope_id=isotope_id,
                source_token=token,
                identity_kind="source",
            )
            stable = len(by_graph) == 1
            for rec in sub:
                out = dict(rec)
                out["parent_id"] = parent_id
                out["identity_kind"] = "source"
                out["identity_stable"] = stable
                assigned.append(out)

    by_graph_left: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in leftovers:
        key = (
            rec["graph_id"],
            int(rec["charge"]),
            int(rec["spin"]),
            rec.get("isotope_id") or "natural",
        )
        by_graph_left[key].append(rec)
    for (graph_id, charge, spin, isotope_id), group in by_graph_left.items():
        if len(group) < 2:
            continue
        parent_id = parent_id_from_parts(
            graph_id=graph_id,
            charge=charge,
            spin=spin,
            isotope_id=isotope_id,
            source_token=None,
            identity_kind="graph",
        )
        for rec in group:
            out = dict(rec)
            out["parent_id"] = parent_id
            out["identity_kind"] = "graph"
            out["identity_stable"] = True
            assigned.append(out)
    return assigned


def _element_fractions(numbers: list[int]) -> np.ndarray:
    n = max(len(numbers), 1)
    counts = {sym: 0 for sym in ELEMENT_ORDER}
    for z in numbers:
        sym = Z_TO_SYM.get(int(z))
        if sym is not None:
            counts[sym] += 1
    return np.array([counts[sym] / n for sym in ELEMENT_ORDER], dtype=np.float64)


def parent_feature_vector(members: list[dict[str, Any]]) -> np.ndarray:
    n_atoms = float(np.mean([m["n_atoms"] for m in members]))
    n_heavy = float(np.mean([m["n_heavy"] for m in members]))
    rg = float(np.mean([m["rg"] for m in members]))
    fmax = float(np.mean([m["fmax"] for m in members]))
    frms = float(np.mean([m["frms"] for m in members]))
    lowdin_vals = [m["mean_abs_lowdin"] for m in members if m.get("mean_abs_lowdin") is not None]
    lowdin = float(np.mean(lowdin_vals)) if lowdin_vals else 0.0
    frac = _element_fractions(members[0]["numbers"])
    hetero = float(frac[2:].sum())
    q = abs(int(members[0]["charge"]))
    return np.concatenate(
        [
            np.array(
                [n_atoms / 60.0, n_heavy / 60.0, hetero, rg / 10.0, fmax / 5.0, frms / 5.0, lowdin, q],
                dtype=np.float64,
            ),
            frac,
        ]
    )


def _standardize(matrix: np.ndarray) -> np.ndarray:
    std = matrix.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (matrix - matrix.mean(axis=0)) / std


def farthest_point_sample(
    ids: list[str],
    features: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> list[str]:
    n = len(ids)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        order = np.arange(n)
        rng.shuffle(order)
        return [ids[i] for i in order.tolist()]
    x = _standardize(features)
    start = int(np.argmin(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))
    selected = [start]
    min_dist = np.full(n, np.inf, dtype=np.float64)
    taken = np.zeros(n, dtype=bool)
    taken[start] = True
    for _ in range(k - 1):
        last = x[selected[-1]]
        min_dist = np.minimum(min_dist, np.linalg.norm(x - last, axis=1))
        min_dist[taken] = -np.inf
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        taken[nxt] = True
    return [ids[i] for i in selected]


def _allocate_capped(targets: list[int], available: list[int]) -> list[int]:
    """Meet ``targets`` without exceeding ``available``, spilling remainder to richer bins."""
    n = len(targets)
    out = [min(t, a) for t, a in zip(targets, available)]
    leftover = sum(targets) - sum(out)
    while leftover > 0:
        room = [a - o for a, o in zip(available, out)]
        if not any(r > 0 for r in room):
            break
        order = sorted(range(n), key=lambda i: room[i], reverse=True)
        progressed = False
        for i in order:
            if leftover <= 0:
                break
            if room[i] <= 0:
                continue
            take = min(room[i], leftover)
            out[i] += take
            leftover -= take
            progressed = True
        if not progressed:
            break
    return out


def _largest_remainder(weights: list[int], total: int) -> list[int]:
    if total <= 0:
        return [0] * len(weights)
    s = sum(weights) or len(weights)
    raw = [total * w / s for w in weights]
    ints = [int(np.floor(x)) for x in raw]
    remain = total - sum(ints)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - ints[i], reverse=True)
    for i in order[:remain]:
        ints[i] += 1
    return ints


def _rng(seed: int, *parts: Any) -> np.random.RandomState:
    import hashlib

    payload = "|".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.blake2s(payload.encode(), digest_size=4).digest()
    return np.random.RandomState(int.from_bytes(digest, "little") % (2**31 - 1))


def build_parent_table(assigned: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in assigned:
        groups[rec["parent_id"]].append(rec)
    table = {}
    for parent_id, members in groups.items():
        members = sorted(members, key=lambda m: (m["shard"], int(m["row_id"])))
        n_atoms = float(np.mean([m["n_atoms"] for m in members]))
        table[parent_id] = {
            "parent_id": parent_id,
            "charge": int(members[0]["charge"]),
            "spin": int(members[0]["spin"]),
            "identity_kind": members[0]["identity_kind"],
            "identity_stable": all(m.get("identity_stable", True) for m in members),
            "graph_id": members[0]["graph_id"],
            "source_mol_id": members[0].get("source_mol_id"),
            "isotope_id": members[0].get("isotope_id") or "natural",
            "n_atoms_mean": n_atoms,
            "size_bin": size_bin(n_atoms),
            "n_conformers": len(members),
            "composition": members[0].get("composition") or formula_from_numbers(members[0]["numbers"]),
            "data_ids": sorted({m["data_id"] for m in members}),
            "members": members,
            "features": parent_feature_vector(members),
        }
    return table


def stratified_select_parents(
    table: dict[str, dict[str, Any]],
    seed: int = SEED,
    charge_split: dict[int, dict[str, int]] | None = None,
    charge_reserve: dict[int, int] | None = None,
) -> dict[str, Any]:
    charge_split = charge_split or CHARGE_SPLIT
    charge_reserve = charge_reserve or CHARGE_RESERVE
    by_cell: dict[tuple[int, str], list[str]] = defaultdict(list)
    for parent_id, info in table.items():
        by_cell[(int(info["charge"]), info["size_bin"])].append(parent_id)

    selected_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    reserve: list[dict[str, Any]] = []
    cell_report = []
    shortages = []

    for charge, split_counts in charge_split.items():
        n_train, n_val, n_test = split_counts["train"], split_counts["validation"], split_counts["test"]
        n_split = n_train + n_val + n_test
        n_res = int(charge_reserve.get(charge, 0))
        avail = [len(by_cell[(charge, b)]) for b in SIZE_BINS]
        split_target = _largest_remainder([1, 1, 1], n_split)
        res_target = _largest_remainder([1, 1, 1], n_res)
        split_alloc = _allocate_capped(split_target, avail)
        remaining_after_split = [a - s for a, s in zip(avail, split_alloc)]
        res_alloc = _allocate_capped(res_target, remaining_after_split)
        train_alloc = _largest_remainder([max(s, 0) for s in split_alloc] or [1, 1, 1], n_train)
        val_alloc = _largest_remainder([max(s, 0) for s in split_alloc] or [1, 1, 1], n_val)
        test_alloc = [s - t - v for s, t, v in zip(split_alloc, train_alloc, val_alloc)]
        test_alloc = [max(0, x) for x in test_alloc]
        leftover_ids: list[str] = []
        leftover_feat: list[np.ndarray] = []

        for bin_i, size in enumerate(SIZE_BINS):
            ids = list(by_cell[(charge, size)])
            rng = _rng(seed, "fps", charge, size)
            ids_sorted = sorted(ids)
            feats = np.stack([table[i]["features"] for i in ids_sorted]) if ids_sorted else np.zeros((0, 1))
            need = split_alloc[bin_i] + res_alloc[bin_i]
            picked = farthest_point_sample(ids_sorted, feats, need, rng) if ids_sorted else []
            if len(picked) < need:
                shortages.append(
                    {"charge": charge, "size_bin": size, "needed": need, "available": len(picked)}
                )
            n_cell_split = min(split_alloc[bin_i], len(picked))
            cell_split = picked[:n_cell_split]
            cell_res = picked[n_cell_split:]
            unused = [i for i in ids_sorted if i not in set(picked)]
            leftover_ids.extend(unused)
            leftover_feat.extend([table[i]["features"] for i in unused])
            rng_split = _rng(seed, "split", charge, size)
            order = np.arange(len(cell_split))
            rng_split.shuffle(order)
            shuffled = [cell_split[i] for i in order.tolist()]
            n_t = min(train_alloc[bin_i], len(shuffled))
            n_v = min(val_alloc[bin_i], max(0, len(shuffled) - n_t))
            n_te = min(test_alloc[bin_i], max(0, len(shuffled) - n_t - n_v))
            selected_split["train"].extend(shuffled[:n_t])
            selected_split["validation"].extend(shuffled[n_t:n_t + n_v])
            selected_split["test"].extend(shuffled[n_t + n_v:n_t + n_v + n_te])
            extra = shuffled[n_t + n_v + n_te:]
            for pid in cell_res + extra:
                reserve.append(
                    {
                        "parent_id": pid,
                        "charge": charge,
                        "size_bin": size,
                        "reason": "preassigned_reserve",
                    }
                )
            cell_report.append(
                {
                    "charge": charge,
                    "size_bin": size,
                    "available": len(ids),
                    "fps_taken": len(picked),
                    "train": n_t,
                    "validation": n_v,
                    "test": n_te,
                    "reserve": len(cell_res + extra),
                }
            )

        def _charge_count(split_name: str) -> int:
            return sum(1 for pid in selected_split[split_name] if table[pid]["charge"] == charge)

        missing = n_split - (_charge_count("train") + _charge_count("validation") + _charge_count("test"))
        if missing > 0 and leftover_ids:
            rng = _rng(seed, "fill", charge)
            feat = np.stack(leftover_feat)
            extra = farthest_point_sample(leftover_ids, feat, missing, rng)
            selected_split["train"].extend(extra)
            shortages.append({"charge": charge, "filled_from_other_bins": extra[:missing]})

        need_rebalance = (
            missing > 0
            or _charge_count("train") != n_train
            or _charge_count("validation") != n_val
            or _charge_count("test") != n_test
        )
        if need_rebalance:
            charge_ids = [
                pid
                for split_name in ("train", "validation", "test")
                for pid in selected_split[split_name]
                if table[pid]["charge"] == charge
            ]
            seen: set[str] = set()
            uniq: list[str] = []
            for pid in charge_ids:
                if pid not in seen:
                    seen.add(pid)
                    uniq.append(pid)
            rng = _rng(seed, "rebalance", charge)
            order = np.arange(len(uniq))
            rng.shuffle(order)
            uniq = [uniq[i] for i in order.tolist()][:n_split]
            for split_name in selected_split:
                selected_split[split_name] = [
                    pid for pid in selected_split[split_name] if table[pid]["charge"] != charge
                ]
            selected_split["train"].extend(uniq[:n_train])
            selected_split["validation"].extend(uniq[n_train : n_train + n_val])
            selected_split["test"].extend(uniq[n_train + n_val : n_train + n_val + n_test])
            used = set(uniq)
            reserve = [r for r in reserve if not (r["charge"] == charge and r["parent_id"] in used)]

    reserve_ids = []
    seen_res = set()
    for item in reserve:
        pid = item["parent_id"]
        if pid in seen_res:
            continue
        if any(pid in selected_split[s] for s in selected_split):
            continue
        seen_res.add(pid)
        reserve_ids.append(item)
    return {
        "split": selected_split,
        "reserve": reserve_ids,
        "cells": cell_report,
        "shortages": shortages,
    }


_DB_CACHE: dict[str, Any] = {}


def _load_row(src: str, shard: str, row_id: int):
    from enerzyme.data.dc_uma_cpcm.index import open_aselmdb

    path = str(Path(src) / shard) if not Path(shard).is_file() else shard
    if not Path(path).is_file():
        path = str(Path(src) / shard)
    db = _DB_CACHE.get(path)
    if db is None:
        db = open_aselmdb(path)
        _DB_CACHE[path] = db
    return db._get_row(int(row_id))


def pick_conformer_pair(
    members: list[dict[str, Any]],
    src: str | None,
    seed: int,
    parent_id: str,
    geometry_cache: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    pool = list(members)
    if len(pool) > MAX_CONFS_FOR_RMSD:
        rng = _rng(seed, "confsub", parent_id)
        idx = sorted(rng.choice(len(pool), MAX_CONFS_FOR_RMSD, replace=False).tolist())
        pool = [pool[i] for i in idx]
    loaded = []
    for rec in pool:
        key = (rec["shard"], int(rec["row_id"]))
        if geometry_cache is not None and key in geometry_cache:
            geo = geometry_cache[key]
        elif src is None:
            geo = {
                "numbers": rec["numbers"],
                "positions": rec["positions"],
                "energy": rec.get("energy"),
                "forces": rec.get("forces"),
            }
        else:
            row = _load_row(src, rec["shard"], rec["row_id"])
            geo = {
                "numbers": [int(z) for z in np.asarray(row.numbers).tolist()],
                "positions": np.asarray(row.positions, dtype=np.float64).tolist(),
                "energy": float(row.energy),
                "forces": np.asarray(row.forces, dtype=np.float64).tolist(),
            }
            if geometry_cache is not None:
                geometry_cache[key] = geo
        item = dict(rec)
        item.update(geo)
        loaded.append(item)
    if len(loaded) < 2:
        raise ValueError(f"parent {parent_id} has fewer than 2 loadable conformers")
    best_rmsd = -1.0
    best = (0, 1)
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            rmsd = heavy_atom_rmsd(
                loaded[i]["numbers"],
                loaded[i]["positions"],
                loaded[j]["numbers"],
                loaded[j]["positions"],
            )
            if rmsd > best_rmsd:
                best_rmsd = float(rmsd)
                best = (i, j)
    pair = [loaded[best[0]], loaded[best[1]]]
    pair = sorted(pair, key=lambda m: (m["shard"], int(m["row_id"])))
    return pair, best_rmsd


def _geom_id(rec: dict[str, Any]) -> str:
    return f"omol25:{rec['shard']}:{int(rec['row_id'])}"


def _geometry_record(rec: dict[str, Any], parent_id: str, split: str, roles: list[str]) -> dict[str, Any]:
    return {
        "geometry_id": _geom_id(rec),
        "parent_id": parent_id,
        "split": split,
        "roles": roles,
        "shard": rec["shard"],
        "row_id": int(rec["row_id"]),
        "data_id": rec.get("data_id"),
        "source": rec.get("source"),
        "reference_source": rec.get("reference_source"),
        "charge": int(rec["charge"]),
        "spin": int(rec["spin"]),
        "n_atoms": int(rec["n_atoms"]),
        "n_basis": rec.get("n_basis"),
        "n_basis_est": rec.get("n_basis_est"),
        "composition": rec.get("composition"),
        "numbers": rec["numbers"],
        "positions": rec["positions"],
        "vacuum_energy_ev": rec.get("energy"),
        "vacuum_forces_ev_a": rec.get("forces"),
        "fmax": rec.get("fmax"),
        "frms": rec.get("frms"),
        "rg": rec.get("rg"),
        "s_squared": rec.get("s_squared"),
        "unrestricted": rec.get("unrestricted"),
    }


def find_near_duplicates(geoms: list[dict[str, Any]], rmsd_cut: float = NEAR_DUP_RMSD_A) -> list[dict[str, Any]]:
    leaks = []
    for i, a in enumerate(geoms):
        for b in geoms[i + 1 :]:
            if a["split"] == b["split"]:
                continue
            if a["parent_id"] == b["parent_id"]:
                continue
            if int(a["n_atoms"]) != int(b["n_atoms"]):
                continue
            if int(a["charge"]) != int(b["charge"]):
                continue
            rmsd = heavy_atom_rmsd(a["numbers"], a["positions"], b["numbers"], b["positions"])
            if rmsd < rmsd_cut:
                leaks.append(
                    {
                        "geometry_a": a["geometry_id"],
                        "geometry_b": b["geometry_id"],
                        "parent_a": a["parent_id"],
                        "parent_b": b["parent_id"],
                        "split_a": a["split"],
                        "split_b": b["split"],
                        "rmsd": rmsd,
                    }
                )
    return leaks


_SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def _drop_split_for_leak(leak: dict[str, Any]) -> str:
    """Drop the later split (val/test before train) using a frozen priority, not model metrics."""
    a, b = leak["split_a"], leak["split_b"]
    return a if _SPLIT_PRIORITY[a] > _SPLIT_PRIORITY[b] else b


def _labels_for_geometry(geom: dict[str, Any], epsilons: Iterable[float], role: str) -> list[dict[str, Any]]:
    out = []
    for eps in epsilons:
        out.append(
            {
                "label_id": f"{geom['geometry_id']}|role={role}|eps={eps:g}",
                "geometry_id": geom["geometry_id"],
                "parent_id": geom["parent_id"],
                "split": geom["split"],
                "role": role,
                "epsilon": float(eps),
            }
        )
    return out


def build_manifest(
    assigned: list[dict[str, Any]],
    *,
    src: str | None = None,
    seed: int = SEED,
    audit: dict[str, Any] | None = None,
    charge_split: dict[int, dict[str, int]] | None = None,
    charge_reserve: dict[int, int] | None = None,
    include_geometries: bool = True,
    near_dup_rmsd: float = NEAR_DUP_RMSD_A,
) -> dict[str, Any]:
    table = build_parent_table(assigned)
    table = {
        pid: info
        for pid, info in table.items()
        if info["n_conformers"] >= CONFS_PER_PARENT
        and all(is_closed_shell(m.get("s_squared")) for m in info["members"])
    }
    selection = stratified_select_parents(
        table, seed=seed, charge_split=charge_split, charge_reserve=charge_reserve
    )
    cache: dict[tuple[str, int], dict[str, Any]] = {}

    def pair_for(parent_id: str) -> tuple[list[dict[str, Any]], float]:
        return pick_conformer_pair(table[parent_id]["members"], src, seed, parent_id, cache)

    parents_out = []
    geoms_out = []
    split_parents = {k: list(v) for k, v in selection["split"].items()}
    reserve_queue = list(selection["reserve"])

    def add_parent(parent_id: str, split: str, roles: list[str]) -> list[dict[str, Any]]:
        pair, rmsd = pair_for(parent_id)
        info = table[parent_id]
        parents_out.append(
            {
                "parent_id": parent_id,
                "split": split,
                "charge": info["charge"],
                "spin": info["spin"],
                "size_bin": info["size_bin"],
                "identity_kind": info["identity_kind"],
                "identity_stable": info["identity_stable"],
                "source_mol_id": info["source_mol_id"],
                "graph_id": info["graph_id"],
                "isotope_id": info["isotope_id"],
                "n_conformers_available": info["n_conformers"],
                "heavy_atom_rmsd": rmsd,
                "composition": info["composition"],
                "data_ids": info["data_ids"],
                "geometry_ids": [_geom_id(p) for p in pair],
            }
        )
        recs = []
        for rec in pair:
            geom = _geometry_record(rec, parent_id, split, list(roles))
            recs.append(geom)
            if include_geometries:
                geoms_out.append(geom)
        return recs

    for split, pids in split_parents.items():
        for pid in pids:
            add_parent(pid, split, [split])

    replacements = []
    leaks = find_near_duplicates(geoms_out, rmsd_cut=near_dup_rmsd)
    protected = {g["parent_id"] for g in geoms_out}
    while leaks:
        leak = leaks[0]
        drop_split = _drop_split_for_leak(leak)
        drop_parent = leak["parent_a"] if leak["split_a"] == drop_split else leak["parent_b"]
        charge = table[drop_parent]["charge"]
        size = table[drop_parent]["size_bin"]
        replacement = None
        for i, item in enumerate(reserve_queue):
            pid = item["parent_id"]
            if pid in protected:
                continue
            if table[pid]["charge"] != charge:
                continue
            if item.get("size_bin") == size or replacement is None:
                replacement = (i, item)
                if item.get("size_bin") == size:
                    break
        if replacement is None:
            raise RuntimeError(
                f"Near-duplicate {leak} could not be repaired; reserve exhausted for charge={charge}"
            )
        res_i, res_item = replacement
        reserve_queue.pop(res_i)
        new_pid = res_item["parent_id"]
        geoms_out[:] = [g for g in geoms_out if g["parent_id"] != drop_parent]
        parents_out[:] = [p for p in parents_out if p["parent_id"] != drop_parent]
        split_parents[drop_split] = [p for p in split_parents[drop_split] if p != drop_parent]
        split_parents[drop_split].append(new_pid)
        add_parent(new_pid, drop_split, [drop_split])
        protected.discard(drop_parent)
        protected.add(new_pid)
        replacements.append(
            {
                "dropped_parent": drop_parent,
                "dropped_split": drop_split,
                "replacement_parent": new_pid,
                "leak": leak,
                "note": "Replacement chosen from the pre-fixed reserve order, not from model metrics.",
            }
        )
        leaks = find_near_duplicates(geoms_out, rmsd_cut=near_dup_rmsd)

    reserve_out = []
    for item in reserve_queue:
        pid = item["parent_id"]
        if pid in protected:
            continue
        if pid not in table:
            continue
        if not all(is_closed_shell(m.get("s_squared")) for m in table[pid]["members"]):
            continue
        pair, rmsd = pair_for(pid)
        info = table[pid]
        reserve_out.append(
            {
                "parent_id": pid,
                "charge": info["charge"],
                "size_bin": info["size_bin"],
                "identity_kind": info["identity_kind"],
                "heavy_atom_rmsd": rmsd,
                "geometry_ids": [_geom_id(p) for p in pair],
                "geometries": [
                    _geometry_record(p, pid, "reserve", ["reserve"]) for p in pair
                ],
                "reason": item.get("reason", "preassigned_reserve"),
            }
        )

    return _finalize_manifest(
        parents_out=parents_out,
        geoms_out=geoms_out,
        reserve_out=reserve_out,
        seed=seed,
        audit=audit,
        sampling={
            "cells": selection["cells"],
            "shortages": selection["shortages"],
            "replacements": replacements,
            "eligible_parents": len(table),
        },
    )


def _parent_from_reserve_item(item: dict[str, Any], split: str) -> dict[str, Any]:
    geoms = item.get("geometries") or []
    first = geoms[0] if geoms else {}
    return {
        "parent_id": item["parent_id"],
        "split": split,
        "charge": item["charge"],
        "spin": int(first.get("spin") or 1),
        "size_bin": item["size_bin"],
        "identity_kind": item.get("identity_kind"),
        "identity_stable": first.get("identity_stable", True),
        "source_mol_id": first.get("source_mol_id"),
        "graph_id": first.get("graph_id"),
        "isotope_id": first.get("isotope_id") or "natural",
        "n_conformers_available": len(geoms) or len(item.get("geometry_ids") or []),
        "heavy_atom_rmsd": item.get("heavy_atom_rmsd"),
        "composition": first.get("composition"),
        "data_ids": first.get("data_ids") or ([first["data_id"]] if first.get("data_id") else []),
        "geometry_ids": [_geom_id_from_geom(g) for g in geoms] or list(item.get("geometry_ids") or []),
    }


def _geom_id_from_geom(geom: dict[str, Any]) -> str:
    return geom.get("geometry_id") or _geom_id(geom)


def _open_shell_parent_ids(geoms: Iterable[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for geom in geoms:
        if is_closed_shell(geom.get("s_squared")):
            continue
        pid = geom["parent_id"]
        if pid not in seen:
            seen.append(pid)
    return seen


def repair_closed_shell_parents(manifest: dict[str, Any]) -> dict[str, Any]:
    """Drop open-shell selected parents and replace them from the pre-fixed reserve.

    Replacement order is the frozen reserve queue (charge match, same size-bin
    preferred). This must not use model metrics.
    """
    parents_out = [dict(p) for p in manifest.get("parents") or []]
    geoms_out = [dict(g) for g in manifest.get("geometries") or []]
    reserve_queue = [dict(item) for item in manifest.get("reserve") or []]
    replacements = list((manifest.get("sampling") or {}).get("replacements") or [])
    open_shell_replacements: list[dict[str, Any]] = []

    for drop_pid in _open_shell_parent_ids(geoms_out):
        parent = next((p for p in parents_out if p["parent_id"] == drop_pid), None)
        if parent is None:
            geoms_out = [g for g in geoms_out if g["parent_id"] != drop_pid]
            continue
        charge = int(parent["charge"])
        size = parent.get("size_bin")
        replacement = None
        for i, item in enumerate(reserve_queue):
            repl_geoms = item.get("geometries") or []
            if any(not is_closed_shell(g.get("s_squared")) for g in repl_geoms):
                continue
            if int(item["charge"]) != charge:
                continue
            if item.get("size_bin") == size or replacement is None:
                replacement = (i, item)
                if item.get("size_bin") == size:
                    break
        if replacement is None:
            raise RuntimeError(
                f"Open-shell parent {drop_pid} could not be replaced; "
                f"closed-shell reserve exhausted for charge={charge}"
            )
        res_i, res_item = replacement
        reserve_queue.pop(res_i)
        new_pid = res_item["parent_id"]
        split = parent["split"]
        idx = next(i for i, p in enumerate(parents_out) if p["parent_id"] == drop_pid)
        geoms_out = [g for g in geoms_out if g["parent_id"] != drop_pid]
        new_geoms = []
        for geom in res_item.get("geometries") or []:
            ng = dict(geom)
            ng["parent_id"] = new_pid
            ng["split"] = split
            ng["roles"] = [split]
            new_geoms.append(ng)
        geoms_out.extend(new_geoms)
        parents_out[idx] = _parent_from_reserve_item(res_item, split)
        note = {
            "dropped_parent": drop_pid,
            "dropped_split": split,
            "replacement_parent": new_pid,
            "reason": "open_shell",
            "note": "Replacement chosen from the pre-fixed reserve order, not from model metrics.",
        }
        replacements.append(note)
        open_shell_replacements.append(note)

    reserve_out = [
        item
        for item in reserve_queue
        if all(is_closed_shell(g.get("s_squared")) for g in item.get("geometries") or [])
    ]
    sampling = dict(manifest.get("sampling") or {})
    sampling["replacements"] = replacements
    sampling["open_shell_replacements"] = open_shell_replacements
    return _finalize_manifest(
        parents_out=parents_out,
        geoms_out=geoms_out,
        reserve_out=reserve_out,
        seed=int(manifest.get("seed") or SEED),
        audit=manifest.get("source_audit_summary"),
        sampling=sampling,
        schema_version=manifest.get("schema_version") or SCHEMA_VERSION,
    )


def _finalize_manifest(
    *,
    parents_out: list[dict[str, Any]],
    geoms_out: list[dict[str, Any]],
    reserve_out: list[dict[str, Any]],
    seed: int,
    audit: dict[str, Any] | None,
    sampling: dict[str, Any],
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    for geom in geoms_out:
        split = geom["split"]
        geom["roles"] = [split]

    train_parent_ids = [p["parent_id"] for p in parents_out if p["split"] == "train"]
    probe_parent_ids = train_parent_ids[:N_PROBE_PARENTS]
    for geom in geoms_out:
        if geom["parent_id"] in probe_parent_ids and geom["split"] == "train":
            if "probe" not in geom["roles"]:
                geom["roles"].append("probe")

    diag_train: list[str] = []
    diag_val: list[str] = []
    for charge, n_need in DIAG_TRAIN_BY_CHARGE.items():
        taken = [p["parent_id"] for p in parents_out if p["split"] == "train" and p["charge"] == charge]
        diag_train.extend(taken[:n_need])
    for charge, n_need in DIAG_VAL_BY_CHARGE.items():
        taken = [p["parent_id"] for p in parents_out if p["split"] == "validation" and p["charge"] == charge]
        diag_val.extend(taken[:n_need])
    diag_set = set(diag_train + diag_val)
    for geom in geoms_out:
        if geom["parent_id"] in diag_set and "diagnostic" not in geom["roles"]:
            geom["roles"].append("diagnostic")

    pilot_specs = [
        (0, "6-15"),
        (0, "16-30"),
        (1, "6-15"),
        (1, "31-60"),
        (-1, "6-15"),
        (-1, "16-30"),
    ]
    pilot_geoms: list[str] = []
    used_pilot_geoms: set[str] = set()
    used_pilot_parents: set[str] = set()
    for charge, sbin in pilot_specs:
        candidates = [
            g
            for g in geoms_out
            if g["parent_id"] in diag_set
            and g["parent_id"] not in used_pilot_parents
            and g["geometry_id"] not in used_pilot_geoms
            and is_closed_shell(g.get("s_squared"))
        ]
        preferred = [
            g
            for g in candidates
            if int(g["charge"]) == charge and size_bin(g["n_atoms"]) == sbin
        ]
        if not preferred:
            preferred = [g for g in candidates if int(g["charge"]) == charge]
        if not preferred:
            preferred = candidates
        if preferred:
            gid = preferred[0]["geometry_id"]
            pilot_geoms.append(gid)
            used_pilot_geoms.add(gid)
            used_pilot_parents.add(preferred[0]["parent_id"])
        if len(pilot_geoms) >= N_PILOT_GEOMS:
            break
    for geom in geoms_out:
        if geom["geometry_id"] in pilot_geoms and "pilot" not in geom["roles"]:
            geom["roles"].append("pilot")

    labels: list[dict[str, Any]] = []
    for geom in geoms_out:
        if not is_closed_shell(geom.get("s_squared")):
            raise ValueError(
                f"open-shell geometry {geom['geometry_id']} leaked into the label matrix"
            )
        roles = set(geom["roles"])
        if "train" in roles:
            labels.extend(_labels_for_geometry(geom, TRAIN_EPSILONS, "train"))
        if "probe" in roles:
            labels.extend(_labels_for_geometry(geom, PROBE_EPSILONS, "probe"))
        if "validation" in roles:
            labels.extend(_labels_for_geometry(geom, EVAL_EPSILONS, "validation"))
        if "test" in roles:
            labels.extend(_labels_for_geometry(geom, EVAL_EPSILONS, "test"))

    pilot_jobs = []
    for gid in pilot_geoms:
        for eps in PILOT_EPSILONS:
            pilot_jobs.append(
                {
                    "job_id": f"{gid}|pilot|eps={eps}",
                    "geometry_id": gid,
                    "epsilon": eps,
                }
            )

    counts = {
        "train_parents": sum(p["split"] == "train" for p in parents_out),
        "validation_parents": sum(p["split"] == "validation" for p in parents_out),
        "test_parents": sum(p["split"] == "test" for p in parents_out),
        "train_geometries": sum(g["split"] == "train" for g in geoms_out),
        "validation_geometries": sum(g["split"] == "validation" for g in geoms_out),
        "test_geometries": sum(g["split"] == "test" for g in geoms_out),
        "probe_geometries": sum("probe" in g["roles"] for g in geoms_out),
        "labels": len(labels),
        "labels_by_role": {
            role: sum(x["role"] == role for x in labels)
            for role in ("train", "probe", "validation", "test")
        },
        "reserve_parents": len(reserve_out),
        "diagnostic_train_parents": len(diag_train),
        "diagnostic_validation_parents": len(diag_val),
        "pilot_geometries": len(pilot_geoms),
        "pilot_jobs": len(pilot_jobs),
        "near_duplicate_replacements": len(
            [r for r in (sampling.get("replacements") or []) if r.get("reason") != "open_shell"]
        ),
        "open_shell_replacements": len(sampling.get("open_shell_replacements") or []),
        "identity_kind": {
            "source": sum(p.get("identity_kind") == "source" for p in parents_out),
            "graph": sum(p.get("identity_kind") == "graph" for p in parents_out),
        },
    }
    if sampling.get("eligible_parents") is not None:
        counts["eligible_parents"] = sampling["eligible_parents"]
    return {
        "schema_version": schema_version,
        "seed": seed,
        "unseen_claim": UNSEEN_CLAIM,
        "unseen_claim_note": UNSEEN_CLAIM_NOTE,
        "counts": counts,
        "parents": parents_out,
        "geometries": geoms_out,
        "labels": labels,
        "reserve": reserve_out,
        "diagnostic": {
            "train_parent_ids": diag_train,
            "validation_parent_ids": diag_val,
            "n_labels": sum(
                1
                for lab in labels
                if lab["parent_id"] in diag_set and lab["role"] in ("train", "validation")
            ),
            "jobs": [
                {
                    "job_id": lab["label_id"],
                    "geometry_id": lab["geometry_id"],
                    "parent_id": lab["parent_id"],
                    "epsilon": lab["epsilon"],
                    "role": lab["role"],
                    "split": lab["split"],
                }
                for lab in labels
                if lab["parent_id"] in diag_set and lab["role"] in ("train", "validation")
            ],
        },
        "pilot": {"geometry_ids": pilot_geoms, "jobs": pilot_jobs},
        "sampling": sampling,
        "source_audit_summary": _audit_summary(audit),
    }


def _audit_summary(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if audit is None:
        return None
    keys = (
        "n_total",
        "n_passed",
        "observed_data_ids",
        "source_parse_ok",
        "source_parse_fail",
        "reject_reasons",
    )
    if any(k in audit for k in keys):
        return {k: audit.get(k) for k in keys}
    return dict(audit)


def load_candidate_records(index_dir: str | Path) -> list[dict[str, Any]]:
    index_dir = Path(index_dir)
    files = sorted(index_dir.glob("candidates/*.jsonl.gz")) + sorted(index_dir.glob("candidates/*.jsonl"))
    if not files:
        files = sorted(index_dir.glob("*.jsonl.gz")) + sorted(index_dir.glob("*.jsonl"))
    records: list[dict[str, Any]] = []
    for path in files:
        if path.name in {"source_audit.json", "dc_uma_cpcm_manifest.json"}:
            continue
        records.extend(read_jsonl(path))
    return records


def annotate_manifest_s_squared(manifest: dict[str, Any], src: str | Path | None) -> int:
    """Fill missing S² on selected and reserve geometries from OMol ASE-LMDB."""
    recs: list[dict[str, Any]] = list(manifest.get("geometries") or [])
    for item in manifest.get("reserve") or []:
        recs.extend(item.get("geometries") or [])
    return annotate_s_squared(recs, src)


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_manifest_counts(manifest: dict[str, Any], *, strict_full_matrix: bool = True) -> None:
    from .constants import (
        N_DIAG_JOBS,
        N_DIAG_TRAIN_PARENTS,
        N_DIAG_VAL_PARENTS,
        N_TEST_GEOMS,
        N_TEST_LABELS,
        N_TEST_PARENTS,
        N_TOTAL_LABELS,
        N_TRAIN_GEOMS,
        N_TRAIN_LABELS,
        N_TRAIN_PARENTS,
        N_VAL_GEOMS,
        N_VAL_LABELS,
        N_VAL_PARENTS,
        N_PROBE_LABELS,
        N_PROBE_GEOMS,
        UNSEEN_CLAIM,
    )

    if manifest.get("unseen_claim") != UNSEEN_CLAIM:
        raise ValueError("manifest unseen_claim must be exactly 'adapter-unseen parent'")
    counts = manifest["counts"]
    parent_ids = [p["parent_id"] for p in manifest["parents"]]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate parent_id in manifest")
    by_split: dict[str, set[str]] = defaultdict(set)
    for parent in manifest["parents"]:
        by_split[parent["split"]].add(parent["parent_id"])
    if by_split["train"] & by_split["validation"] or by_split["train"] & by_split["test"] or by_split["validation"] & by_split["test"]:
        raise ValueError("parent identity leaked across splits")
    geom_parents = {g["parent_id"] for g in manifest["geometries"]}
    if geom_parents - set(parent_ids):
        raise ValueError("geometry parent_id missing from parents list")
    note = (manifest.get("unseen_claim_note") or "").lower()
    if "independent unseen chemistry" not in note:
        raise ValueError("unseen_claim_note must forbid claiming independent unseen chemistry")
    if not strict_full_matrix:
        _assert_closed_shell_geometries(manifest)
        return
    expected = {
        "train_parents": N_TRAIN_PARENTS,
        "validation_parents": N_VAL_PARENTS,
        "test_parents": N_TEST_PARENTS,
        "train_geometries": N_TRAIN_GEOMS,
        "validation_geometries": N_VAL_GEOMS,
        "test_geometries": N_TEST_GEOMS,
        "probe_geometries": N_PROBE_GEOMS,
        "labels": N_TOTAL_LABELS,
    }
    for key, value in expected.items():
        if counts.get(key) != value:
            raise ValueError(f"count {key}={counts.get(key)} != {value}")
    by_role = counts.get("labels_by_role") or {}
    if by_role.get("train") != N_TRAIN_LABELS:
        raise ValueError(f"train labels {by_role.get('train')} != {N_TRAIN_LABELS}")
    if by_role.get("probe") != N_PROBE_LABELS:
        raise ValueError(f"probe labels {by_role.get('probe')} != {N_PROBE_LABELS}")
    if by_role.get("validation") != N_VAL_LABELS:
        raise ValueError(f"validation labels {by_role.get('validation')} != {N_VAL_LABELS}")
    if by_role.get("test") != N_TEST_LABELS:
        raise ValueError(f"test labels {by_role.get('test')} != {N_TEST_LABELS}")
    if counts.get("diagnostic_train_parents") != N_DIAG_TRAIN_PARENTS:
        raise ValueError(
            f"diagnostic train parents {counts.get('diagnostic_train_parents')} != {N_DIAG_TRAIN_PARENTS}"
        )
    if counts.get("diagnostic_validation_parents") != N_DIAG_VAL_PARENTS:
        raise ValueError(
            f"diagnostic validation parents {counts.get('diagnostic_validation_parents')} != {N_DIAG_VAL_PARENTS}"
        )
    diag = manifest.get("diagnostic") or {}
    if diag.get("n_labels") != N_DIAG_JOBS:
        raise ValueError(f"diagnostic labels {diag.get('n_labels')} != {N_DIAG_JOBS}")
    jobs = diag.get("jobs") or []
    if len(jobs) != N_DIAG_JOBS:
        raise ValueError(f"diagnostic jobs {len(jobs)} != {N_DIAG_JOBS}")
    _assert_closed_shell_geometries(manifest)


def _assert_closed_shell_geometries(manifest: dict[str, Any]) -> None:
    for geom in manifest.get("geometries") or []:
        if not is_closed_shell(geom.get("s_squared")):
            raise ValueError(
                f"open-shell geometry {geom.get('geometry_id')} "
                f"(S^2={geom.get('s_squared')}) is excluded this stage"
            )
    for item in manifest.get("reserve") or []:
        for geom in item.get("geometries") or []:
            if not is_closed_shell(geom.get("s_squared")):
                raise ValueError(
                    f"open-shell reserve geometry {geom.get('geometry_id')} "
                    f"(S^2={geom.get('s_squared')}) is excluded this stage"
                )
    geoms = {g["geometry_id"]: g for g in manifest.get("geometries") or []}
    for gid in (manifest.get("pilot") or {}).get("geometry_ids") or []:
        geom = geoms.get(gid)
        if geom is not None and not is_closed_shell(geom.get("s_squared")):
            raise ValueError(f"open-shell pilot geometry {gid} is excluded this stage")
    pilot_ids = (manifest.get("pilot") or {}).get("geometry_ids") or []
    pilot_parents = [geoms[gid]["parent_id"] for gid in pilot_ids if gid in geoms]
    if len(pilot_parents) != len(set(pilot_parents)):
        raise ValueError("pilot geometries must come from distinct parents")
