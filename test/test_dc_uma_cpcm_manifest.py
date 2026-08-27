"""Candidate filters, parent identity, and frozen DC-UMA C-PCM manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from enerzyme.data.dc_uma_cpcm.constants import (
    CLOSED_SHELL_S2_MAX,
    N_PROBE_LABELS,
    N_TOTAL_LABELS,
    N_TRAIN_LABELS,
    SCHEMA_VERSION,
    SEED,
    UNSEEN_CLAIM,
    UNSEEN_CLAIM_NOTE,
)
from enerzyme.data.dc_uma_cpcm.geometry import heavy_atom_rmsd, n_connected_components
from enerzyme.data.dc_uma_cpcm.identity import graph_hash, parse_source_identity
from enerzyme.data.dc_uma_cpcm.index import FilterConfig, candidate_from_row, evaluate_record, is_closed_shell, scan_shard
from enerzyme.data.dc_uma_cpcm.sample import (
    assign_parents,
    build_manifest,
    find_near_duplicates,
    pick_conformer_pair,
    repair_closed_shell_parents,
    validate_manifest_counts,
)


def _atoms(kind="ethane", charge=0, spin=1, dx=0.0, scale=1.0):
    atoms = molecule("C2H6")
    numbers = atoms.numbers.copy()
    if kind == "fluoro":
        numbers[-1] = 9
    elif kind == "chloro":
        numbers[-1] = 17
    elif kind == "alcohol":
        numbers[-1] = 8
    elif kind == "metal":
        numbers[0] = 26
    atoms.set_atomic_numbers(numbers)
    atoms.positions = atoms.positions * scale
    atoms.positions = atoms.positions + np.array([dx, 0.0, 0.0])
    forces = np.zeros((len(atoms), 3))
    forces[0, 0] = 0.2
    forces[-1, 0] = -0.2
    atoms.calc = SinglePointCalculator(atoms, energy=-12.0, forces=forces)
    atoms.info.update({"charge": charge, "spin": spin, "dielectric": 1.0})
    return atoms


def _passing_kwargs(atoms, **overrides):
    payload = {
        "numbers": atoms.numbers,
        "positions": atoms.positions,
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(),
        "charge": 0,
        "spin": 1,
        "pbc": atoms.pbc,
        "n_basis_stored": 200,
        "composition": atoms.get_chemical_formula(),
        "cfg": FilterConfig(),
    }
    payload.update(overrides)
    return payload


def test_evaluate_record_rejects_natoms_elements_spin_charge_fmax_overlap_and_disconnect():
    atoms = _atoms()
    assert evaluate_record(**_passing_kwargs(atoms)) is None

    tiny = molecule("H2O")
    tiny.calc = SinglePointCalculator(tiny, energy=-1.0, forces=np.zeros((3, 3)))
    assert evaluate_record(**_passing_kwargs(tiny, numbers=tiny.numbers, positions=tiny.positions, composition="H2O")) == "natoms"

    metal = _atoms("metal")
    assert evaluate_record(**_passing_kwargs(metal, composition="CFeH6")) == "elements"
    assert evaluate_record(**_passing_kwargs(atoms, spin=3)) == "spin"
    assert evaluate_record(**_passing_kwargs(atoms, charge=2)) == "charge"
    assert evaluate_record(**_passing_kwargs(atoms, s_squared=0.73)) == "open_shell"
    assert evaluate_record(**_passing_kwargs(atoms, s_squared=0.0)) is None
    assert evaluate_record(**_passing_kwargs(atoms, s_squared=1e-8)) is None

    big_f = np.zeros((len(atoms), 3))
    big_f[0, 0] = 9.0
    assert evaluate_record(**_passing_kwargs(atoms, forces=big_f)) == "fmax"

    overlapped = _atoms()
    energy = overlapped.get_potential_energy()
    forces = overlapped.get_forces()
    pos = np.array(overlapped.positions, dtype=float, copy=True)
    pos[1] = pos[0]
    assert (
        evaluate_record(
            **_passing_kwargs(overlapped, positions=pos, energy=energy, forces=forces)
        )
        == "overlap"
    )

    dimer = _atoms()
    extra = _atoms(dx=20.0)
    numbers = np.concatenate([dimer.numbers, extra.numbers])
    positions = np.concatenate([dimer.positions, extra.positions])
    forces = np.zeros((len(numbers), 3))
    assert n_connected_components(numbers, positions) == 2
    assert (
        evaluate_record(
            **_passing_kwargs(
                dimer,
                numbers=numbers,
                positions=positions,
                forces=forces,
                composition="C4H12",
                n_basis_stored=400,
            )
        )
        == "disconnected"
    )


def test_source_identity_parsers_for_ten_data_ids():
    cases = {
        "ani2x": (
            "ani2x/ani2x_013_750202_0_1/orca.tar.zst",
            None,
            "ani2x:750202",
        ),
        "geom_orca6": (
            "geom_orca6/geom_92971110_0_1/orca.tar.zst",
            None,
            "geom:92971110",
        ),
        "orbnet_denali": (
            "orbnet_denali/orbnet_CHEMBL1935278_conformers_abc_675082_0_1/orca.tar.zst",
            None,
            "chembl:CHEMBL1935278",
        ),
        "spice": (
            "spice/SPICE_PubChem_Set_2_Single_Points_Dataset_v1_3_spice_135161093_17_0_1/orca.tar.zst",
            None,
            "spice_pubchem:135161093",
        ),
        "biomolecules": (
            "omol/solvated_protein/outputs_240923/spf_2570544_-1_1/step0/orca.tar.zst",
            None,
            "spf:2570544",
        ),
        "elytes": (
            "omol/electrolytes/outputs_unsolvated_120424/guanidinium_mol744_1_1/step4/orca.tar.zst",
            None,
            "elyte_unsolv:744:guanidinium",
        ),
        "rgd": (
            "rgd_uks/MR_444590_0_9_0_1/orca.tar.zst",
            None,
            "rgd:444590:species0",
        ),
        "reactivity": (
            "ani1xbb/aniBB_022_163758_0_1/orca.tar.zst",
            None,
            "anibb:163758",
        ),
        "metal_complexes": (
            "omol/metal_organics/restart5to6/job_1741410812_3a620755c25f/orca.tar.zst",
            "s3://opencatalysisdata/archive/hot/omol/metal_organics/outputs_070324/499824_1_1_4/step1/orca.tar.zst",
            "metal:499824:1:1:4",
        ),
        "trans1x": (
            "trans1x/t1x_rxn9542_3034_2926_0_1/orca.tar.zst",
            None,
            None,
        ),
    }
    for data_id, (source, ref, token) in cases.items():
        ident = parse_source_identity(data_id, source, ref)
        if token is None:
            assert ident is None, data_id
        else:
            assert ident is not None, data_id
            assert ident.token == token, (data_id, ident.token, token)


def test_orbnet_protomer_and_pmechdb_use_source_or_graph():
    prot = parse_source_identity(
        "orbnet_denali",
        "orbnet_denali/orbnet_CHEMBL2288213_protomer_2_ad616_623257_-1_1/orca.tar.zst",
        None,
    )
    assert prot is not None
    assert prot.token == "chembl:CHEMBL2288213:protomer2"
    assert parse_source_identity("reactivity", "pmechdb/pmechdb_33753_step1_1_1/orca.tar.zst", None) is None
    assert parse_source_identity(
        "orbnet_denali",
        "orbnet_denali/orbnet_saltcomplex_21132_CHEMBL169776_Diethylamine_abc_146089_0_1/orca.tar.zst",
        None,
    ) is None


def _cand(parent_tag, conf, charge=0, kind="ethane", source_mol_id=None):
    atoms = _atoms(kind, charge=charge)
    z = [int(v) for v in atoms.numbers]
    pos = np.array(atoms.positions, dtype=float, copy=True)
    pos[0, 0] += 0.22 * (int(parent_tag) % 13)
    pos[1, 1] += 0.17 * ((int(parent_tag) // 2) % 11)
    pos[:, 2] += 0.08 * int(conf)
    forces = np.zeros((len(z), 3))
    forces[0, 0] = 0.2
    forces[-1, 0] = -0.2
    pos_list = pos.tolist()
    return {
        "shard": "data0000.aselmdb",
        "row_id": parent_tag * 10 + conf + 1,
        "data_id": "geom_orca6",
        "source": f"geom_orca6/geom_{parent_tag}_{charge}_1/orca.tar.zst",
        "reference_source": None,
        "charge": charge,
        "spin": 1,
        "n_atoms": len(z),
        "n_heavy": int(sum(v > 1 for v in z)),
        "n_basis": 200,
        "n_basis_est": 200,
        "energy": -12.0 - 0.01 * conf,
        "fmax": float(np.max(np.abs(forces))),
        "frms": float(np.sqrt(np.mean(forces * forces))),
        "rg": 1.2 + 0.01 * parent_tag,
        "mean_abs_lowdin": 0.1,
        "composition": atoms.get_chemical_formula(),
        "numbers": z,
        "positions": pos_list,
        "forces": forces.tolist(),
        "source_mol_id": source_mol_id or f"geom:{parent_tag}",
        "graph_id": graph_hash(z, pos_list),
        "isotope_id": "natural",
    }


def test_parent_prefers_source_id_and_splits_unstable_bond_graphs():
    a0 = _cand(1, 0, kind="ethane", source_mol_id="chembl:X")
    a1 = _cand(1, 1, kind="ethane", source_mol_id="chembl:X")
    b0 = _cand(2, 0, kind="fluoro", source_mol_id="chembl:X")
    b1 = _cand(2, 1, kind="fluoro", source_mol_id="chembl:X")
    assigned = assign_parents([a0, a1, b0, b1])
    parent_ids = {rec["parent_id"] for rec in assigned}
    assert len(parent_ids) == 2
    assert all(rec["identity_kind"] == "source" for rec in assigned)
    graphs = {rec["parent_id"]: rec["graph_id"] for rec in assigned}
    assert len(set(graphs.values())) == 2


def test_graph_fallback_groups_source_singletons():
    recs = [
        _cand(10, 0, source_mol_id="ani2x:1"),
        _cand(10, 1, source_mol_id="ani2x:2"),
    ]
    recs[1]["numbers"] = list(recs[0]["numbers"])
    recs[1]["graph_id"] = recs[0]["graph_id"]
    recs[1]["positions"] = (np.asarray(recs[0]["positions"]) + np.array([0.2, 0.0, 0.0])).tolist()
    assigned = assign_parents(recs)
    assert len({rec["parent_id"] for rec in assigned}) == 1
    assert assigned[0]["identity_kind"] == "graph"


def test_pick_max_heavy_atom_rmsd_pair():
    members = [_cand(3, c, kind="ethane") for c in range(4)]
    members[0]["positions"] = _atoms("ethane", dx=0.0).positions.tolist()
    members[1]["positions"] = _atoms("ethane", dx=0.05).positions.tolist()
    members[2]["positions"] = _atoms("ethane", dx=1.5).positions.tolist()
    members[3]["positions"] = _atoms("ethane", dx=0.02).positions.tolist()
    pair, rmsd = pick_conformer_pair(members, src=None, seed=SEED, parent_id="p")
    assert len(pair) == 2
    expected = heavy_atom_rmsd(
        members[0]["numbers"], members[0]["positions"], members[2]["numbers"], members[2]["positions"]
    )
    assert rmsd == pytest.approx(expected, rel=1e-6, abs=1e-6)


def _family(n_parents, charge, kind, start=0):
    recs = []
    for i in range(n_parents):
        recs.append(_cand(start + i, 0, charge=charge, kind=kind, source_mol_id=f"geom:{kind}:{charge}:{start+i}"))
        recs.append(_cand(start + i, 1, charge=charge, kind=kind, source_mol_id=f"geom:{kind}:{charge}:{start+i}"))
        recs[-1]["graph_id"] = recs[-2]["graph_id"]
        recs[-1]["numbers"] = list(recs[-2]["numbers"])
        recs[-1]["positions"] = (np.asarray(recs[-2]["positions"]) + np.array([0.35, 0.0, 0.0])).tolist()
    return recs


def test_parent_split_has_no_leak_and_writes_label_matrix():
    recs = (
        _family(6, 0, "ethane", 0)
        + _family(5, 1, "fluoro", 50)
        + _family(5, -1, "chloro", 80)
    )
    assigned = assign_parents(recs)
    charge_split = {
        0: {"train": 2, "validation": 1, "test": 1},
        1: {"train": 1, "validation": 1, "test": 1},
        -1: {"train": 1, "validation": 1, "test": 1},
    }
    charge_reserve = {0: 1, 1: 1, -1: 1}
    manifest = build_manifest(
        assigned,
        src=None,
        seed=SEED,
        charge_split=charge_split,
        charge_reserve=charge_reserve,
        near_dup_rmsd=1.0e-6,
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["unseen_claim"] == UNSEEN_CLAIM
    assert "independent unseen chemistry" in manifest["unseen_claim_note"].lower()
    splits = {name: set() for name in ("train", "validation", "test")}
    for parent in manifest["parents"]:
        splits[parent["split"]].add(parent["parent_id"])
    assert not splits["train"] & splits["validation"]
    assert not splits["train"] & splits["test"]
    assert not splits["validation"] & splits["test"]
    assert manifest["counts"]["train_parents"] == 4
    assert manifest["counts"]["validation_parents"] == 3
    assert manifest["counts"]["test_parents"] == 3
    assert manifest["counts"]["train_geometries"] == 8
    assert all(lab["epsilon"] != 1.0 for lab in manifest["labels"])
    assert "jobs" in manifest["diagnostic"]
    assert manifest["diagnostic"]["n_labels"] == len(manifest["diagnostic"]["jobs"])
    validate_manifest_counts(manifest, strict_full_matrix=False)
    pilot_parents = []
    geoms = {g["geometry_id"]: g for g in manifest["geometries"]}
    for gid in manifest["pilot"]["geometry_ids"]:
        pilot_parents.append(geoms[gid]["parent_id"])
    assert len(pilot_parents) == len(set(pilot_parents))
    # Frozen seed is reproducible.
    again = build_manifest(
        assigned,
        src=None,
        seed=SEED,
        charge_split=charge_split,
        charge_reserve=charge_reserve,
        near_dup_rmsd=1.0e-6,
    )
    assert [p["parent_id"] for p in again["parents"]] == [p["parent_id"] for p in manifest["parents"]]


def test_near_duplicate_uses_preassigned_reserve_not_model_metrics():
    recs = _family(5, 0, "ethane", 0)
    assigned = assign_parents(recs)
    charge_split = {0: {"train": 2, "validation": 1, "test": 1}, 1: {"train": 0, "validation": 0, "test": 0}, -1: {"train": 0, "validation": 0, "test": 0}}
    charge_reserve = {0: 1, 1: 0, -1: 0}
    manifest = build_manifest(
        assigned,
        src=None,
        seed=SEED,
        charge_split=charge_split,
        charge_reserve=charge_reserve,
        near_dup_rmsd=1.0e-6,
    )
    train = [g for g in manifest["geometries"] if g["split"] == "train"][0]
    val = [g for g in manifest["geometries"] if g["split"] == "validation"][0]
    val["numbers"] = list(train["numbers"])
    val["positions"] = [list(p) for p in train["positions"]]
    leaks = find_near_duplicates([train, val], rmsd_cut=0.15)
    assert leaks
    assert leaks[0]["split_a"] != leaks[0]["split_b"]


def test_index_toy_aselmdb(tmp_path):
    db_path = tmp_path / "data0000.aselmdb"
    good = _atoms("ethane")
    metal = _atoms("metal")
    with connect(str(db_path)) as db:
        db.write(
            good,
            data={
                "source": "geom_orca6/geom_1_0_1/orca.tar.zst",
                "reference_source": None,
                "data_id": "geom_orca6",
                "charge": 0,
                "spin": 1,
                "num_atoms": len(good),
                "n_basis": 200,
                "composition": good.get_chemical_formula(),
                "lowdin_charges": np.zeros(len(good)),
            },
        )
        db.write(
            metal,
            data={
                "source": "omol/metal_organics/outputs/1_0_0_1/step0/orca.tar.zst",
                "reference_source": None,
                "data_id": "metal_complexes",
                "charge": 0,
                "spin": 1,
                "num_atoms": len(metal),
                "n_basis": 200,
                "composition": metal.get_chemical_formula(),
            },
        )
    records, stats = scan_shard(str(db_path))
    assert stats.total == 2
    assert stats.passed == 1
    assert stats.reasons.get("elements") == 1
    assert records[0]["source_mol_id"] == "geom:1"
    rec = candidate_from_row(
        shard_name="data0000.aselmdb",
        row_id=1,
        numbers=good.numbers,
        positions=good.positions,
        energy=-12.0,
        forces=good.get_forces(),
        masses=None,
        data={"data_id": "geom_orca6", "source": "geom_orca6/geom_1_0_1/orca.tar.zst", "charge": 0, "spin": 1, "n_basis": 200, "composition": good.get_chemical_formula()},
        pbc=good.pbc,
        cfg=FilterConfig(),
        stats=stats,
    )
    assert rec is not None
    assert is_closed_shell(rec.get("s_squared"))


def test_index_rejects_open_shell_singlet(tmp_path):
    db_path = tmp_path / "data0000.aselmdb"
    good = _atoms("ethane")
    oss = _atoms("chloro")
    with connect(str(db_path)) as db:
        db.write(
            good,
            data={
                "source": "geom_orca6/geom_1_0_1/orca.tar.zst",
                "data_id": "geom_orca6",
                "charge": 0,
                "spin": 1,
                "num_atoms": len(good),
                "n_basis": 200,
                "composition": good.get_chemical_formula(),
                "s_squared": 0.0,
                "unrestricted": True,
            },
        )
        db.write(
            oss,
            data={
                "source": "reactivity/ani1xbb_1_-1_1/orca.tar.zst",
                "data_id": "reactivity",
                "charge": -1,
                "spin": 1,
                "num_atoms": len(oss),
                "n_basis": 200,
                "composition": oss.get_chemical_formula(),
                "s_squared": 0.734,
                "unrestricted": True,
            },
        )
    records, stats = scan_shard(str(db_path))
    assert stats.reasons.get("open_shell") == 1
    assert len(records) == 1
    assert records[0]["s_squared"] == 0.0


def test_assign_parents_drops_open_shell_and_repair_uses_reserve():
    recs = (
        _family(3, 0, "ethane", 0)
        + _family(2, 1, "fluoro", 50)
        + _family(2, -1, "chloro", 80)
    )
    recs[0]["s_squared"] = 0.73
    recs[1]["s_squared"] = 0.60
    assigned = assign_parents(recs)
    assert recs[0]["source_mol_id"] not in {rec.get("source_mol_id") for rec in assigned}

    recs2 = _family(5, 0, "ethane", 0) + _family(3, 1, "fluoro", 50) + _family(3, -1, "chloro", 80)
    assigned2 = assign_parents(recs2)
    charge_split = {
        0: {"train": 2, "validation": 1, "test": 1},
        1: {"train": 1, "validation": 1, "test": 1},
        -1: {"train": 1, "validation": 1, "test": 1},
    }
    charge_reserve = {0: 1, 1: 1, -1: 1}
    manifest = build_manifest(
        assigned2,
        src=None,
        seed=SEED,
        charge_split=charge_split,
        charge_reserve=charge_reserve,
        near_dup_rmsd=1.0e-6,
    )
    assert manifest["reserve"]
    reserve_charges = {item["charge"] for item in manifest["reserve"]}
    oss_parent = next(p["parent_id"] for p in manifest["parents"] if p["charge"] in reserve_charges)
    for geom in manifest["geometries"]:
        if geom["parent_id"] == oss_parent:
            geom["s_squared"] = 0.73
    repaired = repair_closed_shell_parents(manifest)
    assert oss_parent not in {p["parent_id"] for p in repaired["parents"]}
    assert repaired["sampling"]["open_shell_replacements"]
    assert all(is_closed_shell(g.get("s_squared")) for g in repaired["geometries"])
    validate_manifest_counts(repaired, strict_full_matrix=False)


def test_constants_match_draft_label_matrix():
    assert N_TRAIN_LABELS == 896
    assert N_PROBE_LABELS == 96
    assert N_TOTAL_LABELS == 2000
    assert SEED == 20260817
    assert CLOSED_SHELL_S2_MAX == 0.05
    assert UNSEEN_CLAIM == "adapter-unseen parent"
    assert "independent unseen chemistry" in UNSEEN_CLAIM_NOTE.lower()
