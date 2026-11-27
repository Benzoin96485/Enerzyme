import weakref

import numpy as np
import torch
from ase import Atoms
from ase.units import Hartree

from enerzyme.data.transform import Transform
from enerzyme.tasks.batch import _decorate_batch_input
from enerzyme.tasks.calculator import ASECalculator


class PairEnergy(torch.nn.Module):
    def forward(self, inputs):
        ra = inputs["Ra"]
        delta = ra[inputs["idx_j"]] - ra[inputs["idx_i"]]
        energy = delta.square().sum() + inputs["Q"] + inputs["S"]
        force, = torch.autograd.grad(energy.sum(), ra)
        return {"E": energy, "Fa": -force, "Qa": inputs["Za"].to(ra.dtype)}


def calculator(**options):
    return ASECalculator(model=PairEnergy(), transform=Transform({}), device=torch.device("cpu"),
                         dtype=torch.float64, Hartree_in_E=Hartree, **options)


def assert_matches_collator(calc, atoms):
    features = dict(N=len(atoms), Za=atoms.numbers, Ra=atoms.positions,
                    Q=atoms.info.get("charge", 0), S=atoms.info.get("spin", 0) - 1)
    expected, _ = _decorate_batch_input([(features, None)], calc.dtype, calc.device)
    supplied = calc._prepare_input(atoms)
    assert supplied.keys() == expected.keys()
    for k in expected:
        if k == "N":
            assert supplied[k] == expected[k]
        else:
            torch.testing.assert_close(supplied[k], expected[k], rtol=0, atol=0)
    return supplied


def test_molecular_frames_reuse_full_pairs_and_keep_independent_positions():
    calc = calculator()
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = assert_matches_collator(calc, atoms)
    atoms.positions[1, 0] = 2.
    moved = assert_matches_collator(calc, atoms)
    assert tuple(first) == tuple(moved)
    for k in ("idx_i", "idx_j", "Za", "batch_seg"):
        assert moved[k] is first[k]
    assert moved["Ra"].is_leaf and moved["Ra"].requires_grad
    assert moved["Ra"].data_ptr() != first["Ra"].data_ptr()
    assert first["Ra"][1, 0].item() == 1.
    first_positions = weakref.ref(first["Ra"])
    del first
    assert first_positions() is None
    atoms.calc = calc
    np.testing.assert_array_equal(atoms.get_forces(), -4 * (3 * atoms.positions - atoms.positions.sum(0)))
    atoms.positions[0, 1] = 0.5
    np.testing.assert_array_equal(atoms.get_forces(), -4 * (3 * atoms.positions - atoms.positions.sum(0)))


def test_static_inputs_follow_atom_identity_metadata_precision_and_count():
    calc = calculator()
    atoms = Atoms("H2")
    assert_matches_collator(calc, atoms)
    atoms.numbers[1] = 8
    assert_matches_collator(calc, atoms)
    atoms.info.update(charge=0.5, spin=3)
    assert_matches_collator(calc, atoms)
    calc.dtype = torch.float32
    assert_matches_collator(calc, atoms)
    assert_matches_collator(calc, Atoms("He"))


def test_switching_away_from_full_pairs_bypasses_cache():
    calc = calculator()
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[3, 3, 3])
    molecular = calc._prepare_input(atoms)
    calc.neighbor_list_type = None
    own_graph = calc._prepare_input(atoms)
    assert own_graph["idx_i"] is not molecular["idx_i"]
    assert calc._prepare_input(atoms)["idx_i"] is not own_graph["idx_i"]
