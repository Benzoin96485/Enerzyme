"""Regression tests for C-C capping in enerzyme.tasks.extractor.extract_submol.

A carbon that the graph search reaches by two routes (e.g. around a ring entered from one side) re-queues its
single C-C bonds; the repeated visit must keep the planned H cap instead of growing the fragment across the bond.
"""
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")
from rdkit import Chem
from rdkit.Chem import AllChem

from enerzyme.tasks.extractor import extract_submol


def embedded(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def heavy_atoms(mol, atom_map):
    return sorted(i for i in atom_map if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def n_caps(submol, atom_map):
    return submol.GetNumAtoms() - len(atom_map)


def test_methyl_capped_when_ring_is_entered_from_one_side():
    mol = embedded("Cc1ccccc1")  # 0 methyl, 1 ipso, 4 para
    submol, atom_map = extract_submol(mol, {4})
    assert heavy_atoms(mol, atom_map) == [1, 2, 3, 4, 5, 6]
    assert n_caps(submol, atom_map) == 1


def test_biaryl_bond_capped_when_ring_is_entered_from_one_side():
    mol = embedded("c1ccc(cc1)-c1ccccc1")  # first ring 0-5, second ring 6-11
    submol, atom_map = extract_submol(mol, {0})
    assert heavy_atoms(mol, atom_map) == [0, 1, 2, 3, 4, 5]
    assert n_caps(submol, atom_map) == 1

