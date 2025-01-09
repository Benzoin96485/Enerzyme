from queue import Queue
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from rdkit.Chem import MolFromMolFile, MolFromXYZBlock
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity
from sklearn.neighbors import NearestNeighbors, BallTree
from rdkit import Chem
from ..data import REVERSED_PERIODIC_TABLE


def get_bond_lengths(conformer, begin_atom_idx, end_atom_idx):
    return np.linalg.norm(conformer.GetAtomPosition(begin_atom_idx) - conformer.GetAtomPosition(end_atom_idx))


def extract_submol(original_mol, subidx, dual_topology=[]):
    # collect internal bonds and linkings
    subidx_set = set(subidx)
    cappings = dict()
    linking_queue = Queue()
    CC_bond_lengths = []
    CH_bond_lengths = []
    searched = set()
    mol_conformer = original_mol.GetConformer()

    ts_mol = Chem.EditableMol(original_mol)
    for atom1, atom2, bondtype in dual_topology:
        original_bond = original_mol.GetBondBetweenAtoms(atom1, atom2)
        if original_bond is None:
            if bondtype is None:
                bondtype = Chem.BondType.SINGLE
            ts_mol.AddBond(atom1, atom2, bondtype)
            
    mol = ts_mol.GetMol()

    added_bonds = set()
    for atom1, atom2, bondtype in dual_topology:
        changing_bond = mol.GetBondBetweenAtoms(atom1, atom2)
        original_bond = original_mol.GetBondBetweenAtoms(atom1, atom2)
        if original_bond is None:
            added_bonds.add(changing_bond.GetIdx())
        elif original_bond.GetBondType() == Chem.BondType.SINGLE and bondtype != Chem.BondType.SINGLE:
            changing_bond.SetBondType(bondtype)
    
    for bond in mol.GetBonds():
        # completely internal
        bond_idx = bond.GetIdx()
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if begin_atom_idx in subidx_set and end_atom_idx in subidx_set:
            searched.add(bond_idx)
        else:
            if begin_atom_idx in subidx_set:
                linking_queue.put((begin_atom_idx, end_atom_idx))
            if end_atom_idx in subidx_set:
                linking_queue.put((end_atom_idx, begin_atom_idx))

        if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6:
            CC_bond_lengths.append(get_bond_lengths(mol_conformer, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        elif (bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 1) \
            or (bond.GetBeginAtom().GetAtomicNum() == 1 and bond.GetEndAtom().GetAtomicNum() == 6):
            CH_bond_lengths.append(get_bond_lengths(mol_conformer, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    CC_mean = np.mean(CC_bond_lengths)
    CC_std = np.std(CC_bond_lengths)
    CH_mean = np.mean(CH_bond_lengths)
    CH_std = np.std(CH_bond_lengths)

    # mol.UpdatePropertyCache(strict=False)

    while not linking_queue.empty():
        internal_atom_idx, external_atom_idx = linking_queue.get()
        internal_atom = mol.GetAtomWithIdx(internal_atom_idx)
        internal_atomic_num = internal_atom.GetAtomicNum()
        bond = mol.GetBondBetweenAtoms(internal_atom_idx, external_atom_idx)
        bond_idx = bond.GetIdx()
        external_atom = mol.GetAtomWithIdx(external_atom_idx)
        external_atomic_num = external_atom.GetAtomicNum()
        capping_flag = False
        # drop single internal H
        if internal_atomic_num == 1:
            if len(internal_atom.GetBonds()) > 1:
                searched.add(bond_idx)
        # cap single internal C
        elif internal_atomic_num == 6:
            # add C-H bond
            if external_atomic_num == 1:
                searched.add(bond_idx)
            # plan to cap C-C bond
            elif external_atomic_num == 6 and bond.GetBondType() == Chem.BondType.SINGLE and bond_idx not in added_bonds:
                if internal_atom_idx in cappings:
                    cappings[internal_atom_idx] = -1
                    cappings[external_atom_idx] = -1
                    searched.add(bond_idx)
                elif external_atom_idx not in cappings:
                    cappings[external_atom_idx] = internal_atom_idx
                    capping_flag = True
                elif cappings[external_atom_idx] != internal_atom_idx:
                    old_internal_atom_idx = cappings[external_atom_idx]
                    if old_internal_atom_idx != -1:
                        old_bond_idx = mol.GetBondBetweenAtoms(old_internal_atom_idx, external_atom_idx).GetIdx()
                        searched.add(old_bond_idx)
                        cappings[external_atom_idx] = -1
                    searched.add(bond_idx)            
            # add C-X bond and put X's other bonds into queue
            else:
                searched.add(bond_idx)
        else:
            searched.add(bond_idx)

        external_atom_bonds = external_atom.GetBonds()
        if (external_atomic_num != 1 or len(external_atom_bonds) > 1) and not capping_flag:
            for new_bond in external_atom_bonds:
                new_bond_idx = new_bond.GetIdx()
                if new_bond_idx not in searched:
                    linking_queue.put((external_atom_idx, new_bond.GetOtherAtomIdx(external_atom_idx)))

    atom_map = dict()
    # submol = Chem.EditableMol(original_mol)
    # for bond_idx in searched:
    #     if bond_idx not in added_bonds:
    #         submol.RemoveBond(bond_idx)
    submol = Chem.EditableMol(Chem.PathToSubmol(original_mol, [bond_idx for bond_idx in searched if bond_idx not in added_bonds], atomMap=atom_map))

    capping_positions = []
    for external_atom_idx, internal_atom_idx in cappings.items():
        if internal_atom_idx != -1:
            internal_atom_submol_idx = atom_map[internal_atom_idx]
            CC_bond_length = get_bond_lengths(mol_conformer, external_atom_idx, internal_atom_idx)
            capping_atom_submol_idx = submol.AddAtom(Chem.Atom(1))
            submol.AddBond(internal_atom_submol_idx, capping_atom_submol_idx, Chem.BondType.SINGLE)
            CC_bond_vector = mol_conformer.GetAtomPosition(external_atom_idx) - mol_conformer.GetAtomPosition(internal_atom_idx)
            CH_bond_vector = CC_bond_vector / CC_bond_length * ((CC_bond_length - CC_mean) / CC_std * CH_std + CH_mean)
            capping_positions.append((capping_atom_submol_idx, mol_conformer.GetAtomPosition(internal_atom_idx) + CH_bond_vector))

    submol = submol.GetMol()
    submol_conformer = submol.GetConformer()
    for capping_atom_submol_idx, capping_position in capping_positions:
        submol_conformer.SetAtomPosition(capping_atom_submol_idx, capping_position)
    
    try:
        submol.UpdatePropertyCache(strict=True)
    except:
        Chem.MolToMolFile(submol, "error.mol")
        raise RuntimeError("Error in updating property cache")
    return submol, atom_map


def extract_submol_with_center(mol, center_atom_idx, radius=5, dual_topology=[]):
    tree = BallTree(mol.GetConformer().GetPositions())
    subidx = tree.query_radius([mol.GetConformer().GetPositions()[center_atom_idx]], r=radius)[0]
    subidx = set(subidx)
    return extract_submol(mol, subidx, dual_topology)


def make_xyz_block(Za: List[int], Ra: List[List[float]], title: str="") -> str:
    xyz_lines = [f"{len(Za)}\n", f"{title}\n"]
    for Za_, Ra_ in zip(Za, Ra):
        xyz_lines.append(f"{REVERSED_PERIODIC_TABLE.loc[Za_]['atom_type']} {Ra_[0]} {Ra_[1]} {Ra_[2]}\n")
    return "".join(xyz_lines)


class Extractor:
    def __init__(self, 
        reference_mol_path: str,
        fragment_per_frame: int = 1,
        local_uncertainty_radius: float = 5,
        fragment_radius: float = 5
    ) -> None:
        self.reference_mol = MolFromMolFile(reference_mol_path, removeHs=False)
        self.fragment_per_frame = fragment_per_frame
        self.local_uncertainty_radius = local_uncertainty_radius
        self.fragment_radius = fragment_radius

    def build_fragment(self, y_pred: dict, xyzblocks: Optional[List[str]] = None, prefix: str = "") -> None:
        suppl = Chem.SDWriter(f"{prefix}_fragments.sdf")
        for frame_idx in tqdm(range(len(y_pred["Ra"]))):
            Ra = y_pred["Ra"][frame_idx]
            Za = y_pred["Za"][frame_idx]
            # gen dual topology
            xyzblock = xyzblocks[frame_idx] if xyzblocks is not None else make_xyz_block(Za, Ra)
            mol = MolFromXYZBlock(xyzblock)
            DetermineConnectivity(mol, charge=Chem.GetFormalCharge(self.reference_mol))
            dual_topology = []
            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                if self.reference_mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx) is None:
                    dual_topology.append((begin_atom_idx, end_atom_idx, None))
            
            if "Fa_std" in y_pred:
                Fa_std = np.array(y_pred["Fa_std"][frame_idx])
            elif "Fa_var" in y_pred:
                Fa_std = np.sqrt(np.array(y_pred["Fa_var"][frame_idx]))
            else:
                raise ValueError("Fa_std or Fa_var is not in the prediction result")
            nbs = NearestNeighbors(radius=self.local_uncertainty_radius)
            nbs.fit(Ra)
            neighbors = nbs.radius_neighbors(Ra, return_distance=False)
            local_uncertainty = []
            for i in range(len(neighbors)):
                local_uncertainty.append(np.mean(Fa_std[neighbors[i]]))
            sorted_atom_idx = np.argsort(local_uncertainty)

            coord = mol.GetConformer().GetPositions()
            self.reference_mol.GetConformer().SetPositions(coord)
            
            submol_indices = []
            for i in range(-1, -len(sorted_atom_idx) + 1, -1):
                submol, atom_map = extract_submol_with_center(self.reference_mol, sorted_atom_idx[i], self.fragment_radius, dual_topology)
                dup_flag = False
                for submol_index in submol_indices:
                    if atom_map.keys() == submol_index:
                        dup_flag = True
                        break
                if dup_flag:
                    continue
                submol_indices.append(atom_map.keys())
                suppl.write(submol)
                if len(submol_indices) == self.fragment_per_frame:
                    break
        suppl.close()
