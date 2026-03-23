from queue import Queue
from typing import List, Optional, Tuple
import random
from tqdm import tqdm
import numpy as np
from rdkit.Chem import MolFromMolFile, MolFromXYZBlock, Mol
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity
from sklearn.neighbors import NearestNeighbors, BallTree
from rdkit import Chem
from ..data.transform import REVERSED_PERIODIC_TABLE


def get_bond_lengths(conformer, begin_atom_idx, end_atom_idx):
    return np.linalg.norm(conformer.GetAtomPosition(begin_atom_idx) - conformer.GetAtomPosition(end_atom_idx))


def extract_submol(original_mol: Mol, subidx: List[int], dual_topology: List[Tuple[int, int, Optional[int]]]=[]):
    # collect internal bonds and linkings
    subidx_set = set(subidx)
    cappings = dict()
    linking_queue = Queue()
    CC_bond_lengths = []
    CH_bond_lengths = []
    searched = set()
    mol_conformer = original_mol.GetConformer()
    original_pair_set = set([frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in original_mol.GetBonds()])

    ts_mol = Chem.EditableMol(original_mol)
    dual_topology_pair_set = set()
    for atom1, atom2, bondtype in dual_topology:
        dual_topology_pair_set.add(frozenset((atom1, atom2)))
        if bondtype is None:
            ts_mol.AddBond(atom1, atom2, Chem.BondType.SINGLE)
            
    mol = ts_mol.GetMol()

    added_bonds = set()
    for atom1, atom2, bondtype in dual_topology:
        changing_bond = mol.GetBondBetweenAtoms(atom1, atom2)
        if bondtype is None:
            added_bonds.add(changing_bond.GetIdx())
        # elif original_bond.GetBondType() == Chem.BondType.SINGLE and bondtype != Chem.BondType.SINGLE:
        #     changing_bond.SetBondType(bondtype)
    
    if len(dual_topology_pair_set) > 0:
        overlap_pair_set = original_pair_set & dual_topology_pair_set
    else:
        overlap_pair_set = original_pair_set

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

        if frozenset((begin_atom_idx, end_atom_idx)) in overlap_pair_set:
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


def extract_submol_with_center(mol: Mol, center_atom_indices: List[int], radius: float=5, dual_topology: List[Tuple[int, int, Optional[int]]]=[], must_include_indices: List[int]=[]):
    '''
    Extract a fragment from a molecule with center atoms.

    Params:
    -------
    mol: Mol
        The molecule to be extracted.
    center_atom_indices: List[int]
        The indices of the center atoms.
    radius: float
        The radius of the fragmentation.
    dual_topology: List[Tuple[int, int, Optional[int]]]
        The dual topology of the molecule.
    '''
    tree = BallTree(mol.GetConformer().GetPositions())
    subidx = set()
    for center_atom_index in center_atom_indices:
        subidx.update(tree.query_radius([mol.GetConformer().GetPositions()[center_atom_index]], r=radius)[0])
    subidx.update(must_include_indices)
    return extract_submol(mol, subidx, dual_topology)


def make_xyz_block(Za: List[int], Ra: List[List[float]], title: str="") -> str:
    xyz_lines = [f"{len(Za)}\n", f"{title}\n"]
    for Za_, Ra_ in zip(Za, Ra):
        xyz_lines.append(f"{REVERSED_PERIODIC_TABLE.loc[Za_]['atom_type']} {Ra_[0]:.15f} {Ra_[1]:.15f} {Ra_[2]:.15f}\n")
    return "".join(xyz_lines)


class Extractor:
    def __init__(self, 
        reference_mol_path: str,
        fragment_per_frame: int = 1,
        local_uncertainty_radius: float = 5,
        fragment_radius: float = 5,
        n_centers: int = 1,
        must_include_indices: List[int] = [],
        extract_method: str = "local_uncertainty"
    ) -> None:
        '''
        Extractor class for extracting fragments from molecules.

        Params:
        -------
        reference_mol_path: str
            The path to the reference molecule.
        fragment_per_frame: int
            The number of fragments to be extracted per frame.
        local_uncertainty_radius: float
            The radius of the local uncertainty quantification.
        fragment_radius: float
            The radius of the fragmentation.
        n_centers: int
            The number of centers to be used for extracting only one fragment.
        '''
        if reference_mol_path.endswith(".sdf"):
            self.reference_mol = list(Chem.SDMolSupplier(reference_mol_path, removeHs=False))
        else:
            self.reference_mol = [MolFromMolFile(reference_mol_path, removeHs=False)]
        self.fragment_per_frame = fragment_per_frame
        self.local_uncertainty_radius = local_uncertainty_radius
        self.fragment_radius = fragment_radius
        self.n_centers = n_centers
        self.must_include_indices = must_include_indices
        self.extract_method = extract_method

    def build_fragment(self, y_pred: dict, xyzblocks: Optional[List[str]] = None, prefix: str = "") -> None:
        '''
        Build fragments from the prediction results.

        Params:
        -------
        y_pred: dict
            The prediction results with uncertainty quantification.
        xyzblocks: Optional[List[str]]
            The xyz blocks of the molecules.
        prefix: str
            The prefix of the output sdf file.
        '''
        suppl = Chem.SDWriter(f"{prefix}_fragments.sdf")
        for frame_idx in tqdm(range(len(y_pred["Ra"]))):
            if len(self.reference_mol) == 1:
                reference_mol = self.reference_mol[0]
            else:
                reference_mol = self.reference_mol[frame_idx]
            Ra = y_pred["Ra"][frame_idx]
            Za = y_pred["Za"][frame_idx]
            # gen dual topology
            xyzblock = xyzblocks[frame_idx] if xyzblocks is not None else make_xyz_block(Za, Ra)
            mol = MolFromXYZBlock(xyzblock)
            DetermineConnectivity(mol, charge=Chem.GetFormalCharge(reference_mol))
            dual_topology = []
            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                bond = reference_mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                dual_topology.append((begin_atom_idx, end_atom_idx, bond.GetBondType() if bond is not None else None))

            if self.extract_method == "local_uncertainty":
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
                sorted_atom_idx = np.argsort(local_uncertainty[::-1])

                coord = mol.GetConformer().GetPositions()
                reference_mol.GetConformer().SetPositions(coord)
            elif self.extract_method == "random":
                sorted_atom_idx = list(range(len(Ra)))
                random.shuffle(sorted_atom_idx)
            
            submol_indices = []
            for i in range(0, len(sorted_atom_idx), self.n_centers):
                submol, atom_map = extract_submol_with_center(reference_mol, sorted_atom_idx[i: i + self.n_centers], self.fragment_radius, dual_topology, self.must_include_indices)
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
