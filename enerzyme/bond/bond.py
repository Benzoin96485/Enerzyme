from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.Draw import MolsToGridImage


def get_atom_map(pdb_coords: list, mol_coords: list, tol: float=0.01):
    pdb_coords = np.array(pdb_coords)
    mol_coords = np.array(mol_coords)
    dists = np.linalg.norm(pdb_coords[:, None] - mol_coords, axis=2)
    min_dists = np.min(dists, axis=1)
    raw_atom_map = np.argmin(dists, axis=1)
    raw_atom_map[np.where(min_dists > tol)] = -1
    clean_atom_map = [-1] * len(mol_coords)
    for i in range(len(raw_atom_map)):
        if raw_atom_map[i] == -1:
            continue
        clean_atom_map[raw_atom_map[i]] = i
    # clean_atom_map[i] is the index of the atom in the pdb that is closest to the i-th atom in the mol
    return clean_atom_map


def bond_with_template(mol: Chem.Mol, pdb_path: str, template_path: str) -> None:
    template_mols = {tmp_mol.GetProp("_Name"): tmp_mol for tmp_mol in SDMolSupplier(template_path, removeHs=False)}
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()

    res_pointers = dict()
    res_lens = defaultdict(int)
    pdb_coords = []
    cur_res_key = ''
    atom_cnt = 0
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_resname = line[17:20].strip()
            atom_chainid = line[21].strip()
            atom_resid = line[22:26].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            pdb_coords.append((x, y, z))
            res_key = f"{atom_resname}_{atom_chainid}_{atom_resid}"
            if res_key != cur_res_key:
                cur_res_key = res_key
                res_pointers[cur_res_key] = atom_cnt
            atom_cnt += 1
            res_lens[cur_res_key] += 1
    pdb_coords = np.array(pdb_coords)

    for res_key, res_pointer in res_pointers.items():
        if res_key in template_mols:
            res_len = res_lens[res_key]
            template_mol = template_mols[res_key]
            mol_coords = np.array(template_mol.GetConformers()[0].GetPositions())
            atom_map = get_atom_map(pdb_coords, mol_coords)
            edit_mol = Chem.EditableMol(mol)

            # remove all bonds in edited mol
            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                if begin_atom_idx >= res_pointer and begin_atom_idx < res_pointer + res_len \
                    and end_atom_idx >= res_pointer and end_atom_idx < res_pointer + res_len:
                    edit_mol.RemoveBond(begin_atom_idx, end_atom_idx)

            # add bonds in edited mol
            for bond in template_mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                if begin_atom.GetAtomicNum() == 0 or end_atom.GetAtomicNum() == 0:
                    continue
                begin_atom_idx = atom_map[begin_atom.GetIdx()]
                end_atom_idx = atom_map[end_atom.GetIdx()]
                edit_mol.AddBond(begin_atom_idx, end_atom_idx, bond.GetBondType())

            mol = edit_mol.GetMol()
            # adjust atom properties
            for atom in template_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    continue
                atom_idx = atom_map[atom.GetIdx()]
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(atom.GetFormalCharge())
                mol.GetAtomWithIdx(atom_idx).SetNoImplicit(True)
            mol.UpdatePropertyCache(strict=False)
    return mol


def pdb2mol(pdb_path: str, mol_path: str, img_path: str='', template_path: str='') -> None:
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)

    if template_path:
        mol = bond_with_template(mol, pdb_path, template_path)

    # general bond fix
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetNumImplicitHs() >= 1 and end_atom.GetNumImplicitHs() >= 1:
            bond.SetBondType(Chem.BondType.DOUBLE)
            begin_atom.UpdatePropertyCache(strict=False)
            end_atom.UpdatePropertyCache(strict=False)
        begin_res_name = begin_atom.GetPDBResidueInfo().GetResidueName().strip()
        end_res_name = end_atom.GetPDBResidueInfo().GetResidueName().strip()
        begin_atom_name = begin_atom.GetPDBResidueInfo().GetName().strip()
        end_atom_name = end_atom.GetPDBResidueInfo().GetName().strip()

        # Fix GLU carboxylate
        if begin_res_name in {"GLU"} and end_res_name in {"GLU"}:
            if {begin_atom_name, end_atom_name} == {"OE1", "HOE1"}:
                bond.SetBondType(Chem.BondType.SINGLE)
                begin_atom.SetFormalCharge(0)
                end_atom.SetFormalCharge(0)
                begin_atom.SetNoImplicit(True)
                end_atom.SetNoImplicit(True)
                if begin_atom_name == "OE1":
                    OE1 = begin_atom
                if end_atom_name == "OE1":
                    OE1 = end_atom
                for OE1_bond in OE1.GetBonds():
                    OE1_bond_begin_atom = OE1_bond.GetBeginAtom()
                    OE1_bond_begin_atom_name = OE1_bond_begin_atom.GetPDBResidueInfo().GetName().strip()
                    OE1_bond_end_atom = OE1_bond.GetEndAtom()
                    OE1_bond_end_atom_name = OE1_bond_end_atom.GetPDBResidueInfo().GetName().strip()
                    if OE1_bond_begin_atom_name == "CD":
                        CD = OE1_bond_begin_atom
                    elif OE1_bond_end_atom_name == "CD":
                        CD = OE1_bond_end_atom
                    else:
                        continue
                    OE1_bond.SetBondType(Chem.BondType.SINGLE)
                for CD_bond in CD.GetBonds():
                    CD_bond_begin_atom = CD_bond.GetBeginAtom()
                    CD_bond_begin_atom_name = CD_bond_begin_atom.GetPDBResidueInfo().GetName().strip()
                    CD_bond_end_atom = CD_bond.GetEndAtom()
                    CD_bond_end_atom_name = CD_bond_end_atom.GetPDBResidueInfo().GetName().strip()
                    if CD_bond_begin_atom_name == "OE2":
                        OE2 = CD_bond_begin_atom
                    elif CD_bond_end_atom_name == "OE2":
                        OE2 = CD_bond_end_atom
                    else:
                        continue
                    CD_bond.SetBondType(Chem.BondType.DOUBLE)
                    OE2.SetFormalCharge(0)

        # Fix ASP carboxylate
        if begin_res_name in {"ASP"} and end_res_name in {"ASP"}:
            if {begin_atom_name, end_atom_name} == {"OD1", "HOD1"}:
                bond.SetBondType(Chem.BondType.SINGLE)
                begin_atom.SetFormalCharge(0)
                end_atom.SetFormalCharge(0)
                begin_atom.SetNoImplicit(True)
                end_atom.SetNoImplicit(True)
                if begin_atom_name == "OD1":
                    OD1 = begin_atom
                if end_atom_name == "OD1":
                    OD1 = end_atom
                for OD1_bond in OD1.GetBonds():
                    OD1_bond_begin_atom = OD1_bond.GetBeginAtom()
                    OD1_bond_begin_atom_name = OD1_bond_begin_atom.GetPDBResidueInfo().GetName().strip()
                    OD1_bond_end_atom = OD1_bond.GetEndAtom()
                    OD1_bond_end_atom_name = OD1_bond_end_atom.GetPDBResidueInfo().GetName().strip()
                    if OD1_bond_begin_atom_name == "CG":
                        CG = OD1_bond_begin_atom
                    elif OD1_bond_end_atom_name == "CG":
                        CG = OD1_bond_end_atom
                    else:
                        continue
                    OD1_bond.SetBondType(Chem.BondType.SINGLE)
                for CG_bond in CG.GetBonds():
                    CG_bond_begin_atom = CG_bond.GetBeginAtom()
                    CG_bond_begin_atom_name = CG_bond_begin_atom.GetPDBResidueInfo().GetName().strip()
                    CG_bond_end_atom = CG_bond.GetEndAtom()
                    CG_bond_end_atom_name = CG_bond_end_atom.GetPDBResidueInfo().GetName().strip()
                    if CG_bond_begin_atom_name == "OD2":
                        OD2 = CG_bond_begin_atom
                    elif CG_bond_end_atom_name == "OD2":
                        OD2 = CG_bond_end_atom
                    else:
                        continue
                    CG_bond.SetBondType(Chem.BondType.DOUBLE)
                    OD2.SetFormalCharge(0)

    mol.UpdatePropertyCache(strict=False)
    # general atom fix
    
    broken_HX = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetExplicitValence() == 3:
            for C_bond in atom.GetBonds():
                C_bond_begin_atom = C_bond.GetBeginAtom()
                C_bond_end_atom = C_bond.GetEndAtom()
                if C_bond_begin_atom.GetAtomicNum() == 7:
                    N = C_bond_begin_atom
                elif C_bond_end_atom.GetAtomicNum() == 7:
                    N = C_bond_end_atom
                else:
                    continue
                if N.GetFormalCharge() == -1:
                    if C_bond.GetBondType() == Chem.BondType.SINGLE:
                        C_bond.SetBondType(Chem.BondType.DOUBLE)
                    elif C_bond.GetBondType() == Chem.BondType.DOUBLE:
                        C_bond.SetBondType(Chem.BondType.TRIPLE)
                    N.SetFormalCharge(0)
                    N.UpdatePropertyCache(strict=False)
                    atom.UpdatePropertyCache(strict=False)
                    break
        if atom.GetNumImplicitHs() == 1:
            old_charge = atom.GetFormalCharge()
            if atom.GetAtomicNum() == 7:
                if old_charge == 1:
                    atom.SetFormalCharge(0)
                elif old_charge == 0:
                    atom.SetFormalCharge(-1)
            if atom.GetAtomicNum() == 8 and old_charge == 0:
                atom.SetFormalCharge(-1)
            if atom.GetAtomicNum() == 6 and old_charge == 0:
                for C_bond in atom.GetBonds():
                    C_bond_begin_atom = C_bond.GetBeginAtom()
                    C_bond_end_atom = C_bond.GetEndAtom()
                    if C_bond_begin_atom.GetAtomicNum() == 7:
                        N = C_bond_begin_atom
                    elif C_bond_end_atom.GetAtomicNum() == 7:
                        N = C_bond_end_atom
                    else:
                        continue
                    if N.GetNumImplicitHs() == 0:
                        N.SetFormalCharge(1)
                        C_bond.SetBondType(Chem.BondType.DOUBLE)
                        atom.UpdatePropertyCache(strict=False)
                        N.UpdatePropertyCache(strict=False)
                        break
            if atom.GetAtomicNum() == 16:
                if old_charge == 0 and atom.GetExplicitValence() == 3:
                    atom.SetFormalCharge(1)
                if old_charge == 0 and atom.GetExplicitValence() == 1:
                    atom.SetFormalCharge(-1)
            if atom.GetAtomicNum() == 1:
                begin_atom_name = atom.GetPDBResidueInfo().GetName().strip()
                for atom_X in mol.GetAtoms():
                    if atom_X.GetPDBResidueInfo().GetName().strip() == begin_atom_name[1:] \
                        and atom.GetPDBResidueInfo().GetResidueName().strip() == atom_X.GetPDBResidueInfo().GetResidueName().strip():
                        atom_X.SetFormalCharge(0)
                        broken_HX.append((atom.GetIdx(), atom_X.GetIdx()))

    edit_mol = Chem.EditableMol(mol)
    for start_idx, end_idx in broken_HX:
        edit_mol.AddBond(start_idx, end_idx, Chem.BondType.SINGLE)
    mol = edit_mol.GetMol()
    mol.UpdatePropertyCache(strict=False)

    if img_path:
        # save img as a png file
        frag_assign = []
        raw_frags = Chem.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=frag_assign, sanitizeFrags=False)
        processed_frags = []
        for i, frag in enumerate(raw_frags):
            processed_frag = frag.__copy__()
            Chem.rdDepictor.Compute2DCoords(processed_frag)
            for j, atom in enumerate(processed_frag.GetAtoms()):
                atom.SetProp("atomNote", str(frag_assign[i][j]))
            processed_frags.append(processed_frag)
        img = MolsToGridImage(processed_frags[:], subImgSize=(1000, 1000), molsPerRow=3)
        img.save(img_path)

    Chem.MolToMolFile(mol, mol_path)
    return mol