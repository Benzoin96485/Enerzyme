from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage


def pdb2mol(pdb_path: str, mol_path: str, img_path: str='') -> None:
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    delete_bonds = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetNumImplicitHs() >= 1 and end_atom.GetNumImplicitHs() >= 1:
            bond.SetBondType(Chem.BondType.DOUBLE)
            begin_atom.UpdatePropertyCache(strict=False)
            end_atom.UpdatePropertyCache(strict=False)
        begin_atom_idx = begin_atom.GetIdx()
        end_atom_idx = end_atom.GetIdx()
        begin_res_name = begin_atom.GetPDBResidueInfo().GetResidueName().strip()
        end_res_name = end_atom.GetPDBResidueInfo().GetResidueName().strip()
        begin_atom_name = begin_atom.GetPDBResidueInfo().GetName().strip()
        end_atom_name = end_atom.GetPDBResidueInfo().GetName().strip()
        
        if begin_res_name in {"MGT"} and end_res_name in {"MGT"}:
            if {begin_atom_name, end_atom_name} == {"PA", "PB"}:
                delete_bonds.append((begin_atom_idx, end_atom_idx))
            if {begin_atom_name, end_atom_name} in [{"PA", "O2A"}, {"O2B", "PB"}]:
                bond.SetBondType(Chem.BondType.DOUBLE)
                if begin_atom.GetSymbol() == "O":
                    begin_atom.SetFormalCharge(0)
                    end_atom.SetNoImplicit(True)
                else:
                    end_atom.SetFormalCharge(0)
                    begin_atom.SetNoImplicit(True)
                # begin_atom.UpdatePropertyCache(strict=False)
                # end_atom.UpdatePropertyCache(strict=False)
            if {begin_atom_name, end_atom_name} == {"C8", "N7"}:
                if begin_atom.GetSymbol() == "C":
                    begin_atom.SetFormalCharge(0)
                    end_atom.SetFormalCharge(1)
                else:
                    begin_atom.SetFormalCharge(1)
                    end_atom.SetFormalCharge(0)
                bond.SetBondType(Chem.BondType.DOUBLE)
                # begin_atom.UpdatePropertyCache(strict=False)
                # end_atom.UpdatePropertyCache(strict=False)
    mol.UpdatePropertyCache(strict=False)
    broken_HX = []
    for atom in mol.GetAtoms():
        if atom.GetNumImplicitHs() == 1:
            old_charge = atom.GetFormalCharge()
            if atom.GetAtomicNum() == 7 and old_charge == 1:
                atom.SetFormalCharge(0)
            if atom.GetAtomicNum() == 8 and old_charge == 0:
                atom.SetFormalCharge(-1)
            if atom.GetAtomicNum() == 6 and old_charge == 0:
                atom.SetFormalCharge(-1)
            if atom.GetAtomicNum() == 16 and old_charge == 0 and atom.GetExplicitValence() == 3:
                atom.SetFormalCharge(1)
            # atom.UpdatePropertyCache(strict=False)
            if atom.GetAtomicNum() == 1:
                begin_atom_name = atom.GetPDBResidueInfo().GetName().strip()
                for atom_X in mol.GetAtoms():
                    if atom_X.GetPDBResidueInfo().GetName().strip() == begin_atom_name[1:] \
                        and atom.GetPDBResidueInfo().GetResidueName().strip() == atom_X.GetPDBResidueInfo().GetResidueName().strip():
                        atom_X.SetFormalCharge(0)
                        broken_HX.append((atom.GetIdx(), atom_X.GetIdx()))

    # mol.UpdatePropertyCache(strict=False)
    edit_mol = Chem.EditableMol(mol)
    for start_idx, end_idx in delete_bonds:
        edit_mol.RemoveBond(start_idx, end_idx)
    for start_idx, end_idx in broken_HX:
        edit_mol.AddBond(start_idx, end_idx, Chem.BondType.SINGLE)
    mol = edit_mol.GetMol()
    mol.UpdatePropertyCache(strict=False)

    #mol = Chem.MolFromPDBFile("/home/gridsan/wlluo/multiscale/qmcluster/HcgC/HcgC-reactant-opt/cluster.pdb", removeHs=False)

    
    
    if img_path:
        # save img as a png file
        frag_assign = []
        raw_frags = Chem.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=frag_assign, sanitizeFrags=True)
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