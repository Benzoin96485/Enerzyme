from typing import Optional
from argparse import ArgumentParser
import os
from shutil import copy
import pandas as pd
import ase.io
from enerzyme.bond.bond import pdb2mol
from rdkit.Chem import GetFormalCharge
from enerzymette.terachem.io import write_terachem_input as _write_terachem_input
from enerzymette.altoolkit.get_index import get_indices
from enerzymette.plumed_config_generator.sammt import get_sammt_index

PCM_RADII_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcm_radii")

def parse_xyz_block(xyz_path: str) -> str:
    xyz_blocks = []
    with open(xyz_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            xyz_block = [line]
            n_atoms = int(line.strip())
            xyz_block.append(f.readline())
            for _ in range(n_atoms):
                xyz_block.append(f.readline())
            xyz_blocks.append(xyz_block)
    return xyz_blocks

def write_terachem_input(output_path: str, coordinates_path: str, charge: int, scr_dir: str, constraint_freeze_info: dict = None, constraint_scan_info: dict = None, guess_path: Optional[str] = None) -> None:
    info = {
        "main": {
            "run": "minimize",
            "new_minimizer": "yes",
            "coordinates": coordinates_path,
            "basis": "6-31gs",
            "method": "b3lyp",
            "charge": charge,
            "spinmult": 1,
            "maxit": 1000,
            "dftd": "d3",
            "scrdir": scr_dir,
            "pcm": "cosmo",
            "epsilon": 10,
            "scf": "diis+a",
            "pcm_radii": "read",
            "pcm_radii_file": "pcm_radii",
        } | ({"guess": guess_path} if guess_path is not None else {}),
        "constraint_freeze": constraint_freeze_info,
    } | ({"constraint_scan": constraint_scan_info} if constraint_scan_info is not None else {})
    _write_terachem_input(info, terachem_input_file=output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--sbatch_head", help="Path to the sbatch head file", type=str, default=None)
    parser.add_argument("-i", "--init_reactant", help="Path to the initial reactant file", type=str, default=None)
    parser.add_argument("-m", "--master_list", required=True, help="Path to master_list.csv", type=str)
    parser.add_argument("-e", "--enzyme", required=True, help="enzyme field in master_list", type=str)
    parser.add_argument("-p", "--pdb_id", help="pdb_id field in master_list", type=str, default=None)
    parser.add_argument("-t", "--tmpdir", help="Temporary directory", type=str, default=".")
    parser.add_argument("-g", "--guess", action="store_true", help="Use MO guess from last step", default=False)
    args = parser.parse_args()

    def _is_empty(value) -> bool:
        return pd.isna(value) or str(value).strip() == ""

    master_list_dir = os.path.dirname(os.path.abspath(args.master_list))
    df = pd.read_csv(args.master_list)
    mask = df["enzyme"] == args.enzyme
    if args.pdb_id is not None:
        mask &= df["pdb_id"] == args.pdb_id
    else:
        mask &= df["pdb_id"].apply(_is_empty)
    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"no row found for enzyme={args.enzyme!r}, pdb_id={args.pdb_id!r} in {args.master_list}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple rows found for enzyme={args.enzyme!r}, pdb_id={args.pdb_id!r} in {args.master_list}"
        )
    row = matches.iloc[0]
    resname_substrate = row["substrate"]
    atomname_nucleophile = row["nucleophile"]
    target_x1 = float(row["target_x1"])
    if not _is_empty(row["pdb_file"]):
        pdb_file = os.path.join(master_list_dir, row["pdb_file"])
        sdf_file = None
    else:
        pdb_file = os.path.join("raw", f"{row['center']}.pdb")
        sdf_file = os.path.join("raw", f"{row['pdb_id']}_ligands.sdf")

    copy(PCM_RADII_PATH, "pcm_radii")
    if args.init_reactant is not None:
        copy(args.init_reactant, "init_reactant.xyz")
        atoms = ase.io.read(args.init_reactant)
    else:
        atoms = ase.io.read(pdb_file)
        atoms.write("init_reactant.xyz", format="xyz")

    charge = GetFormalCharge(pdb2mol(pdb_file, "cluster.mol", template_path=sdf_file))
    n_atoms = len(atoms)
    constraint_freeze_info = {
        "xyz": get_indices(pdb_path=pdb_file, idx_start_from=1, index_types=["backbone"]),
    }
    _, index_methyl_carbon, index_nucleophile = get_sammt_index(
        idx_start_from=1, 
        reference_pdb_file=pdb_file, 
        substrate=resname_substrate, 
        nucleophile=atomname_nucleophile,
    )
    constraint_scan_info = {
        "bond": {
            "x0": 0, # this will be updated later
            "x1": target_x1,
            "num": 25,
            "i0": index_methyl_carbon,
            "i1": index_nucleophile,
        }
    }
    
    write_terachem_input(
        output_path="reactant_opt.in",
        coordinates_path="init_reactant.xyz",
        charge=charge,
        scr_dir=os.path.join("__TMPDIR__", "scr_reactant_opt"),
        constraint_freeze_info=constraint_freeze_info,
        constraint_scan_info=None,
        guess_path=os.path.join("__TMPDIR__", "scr_reactant_opt", "c0") if args.guess else None,
    )
    write_terachem_input(
        output_path="scan.in",
        coordinates_path="reactant.xyz",
        charge=charge,
        scr_dir=os.path.join("__TMPDIR__", "scr_scan"),
        constraint_freeze_info=constraint_freeze_info,
        constraint_scan_info=constraint_scan_info,
        guess_path=os.path.join("__TMPDIR__", "scr_scan", "c0") if args.guess else None,
    )
    write_terachem_input(
        output_path="product_opt.in",
        coordinates_path="init_product.xyz",
        charge=charge,
        scr_dir=os.path.join("__TMPDIR__", "scr_product_opt"),
        constraint_freeze_info=constraint_freeze_info,
        constraint_scan_info=None,
        guess_path=os.path.join("__TMPDIR__", "scr_product_opt", "c0") if args.guess else None,
    )

    sbatch_head = None
    if args.sbatch_head is not None:
        with open(args.sbatch_head, "r") as f:
            sbatch_head = f.read()

    with open("job.sh", "w") as f:
        if sbatch_head is not None:
            f.write(sbatch_head)
        f.write(f'sed -i "s|__TMPDIR__|{args.tmpdir}|g" reactant_opt.in scan.in product_opt.in\n')
        f.write(f"terachem reactant_opt.in > reactant_opt.out\n")
        f.write(f"tail -{n_atoms+2} {os.path.join(args.tmpdir, 'scr_reactant_opt', 'optim.xyz')} > reactant.xyz\n")
        f.write(f"enerzymette update_terachem_scan -i scan.in -s reactant.xyz -o scan.in\n")
        f.write(f"terachem scan.in > scan.out\n")
        f.write(f"cp {os.path.join(args.tmpdir, 'scr_scan', 'scan_optim.xyz')} scan_optim.xyz\n")
        f.write(f"tail -{n_atoms+2} scan_optim.xyz > init_product.xyz\n")
        f.write(f"terachem product_opt.in > product_opt.out\n")
        f.write(f"tail -{n_atoms+2} {os.path.join(args.tmpdir, 'scr_product_opt', 'optim.xyz')} > product.xyz\n")
