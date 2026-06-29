from enerzymette.nebtoolkit.io import write_orca_neb_in
from argparse import ArgumentParser
import os
from shutil import copy
import pandas as pd
from enerzyme.bond.bond import pdb2mol
from rdkit.Chem import GetFormalCharge
from enerzymette.altoolkit.get_index import get_indices

PCM_RADII_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcm_radii")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--master_list", required=True, help="Path to master_list.csv", type=str)
    parser.add_argument("-e", "--enzyme", required=True, help="enzyme field in master_list", type=str)
    parser.add_argument("-p", "--pdb_id", help="pdb_id field in master_list", type=str, default=None)
    parser.add_argument("-n", "--n_images", help="Number of images", type=int, default=8)
    parser.add_argument("-b", "--sbatch_head", help="Path to the sbatch head file", type=str, default=None)
    parser.add_argument("--reactant", required=True, help="Path to the initial reactant file", type=str)
    parser.add_argument("--product", required=True, help="Path to the initial product file", type=str)
    parser.add_argument("-t", "--ts_guess", help="Path to the ts initial guess file", type=str, default=None)
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
    if not _is_empty(row["pdb_file"]):
        pdb_file = os.path.join(master_list_dir, row["pdb_file"])
        sdf_file = None
    else:
        pdb_file = os.path.join("raw", f"{row['center']}.pdb")
        sdf_file = os.path.join("raw", f"{row['pdb_id']}_ligands.sdf")

    charge = GetFormalCharge(pdb2mol(pdb_file, "cluster.mol", template_path=sdf_file))
    constraint_freeze_xyz = get_indices(pdb_path=pdb_file, idx_start_from=1, index_types=["backbone"])

    template_terachem_input = f'''run gradient
basis 6-31gs
method b3lyp
charge {charge}
maxit 1000
pcm cosmo
epsilon 10
scf diis+a
dftd d3
nstep 1000
pcm_radii read
pcm_radii_file pcm_radii
end
'''
    sbatch_head = None
    if args.sbatch_head is not None:
        with open(args.sbatch_head, "r") as f:
            sbatch_head = f.read()

    copy(PCM_RADII_PATH, "pcm_radii")
    if not os.path.exists("reactant.xyz"):
        copy(args.reactant, "reactant.xyz")
    if not os.path.exists("product.xyz"):
        copy(args.product, "product.xyz")
    if args.ts_guess is not None and not os.path.exists("ts.xyz"):
        copy(args.ts_guess, "ts.xyz")

    with open("template.terachem.inp", "w") as f:
        f.write(template_terachem_input)
    with open("terachem_wrapper.sh", "w") as f:
        f.write("enerzymette orca_terachem_request -i $1 -t template.terachem.inp\n")
    wrapper_path = os.path.abspath("terachem_wrapper.sh")
    write_orca_neb_in(
        neb_in_path="neb.inp",
        wrapper_path=wrapper_path,
        n_images=args.n_images,
        use_ts=(args.ts_guess is not None),
        constraint_freeze_xyz=constraint_freeze_xyz,
        charge=charge
    )

    with open("job.sh", "w") as f:
        if sbatch_head is not None:
            f.write(sbatch_head)
        f.write(f"chmod +x terachem_wrapper.sh\n")
        f.write(f"$ORCA_PATH neb.inp > neb.out\n")
