import os
from argparse import ArgumentParser
from itertools import combinations

import ase.io
import numpy as np
import pandas as pd
from ase import units
from ase.calculators.plumed import Plumed
from ase.constraints import Hookean
from ase.md.langevin import Langevin
from enerzyme.bond.bond import pdb2mol
from enerzymette.altoolkit.get_index import get_indices
from enerzymette.plumed_config_generator.sammt import get_naive_sammt_config
from rdkit.Chem import GetFormalCharge
from xtb.ase.calculator import XTB

N_STEP = 26000
TIME_STEP = 0.5 * units.fs
LOG_INTERVAL = 20


def write_xyz(path, atoms=None):
    ase.io.write(path, atoms, append=True)


def _is_empty(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def resolve_master_list_row(master_list: str, enzyme: str, pdb_id: str | None):
    master_list_dir = os.path.dirname(os.path.abspath(master_list))
    df = pd.read_csv(master_list)
    mask = df["enzyme"] == enzyme
    if pdb_id is not None:
        mask &= df["pdb_id"] == pdb_id
    else:
        mask &= df["pdb_id"].apply(_is_empty)
    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"no row found for enzyme={enzyme!r}, pdb_id={pdb_id!r} in {master_list}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple rows found for enzyme={enzyme!r}, pdb_id={pdb_id!r} in {master_list}"
        )
    row = matches.iloc[0]
    if not _is_empty(row["pdb_file"]):
        pdb_file = os.path.join(master_list_dir, row["pdb_file"])
        sdf_file = None
    else:
        pdb_file = os.path.join("raw", f"{row['center']}.pdb")
        sdf_file = os.path.join("raw", f"{row['pdb_id']}_ligands.sdf")
    return pdb_file, sdf_file, row["substrate"], row["nucleophile"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory", type=str)
    parser.add_argument("-m", "--master_list", required=True, help="Path to master_list.csv", type=str)
    parser.add_argument("-e", "--enzyme", required=True, help="enzyme field in master_list", type=str)
    parser.add_argument("-p", "--pdb_id", help="pdb_id field in master_list", type=str, default=None)
    args = parser.parse_args()

    pdb_file, sdf_file, substrate, nucleophile = resolve_master_list_row(
        args.master_list, args.enzyme, args.pdb_id
    )

    system = ase.io.read(pdb_file)

    xtb_calculator = XTB(
        method="GFN1-xTB",
        solvent="THF",
    )
    system.calc = xtb_calculator

    charge = GetFormalCharge(pdb2mol(pdb_file, os.path.join(args.output_dir, "cluster.mol"), template_path=sdf_file))
    print("charge: ", charge)
    multiplicity = 1

    system.info.update({"charge": charge, "spin": multiplicity - 1})
    system.set_initial_charges([charge] + [0] * (len(system) - 1))
    system.set_initial_magnetic_moments([multiplicity - 1] + [0] * (len(system) - 1))

    fix_atoms = get_indices(pdb_file, 0, ["C_alpha", "O_water", "ions"])
    print("fix_atoms: ", fix_atoms)
    for i, j in combinations(fix_atoms, 2):
        system.constraints.append(
            Hookean(
                i,
                j,
                k=0.05 * units.Ha / units.Angstrom**2,
                rt=np.linalg.norm(system[i].position - system[j].position),
            )
        )

    setup = get_naive_sammt_config(
        system,
        integrate_config={"n_step": N_STEP},
        idx_start_from=1,
        dump_interval=LOG_INTERVAL,
        upper_bound=2,
        lower_bound=-2,
        warmup_steps=1000,
        reference_pdb_file=pdb_file,
        substrate=substrate,
        nucleophile=nucleophile,
    )
    print("setup: ", setup)

    metad_calc = Plumed(
        calc=xtb_calculator,
        input=setup,
        timestep=TIME_STEP,
        atoms=system,
        log=os.path.join(args.output_dir, "plumed.log"),
        kT=5000 * units.kB,
    )

    integrator2 = Langevin(
        system,
        timestep=TIME_STEP,
        temperature_K=500,
        friction=0.01 / units.fs,
        logfile=os.path.join(args.output_dir, "metad.log"),
        loginterval=LOG_INTERVAL,
    )
    integrator2.attach(write_xyz, interval=LOG_INTERVAL, atoms=system, path=os.path.join(args.output_dir, "metad-traj.xyz"))
    integrator2.run(steps=N_STEP)
