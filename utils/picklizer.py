from pickle import dump
from tqdm import tqdm
import numpy as np
from glob import glob
import pandas as pd


def parse_xyz(xyz_file):
    with open(xyz_file) as f:
        lines = f.readlines()[2:]
    atom_types, coords = [], []
    for line in lines:
        if not line.strip():
            break
        atom_type, x, y, z = line.split()
        atom_types.append(atom_type.strip())
        coords.append([float(x), float(y), float(z)])
    return atom_types, np.array(coords)


def parse_terachem_grad(grad_file):
    with open(grad_file) as f:
        _ = f.readline()
        title = f.readline()
    energy = float(title.split()[6])
    grads = np.loadtxt(grad_file, skiprows=2, usecols=(1,2,3)) / 0.5291772108 # Ha/Bohr to Ha/Ang !!!
    return energy, grads


def parse_terachem_energy(xyz_file):
    with open(xyz_file) as f:
        _ = f.readline()
        title = f.readline()
    energy = float(title.split()[0])
    return energy, None


def parse_chrg(chrg_file):
    return np.loadtxt(chrg_file, usecols=4)


def parse_dipole(dipole_file):
    df = pd.read_csv(dipole_file, sep="\s+", header=None)
    df[0] = df[0].apply(lambda x: x.strip(":"))
    df = df.set_index(0).loc[:,4:]
    df.columns = ["x", "y", "z"]
    return df


def parse_dipole_new(dipole_file):
    with open(dipole_file) as f:
        lines = f.readlines()
    x, y, z = lines[5].split()
    return np.array([float(x), float(y), float(z)]) * 0.2081943

def picklizer(file_lists, output, flavor="terachem", use_chrg=True, use_dipole=True, new_dipole=True, provide_Q=None):
    data = []
    dipole_file = None
    dipole_df = None
    for file_group in tqdm(file_lists):
        if flavor == "terachem":
            atom_type, coord = parse_xyz(file_group["coord"])
            if "grad" in file_group:
                energy, grad = parse_terachem_grad(file_group["grad"])
            else:
                energy, grad = parse_terachem_energy(file_group["energy"])
            chrg = parse_chrg(file_group["chrg"]) if use_chrg else None
            total_chrg = provide_Q
            if new_dipole:
                dipole = parse_dipole_new(file_group["dipole"])
            else:
                if file_group["dipole"] != dipole_file:
                    dipole_file = file_group["dipole"]
                    dipole_df = parse_dipole(dipole_file)
                dipole = dipole_df.loc[file_group["coord"].split("/")[-1], ["x", "y", "z"]].to_numpy() if use_dipole else None
        elif flavor == "xtb":
            pass
        datapoint = {
            "atom_type": atom_type,
            "coord": coord,
            "energy": energy,
            "grad": grad,
            "chrg": chrg,
            "dipole": dipole,
            "total_chrg": total_chrg
        }
        datapoint.update({
            k: v for k, v in file_group.items() if k not in datapoint.keys()
        })
        datapoint = {k: v for k, v in datapoint.items() if v is not None}
        data.append(datapoint)
    with open(output, "wb") as f:
        dump(data, f)
    return data
    

if __name__ == "__main__":
    pass