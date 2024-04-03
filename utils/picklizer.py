from pickle import dump
from tqdm import tqdm
import numpy as np
from glob import glob


def parse_xyz(xyz_file):
    with open(xyz_file) as f:
        lines = f.readlines()[2:]
    atom_types, coords = [], []
    for line in lines:
        atom_type, x, y, z = line.split()
        atom_types.append(atom_type.strip())
        coords.append([float(x), float(y), float(z)])
    return atom_types, np.array(coords)


def parse_terachem_grad(grad_file):
    with open(grad_file) as f:
        _ = f.readline()
        title = f.readline()
    energy = float(title.split()[6])
    grads = np.loadtxt(skiprows=2, usecols=(1,2,3))
    return energy, grads


def parse_chrg(chrg_file):
    return np.loadtxt(usecols=4)


def picklizer(file_lists, output, flavor="terachem", chrg=True):
    data = []
    for file_group in tqdm(file_lists):
        if flavor == "terachem":
            atom_type, coord = parse_xyz(file_group["coord"])
            energy, grad = parse_terachem_grad(file_group["grad"])
            chrg = parse_chrg(file_group["chrg"]) if chrg else None
        elif flavor == "xtb":
            pass

        data.append({
            "atom_type": atom_type,
            "coord": coord,
            "energy": energy,
            "grad": grad,
            "chrg": chrg
        })
    dump(data, output)
    

if __name__ == "__main__":
    pass