import os
import pathlib
import pandas as pd
import numpy as np
from threading import Thread
from tqdm import tqdm
from ..utils import logger

PERIODIC_TABLE_PATH = os.path.join(
    pathlib.Path(__file__).parent.resolve(),
    'periodic-table.csv'
)
PERIODIC_TABLE = pd.read_csv(PERIODIC_TABLE_PATH, index_col="atom_type")


def parse_Za(atom_types):
    if isinstance(atom_types[0], int):
        return np.array(atom_types)
    elif isinstance(atom_types[0], str):
        return PERIODIC_TABLE.loc[atom_types]["Za"].to_numpy()
    else:
        logger.info("Parsing atom type")
        Zas = []
        for atom_types_ in tqdm(atom_types):
            Zas.append(parse_Za(atom_types_))
        return Zas


def load_atomic_energy(atomic_energy_path):
    if os.path.exists(atomic_energy_path):
        atomic_energies = pd.read_csv(atomic_energy_path)
        atomic_energies["Za"] = parse_Za(atomic_energies["atom_type"])
        return atomic_energies.set_index("Za")
    else:
        raise FileNotFoundError(f"Atomic energy file {atomic_energy_path} not found!")


class AtomicEnergyTransform:
    def __init__(self, atomic_energy_path, *args, **kwargs):
        self.atomic_energies = load_atomic_energy(atomic_energy_path)
        self.transform_type = "shift"

    def transform(self, new_input):
        for i in range(len(new_input["E"])):
            new_input["E"][i] -= sum(self.atomic_energies.loc[new_input["Za"][i]]["atomic_energy"])
    
    def inverse_transform(self, new_output):
        for i in range(len(new_output["E"])):
            new_output["E"][i] += sum(self.atomic_energies.loc[new_output["Za"][i]]["atomic_energy"])
    

class NegativeGradientTransform:
    def __init__(self, *args, **kwargs):
        self.transform_type = "scale"

    def transform(self, new_input):
        for i in range(len(new_input["Fa"])):
            new_input["Fa"][i] = -new_input["Fa"][i]
    
    def inverse_transform(self, new_output):
        for i in range(len(new_output["Fa"])):
            new_output["Fa"][i] = -new_output["Fa"][i]


class Transform:
    def __init__(self, transform_args):
        self.backup_keys = set()
        self.shifts = []
        self.scales = []
        for k, v in transform_args.items():
            if k == "atomic_energy":
                self.shifts.append(AtomicEnergyTransform(v))
                self.backup_keys.add("E")
            if k == "negative_gradient" and v:
                self.scales.append(NegativeGradientTransform())
                self.backup_keys.add("Fa")

    def transform(self, raw_input):
        for shift in self.shifts:
            shift.transform(raw_input)
        for scale in self.scales:
            scale.transform(raw_input)

    def inverse_transform(self, raw_output):
        for scale in self.scales:
            scale.inverse_transform(raw_output)
        for shift in self.shifts:
            shift.inverse_transform(raw_output)

