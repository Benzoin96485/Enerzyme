import numpy as np


PERIODIC_TABLE = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 31,
    "S": 32
}


def total_charge(data):
    if "chrg" in data:
        return (np.array(data["chrg"]).sum(axis=1) + 0.5).astype(int)
    else:
        return None


def atom_type_to_Z(data):
    return np.array([PERIODIC_TABLE[atom_type] for atom_type in data["atom_type"]], dtype=int)


FEATURE_REGISTER = {
    "Q": total_charge,
    "Ra": lambda data: np.array(data["coord"]),
    "Za": atom_type_to_Z,
    "N": count_atom
}