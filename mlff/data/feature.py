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
        return [int(sum(data["chrg"][i]) + 0.5) for i in len(data)]
    else:
        return [0] * len(data)


def atom_type_to_Z(data):
    return [PERIODIC_TABLE[atom_type] for atom_type in data["atom_type"]]


FEATURE_REGISTER = {
    "Q": total_charge,
    "Ra": lambda data: data["coord"],
    "Za": atom_type_to_Z
}