import os
import pathlib
import numpy as np
import pandas as pd


PERIODIC_TABLE_PATH = os.path.join(
    pathlib.Path(__file__).parent.resolve(),
    'periodic-table.csv'
)
PERIODIC_TABLE = pd.read_csv(PERIODIC_TABLE_PATH, index_col="atom_type")

def total_charge(data):
    if "chrg" in data:
        return [(sum(chrgs) + 0.5).astype(int) for chrgs in data["chrg"]]
    else:
        return None


def atom_type_to_Za(data):
    return [PERIODIC_TABLE.loc[atom_types]["Za"].to_numpy() for atom_types in data["atom_type"]]


FEATURE_REGISTER = {
    "Q": total_charge,
    "Ra": lambda data: data["coord"],
    "Za": atom_type_to_Za
}