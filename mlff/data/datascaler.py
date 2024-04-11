import os
import pandas as pd
import numpy as np


def load_energy_bias(energy_bias_path):
    if os.path.exists(energy_bias_path):
        return pd.read_csv(energy_bias_path, index_col="atom_type")
    else:
        return None


class TargetScaler:
    def __init__(self, task, energy_bias_path, load_dir=None):
        if load_dir is not None:
            path = os.path.join(load_dir, 'energy_bias.csv')
        else:
            path = energy_bias_path
        self.bias = load_energy_bias(path)
        self.task = task
    
    def transform(self, data):
        targets = dict()
        if "e" in self.task:
            if self.bias is None:
                targets["E"] = np.array(data["energy"])
            else:
                e0 = np.array([self.bias.loc[frame]["atom_energy"].sum() for frame in data["atom_type"]])
                targets["E"] = np.array(data["energy"]) - e0
            targets["F"] = [-grad for grad in data["grad"]]
        if "q" in self.task:
            targets["Qa"] = data["chrg"]
        targets["atom_type"] = data["atom_type"]
        return targets

    def inverse_transform(self, pred):
        targets = dict()
        if "e" in self.task:
            if self.bias is None:
                targets["E"] = pred["E"]
            else:
                e0 = np.array([self.bias.loc[frame]["atom_energy"].sum() for frame in pred["atom_type"]])
                targets["E"] = pred["E"] + e0
                targets["F"] = pred["F"]
        if "q" in self.task:
            targets["Qa"] = pred["Qa"]
        return targets
