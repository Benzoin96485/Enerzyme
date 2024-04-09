import os
import pandas as pd
import numpy as np


def load_energy_bias(energy_bias_path):
    if os.path.exists(energy_bias_path):
        return pd.read_csv(energy_bias_path, index_col="atom_type")
    else:
        return None


class EnergyScaler(object):
    def __init__(self, energy_bias_path, load_dir=None):
        default_path = os.path.join(load_dir, 'energy_bias.csv')
        if load_dir is not None:
            self.bias = load_energy_bias(default_path)
        else:
            self.bias = load_energy_bias(energy_bias_path)
    
    def transform(self, energy, atom_type):
        if self.bias is None:
            return np.array(energy)
        else:
            e0 = np.array([self.bias.loc[frame]["atom_energy"].sum() for frame in atom_type])
            return np.array(energy) - e0

    def inverse_transform(self, energy, atom_type):
        if self.bias is None:
            return np.array(energy)
        else:
            e0 = np.array([self.bias.loc[frame]["atom_energy"].sum() for frame in atom_type])
            return np.array(energy) + e0


class ForceScaler(object):
    def __init__(self):
        pass
    
    @classmethod
    def transform(self, grad):
        return -np.array(grad)
    

class ChargeScaler(object):
    def __init__(self):
        pass
    
    @classmethod
    def transform(self, chrg):
        return np.array(chrg)