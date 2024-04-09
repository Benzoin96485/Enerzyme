import pickle
import os
from collections import defaultdict
from .datascaler import EnergyScaler, ForceScaler, ChargeScaler
from .feature import FEATURE_REGISTER


def basic_key_from_task(task):
    keys = ["atom_type", "coord"]
    if 'e' in task:
        keys += ["energy", "grad"]
    if 'q' in task:
        keys += ["chrg"]
    return keys


def load_from_pickle(data_path=None, task=None):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    dd = {key: [datapoint[key] for datapoint in data] for key in basic_key_from_task(task)}
    return dd


class DataHub(object):
    def __init__(self, task=None, is_train=True, dump_dir=None, data_path=None, energy_bias_path=None, **params):
        self.data_path = data_path
        self.task = task
        self.is_train = is_train
        self.dump_dir = dump_dir
        self.energy_bias_path = energy_bias_path
        self._init_data()
        self._init_features(**params.Feature)

        
    def _init_data(self):
        self.data = defaultdict(dict)
        if self.data_path is not None:
            self.data = load_from_pickle(self.data_path, self.task)
        else:
            raise ValueError('No data path provided.')
        
        if "e" in self.task:
            self.data["energy_scaler"] = EnergyScaler(self.energy_bias_path,self.dump_dir)
            self.data["target"]["E"] = self.data["energy_scaler"].transform(self.data["energy"], self.data["atom_type"])
            self.data["target"]["F"] = ForceScaler.transform(self.data["grad"])

        if "q" in self.task:
            self.data["target"]["Qa"] = ChargeScaler.transform(self.data["chrg"])
        
        # ss_method = params.get('binding_energy', 'none')

    def _init_features(self, **feature_names):
        self.features = defaultdict(dict)
        for feature_name, feature_dict in feature_names.items():
            if feature_dict.get("active", False):
                if feature_name in FEATURE_REGISTER:
                    self.features[feature_name] = FEATURE_REGISTER[feature_name](self.data)
                else:
                    raise ValueError('Unknown feature name: {}'.format(feature_name))
