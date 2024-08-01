import pickle
from collections import defaultdict
from .datascaler import TargetScaler
from .feature import FEATURE_REGISTER


def basic_key_from_task(task):
    keys = ["atom_type", "coord"]
    if 'e' in task:
        keys += ["energy", "grad"]
    if 'q' in task:
        keys += ["chrg"]
    if 'p' in task:
        keys += ["dipole"]
    return keys


def load_from_pickle(data_path=None, task=None):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    keys = data[0].keys()
    dd = {key: [datapoint[key] for datapoint in data] for key in keys}
    return dd


class DataHub(object):
    def __init__(self, task=None, is_train=True, dump_dir=None, data_path=None, energy_bias_path=None, **params):
        self.data_path = data_path
        self.task = task
        self.is_train = is_train
        self.dump_dir = dump_dir
        self.energy_bias_path = energy_bias_path
        self._init_data()
        self._init_features(**params["Feature"])

    def _init_data(self):
        self.data = defaultdict(dict)
        if self.data_path is not None:
            self.data = load_from_pickle(self.data_path, self.task)
        else:
            raise ValueError('No data path provided.')
        
        self.data["target_scaler"] = TargetScaler(self.task, self.energy_bias_path, self.dump_dir)
        self.data["target"] = dict()
        for k, v in self.data["target_scaler"].transform(self.data).items():
            self.data["target"][k] = v

    def _init_features(self, **feature_names):
        self.features = defaultdict(dict)
        for feature_name, feature_dict in feature_names.items():
            if feature_dict.get("active", False):
                if feature_name in FEATURE_REGISTER:
                    self.features[feature_name] = FEATURE_REGISTER[feature_name](self.data)
                else:
                    raise ValueError('Unknown feature name: {}'.format(feature_name))
