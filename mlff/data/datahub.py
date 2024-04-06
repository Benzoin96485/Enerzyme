from collections import defaultdict
from .datascaler import EnergyScaler
import pickle


def load_from_pickle(self, data_path=None, task=None, **params):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if task == "q":
        dd = {key: [datapoint[key] for datapoint in data] for key in ("atom_type", "coord", "chrg")}
    elif task == "e":
        dd = {key: [datapoint[key] for datapoint in data] for key in ("atom_type", "coord", "energy", "grad")}
    elif task == "qe":
        dd = {key: [datapoint[key] for datapoint in data] for key in ("atom_type", "coord", "chrg", "energy", "grad")}
    return dd


class DataHub(object):
    def __init__(self, data_path=None, task=None, is_train=True, dump_dir=None, **params):
        self.data_path = data_path
        self.task = task
        self.is_train = is_train
        self.dump_dir = dump_dir

        self._init_data(**params)
        self._init_features(**params)

    def _init_data(self, **params):
        self.data = defaultdict(dict)
        if self.data_path is not None:
            self.data = load_from_pickle(self.data_path, self.task, **params)
        else:
            raise ValueError('No data path provided.')
        
        # ss_method = params.get('binding_energy', 'none')

    def _init_features(self, **params):
        self.features = defaultdict(dict)
        pass
