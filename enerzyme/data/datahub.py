import pickle
import os
import h5py
import numpy as np
from hashlib import md5
from addict import Dict
from .transform import parse_Za, Transform
from ..utils import YamlHandler, logger


def load_from_pickle(data_path=str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        keys = data[0].keys()
        dd = {key: [datapoint[key] for datapoint in data] for key in keys}
        return dd
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError(f"Unknown data type in {data_path}!")


def _collect_types(types: dict):
    if isinstance(types, list):
        return {single_type: single_type for single_type in types}
    else:
        return types


def array_padding(data, max_N):
    for i in range(len(data)):
        pad_shape = [(0, max_N - len(data[i]))] + [(0,0)] * (len(data[i].shape) - 1)
        data[i] = np.pad(data[i], pad_shape)
    return np.array(data)


class DataHub:
    def __init__(self,  
        dump_dir=".",
        data_format=None,
        data_path=None, 
        preload=True,
        save=True,
        features=None,
        targets=None,
        transforms=None,
        neighbor_list=None,
        hash_length=16,
        **params
    ):
        self.data_path = data_path
        self.dump_dir = dump_dir
        self.data_format = data_format
        self.preload = preload
        self.save = save
        self.feature_types = _collect_types(features)
        self.target_types = _collect_types(targets)
        self.data_types = self.feature_types | self.target_types
        self.data = dict()
        self.neighbor_list_type = neighbor_list
        self.transforms = transforms
        datahub_str = neighbor_list + str(sorted(transforms.items()))
        self.hash = md5(datahub_str.encode("utf-8")).hexdigest()[:hash_length]
        self.preload_path = os.path.join(self.dump_dir, f"processed_dataset_{self.hash}")
        if not self.preload or not self.preload_data():
            self._init_data()
            self._init_neighbor_list()
            self._init_transform()
            if self.save:
                self._save()

    def _preload_data(self, hdf5_path):
        loaded_file = h5py.File(hdf5_path, mode="r")
        loaded_data = loaded_file["data"]
        self.data["N"] = list(loaded_data["N"])
        for k in self.data_types:
            if k == "N":
                continue
            elif k[-1] == "a":
                self._load_atomic_data(k, loaded_data, preload=True)
            else:
                self._load_molecular_data(k, loaded_data, preload=True)
        loaded_file.close()

    def preload_data(self): 
        hdf5_path = os.path.join(self.preload_path, "pre_transformed.hdf5")
        config_path = os.path.join(self.preload_path, "datahub.yaml")
        if (
            os.path.isdir(self.preload_path) and
            os.path.isfile(hdf5_path) and
            os.path.isfile(config_path)
        ):
            handler = YamlHandler(config_path)
            datahub_config = handler.read_yaml()
            preload_data_types = set((_collect_types(datahub_config.feature) | _collect_types(datahub_config.target)).keys())
            if preload_data_types <= set(self.data_types.keys()):
                # all kinds of features and targets are contained in the processed dataset
                self._preload_data(hdf5_path)
                logger.info(f"Data matched and preloaded from {self.preload_path}")
                return True
        return False

    def _load_molecular_data(self, k, raw_data, preload=False):
        if preload:
            self.data[k] = [v for v in raw_data[k]]
        else:
            if self.data_types[k] in raw_data.keys():
                values = raw_data[self.data_types[k]]
                if isinstance(values, int):
                    self.data[k] = [values] * self.n_datapoint
                else:
                    n_feature = len(values)
                    if n_feature == 1:
                        self.data[k] = [values[0]] * self.n_datapoint
                    elif n_feature == self.n_datapoint:
                        self.data[k] = values
                    else:
                        raise IndexError(f"Length of '{k}' should be n_datapoint or 1")
            elif self.data_types[k + "a"] in raw_data.keys():
                self._load_atomic_data(k + "a", raw_data)
                if k in ["Q", "S"]:
                    self.data[k] = [round(sum(v)) for v in self.data[k + "a"]]
                else:
                    self.data[k] = [sum(v) for v in self.data[k + "a"]]

    def _load_atomic_data(self, k, raw_data, preload=False):
        if k in self.data:
            return
        values = raw_data[k if preload else self.data_types[k]]
        if preload or len(values) == self.n_datapoint:
            self.data[k] = [values[i][:self.data["N"][i]] for i in range(len(values))]
        else:
            raise IndexError(f"Length of {k} ({self.data_types[k]}) should be n_datapoint")

    def _init_data(self):
        if not os.path.isfile(self.data_path):
            raise ValueError(f"Data path {self.data_path} doesn't exist.")
        suffix = self.data_path.split(".")[-1]
        if self.data_format == "hdf5" or suffix == "hdf5":
            raw_data = h5py.File(self.data_path, mode="r")["data"]
        elif self.data_format == "pickle" or suffix == "pkl" or suffix == "pickle":
            raw_data = load_from_pickle(self.data_path)
        elif self.data_format == "npz" or suffix == "npz":
            raw_data = np.load(self.data_path, allow_pickle=True)
        else:
            raise ValueError(f"Data format of {self.data_path} is unknown")

        if "Ra" not in self.data_types:
            raise KeyError(f"Dataset must contain 'Ra' key (Atomic positions)")
        n_datapoint = len(raw_data[self.data_types["Ra"]])
        self.n_datapoint = n_datapoint

        if "Za" not in self.data_types:
            raise KeyError(f"Dataset must contain 'Za' key (Atomic numbers)")
        n_Za = len(raw_data[self.data_types["Za"]])
        if n_Za == 1:
            Za = parse_Za(raw_data[self.data_types["Za"]])
            if self.data_types["N"] not in raw_data.keys():
                self.data["N"] = [len(Za)] * n_datapoint
                self.data["Za"] = [Za[0]] * n_datapoint
            else:
                self._load_molecular_data("N", raw_data)
                self.data["Za"] = [Zas[i][:self.data["N"][i]] for i in range(n_datapoint)]
        elif n_Za == n_datapoint:
            Zas = parse_Za(raw_data[self.data_types["Za"]])
            if self.data_types["N"] not in raw_data.keys():
                self.data["N"] = [len(Za) for Za in Zas]
                self.data["Za"] = Zas
            else:
                self._load_molecular_data("N", raw_data)
                self.data["Za"] = [Zas[i][:self.data["N"][i]] for i in range(n_datapoint)]
        else:
            raise IndexError(f"Length of 'Za' should be n_datapoint or 1")
        
        for k in self.data_types:
            if k in ["Za", "N"]:
                continue
            elif k[-1] == "a":
                self._load_atomic_data(k, raw_data)
            else:
                self._load_molecular_data(k, raw_data)
        
        if self.data_format == "hdf5" or suffix == "hdf5":
            raw_data.close()

    def _init_neighbor_list(self):
        if self.neighbor_list_type == "full":
            from .neighbor_list import full_neighbor_list
            idx_i, idx_j = full_neighbor_list(self.data["N"])
        else:
            raise NotImplementedError
        self.data["idx_i"] = idx_i
        self.data["idx_j"] = idx_j

    def _init_transform(self):
        self.transform = Transform(self.data, self.transforms)
        self.data = self.transform.transform(self.data)

    def _save(self):
        if os.path.exists(self.preload_path):
            logger.warning(f"Preload path {self.preload_path} exists and will be covered")
        else:
            os.makedirs(self.preload_path, exist_ok=True)
        loaded_file = h5py.File(os.path.join(self.preload_path, "pre_transformed.hdf5"), mode="w")
        loaded_data = loaded_file.create_group("data")
        max_N = max(self.data["N"])
        for k, v in self.data.items():
            if k[-1] == "a":
                loaded_data.create_dataset(k, data=array_padding(v, max_N))
            else:
                loaded_data.create_dataset(k, data=np.array(v))
        loaded_file.close()
        handler = YamlHandler(os.path.join(self.preload_path, "datahub.yaml"))
        datahub_config = Dict({
            "feature": self.feature_types,
            "target": self.target_types,
            "transform_args": self.transforms,
            "neighbor_list": self.neighbor_list_type
        })
        handler.write_yaml(datahub_config)
        logger.info(f"Save preloaded dataset at {self.preload_path}")

    @property
    def features(self):
        return {k: v for k, v in self.data.items() if k in self.feature_types}
    
    @property
    def targets(self):
        return {k: v for k, v in self.data.items() if k in self.target_types}