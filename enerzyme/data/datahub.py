import pickle
import os
import h5py
import numpy as np
from hashlib import md5
from typing import Union, List, Dict
from addict import Dict
from tqdm import tqdm
from .datatype import is_atomic, is_rounded
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


def _collect_types(types: Union[List, Dict]) -> Dict:
    if isinstance(types, list):
        return {single_type: single_type for single_type in types}
    else:
        return {k: v if v is not None else k for k, v in types.items()}


def array_padding(data, max_N, pad_value=0):
    for i in range(len(data)):
        pad_shape = [(0, max_N - len(data[i]))] + [(0,0)] * (len(data[i].shape) - 1)
        data[i] = np.pad(data[i], pad_shape, constant_values=pad_value)
    return np.array(data)


class DataHub:
    def __init__(self,  
        dump_dir=".",
        data_format=None,
        data_path=None, 
        preload=True,
        features=None,
        targets=None,
        transforms=None,
        neighbor_list=None,
        hash_length=16,
        **params
    ):
        self.data_path = os.path.abspath(data_path)
        self.dump_dir = dump_dir
        self.data_format = data_format
        self.preload = preload
        self.feature_types = _collect_types(features)
        self.target_types = _collect_types(targets)
        self.data_types = self.feature_types | self.target_types
        self.neighbor_list_type = neighbor_list
        self.transforms = transforms
        datahub_str = data_path + neighbor_list + str(sorted(transforms.items()))
        self.hash = md5(datahub_str.encode("utf-8")).hexdigest()[:hash_length]
        self.preload_path = os.path.join(self.dump_dir, f"processed_dataset_{self.hash}")
        self.transform = Transform(self.transforms, self.preload_path)
        if not self.preload or not self.preload_data():
            self.get_handle("w")
            self._init_data()
            self._init_neighbor_list()
            self.transform.transform(self.data)
            self._save_config()

    def _preload_data(self, hdf5_path):
        loaded_file = h5py.File(hdf5_path, mode="r")
        loaded_data = loaded_file["data"]
        self.data["N"] = loaded_data["N"]
        for k in self.data_types:
            if k == "N":
                continue
            elif is_atomic(k):
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
            preload_data_types = _collect_types(datahub_config.feature) | _collect_types(datahub_config.target)
            if preload_data_types.keys() <= self.data_types.keys():
                # all kinds of features and targets are contained in the processed dataset
                self.get_handle()
                logger.info(f"Data matched and preloaded from {self.preload_path}")
                return True
        return False

    def _load_molecular_data(self, k, raw_data):
        if self.data_types[k] in raw_data.keys():
            values = raw_data[self.data_types[k]]
            if isinstance(values, int) or isinstance(values, float):
                self.data.create_dataset(k, data=np.full(values, self.n_datapoint))
            else:
                n_feature = len(values)
                if n_feature == 1:
                    self.data.create_dataset(k, data=np.full(values[0], self.n_datapoint))
                elif n_feature == self.n_datapoint:
                    self.data.create_dataset(k, data=values)
                else:
                    raise IndexError(f"Length of '{k}' should be n_datapoint or 1")
        elif self.data_types[k + "a"] in raw_data.keys():
            self._load_atomic_data(k + "a", raw_data)
            if is_rounded(k):
                self.data.create_dataset(k, data=[
                    round(sum(self.data[k + "a"][i][:self.data["N"][i]])) for i in range(self.n_datapoint)
                ])
            else:
                self.data.create_dataset(k, data=[
                    sum(self.data[k + "a"][i][:self.data["N"][i]]) for i in range(self.n_datapoint)
                ])

    def _load_atomic_data(self, k: str, raw_data: Dict) -> None:
        if k in self.data:
            return
        values = raw_data[self.data_types[k]]
        if len(values) == self.n_datapoint:
            self.data.create_dataset(k, data=array_padding([values[i] for i in range(self.n_datapoint)], self.max_N))
        else:
            raise IndexError(f"Length of {k} ({self.data_types[k]}) should be n_datapoint")

    def _init_data(self) -> None:
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
                self.data.create_dataset("N", data=np.full(n_datapoint, len(Za)))
                self.data.create_dataset("Za", data=np.full(n_datapoint, Za[0]))
            else:
                self._load_molecular_data("N", raw_data)
                self.data.create_dataset("Za", data=[Zas[i] for i in range(n_datapoint)])
            self.max_N = max(self.data["N"])
        elif n_Za == n_datapoint:
            Zas = parse_Za(raw_data[self.data_types["Za"]])
            if self.data_types["N"] not in raw_data.keys():
                self.data.create_dataset("N", data=[len(Za) for Za in Zas])
            else:
                self._load_molecular_data("N", raw_data)
            self.max_N = max(self.data["N"])
            self.data.create_dataset("Za", data=array_padding(Zas, self.max_N))
        else:
            raise IndexError(f"Length of 'Za' should be n_datapoint or 1")
        
        for k in self.data_types:
            if k in ["Za", "N"]:
                continue
            elif is_atomic(k):
                self._load_atomic_data(k, raw_data)
            else:
                self._load_molecular_data(k, raw_data)
        
        if not (self.data_format == "pickle" or suffix == "pkl" or suffix == "pickle"):
            raw_data.close()

    def _init_neighbor_list(self):
        if self.neighbor_list_type == "full":
            from .neighbor_list import full_neighbor_list
            max_N = max(self.data["N"])
            max_N_pairs = max_N * (max_N - 1)
            self.data.create_dataset("idx_i", shape=(self.n_datapoint, max_N_pairs), dtype=int)
            self.data.create_dataset("idx_j", shape=(self.n_datapoint, max_N_pairs), dtype=int)
            self.data.create_dataset("N_pair", shape=self.n_datapoint, dtype=int)
            logger.info("producing neighbor list")
            for i in tqdm(range(self.n_datapoint)):
                idx_i, idx_j = full_neighbor_list(self.data["N"][i])
                self.data["N_pair"][i] = len(idx_i)
                self.data["idx_i"][i] = array_padding([idx_i], max_N_pairs, pad_value=-1)
                self.data["idx_j"][i] = array_padding([idx_j], max_N_pairs, pad_value=-1)
        else:
            raise NotImplementedError

    def get_handle(self, mode="r"):
        if os.path.exists(self.preload_path):
            logger.warning(f"Preload path {self.preload_path} exists and will be covered")
        else:
            os.makedirs(self.preload_path, exist_ok=True)
        self.file = h5py.File(os.path.join(self.preload_path, "pre_transformed.hdf5"), mode=mode, rdcc_nbytes=1024 ** 3 * 10)
        if mode == "r":
            self.data = self.file["data"]
        else:
            self.file.clear()
            self.data = self.file.create_group("data")

    def _save_config(self):
        handler = YamlHandler(os.path.join(self.preload_path, "datahub.yaml"))
        datahub_config = Dict({
            "feature": self.feature_types,
            "target": self.target_types,
            "transforms": self.transforms,
            "neighbor_list": self.neighbor_list_type
        })
        handler.write_yaml(datahub_config)
        logger.info(f"Save preloaded dataset at {self.preload_path}")

    @property
    def features(self):
        return {k: v for k, v in self.data.items() if k in self.feature_types.keys() | {"idx_i", "idx_j", "N_pair"}}
    
    @property
    def targets(self):
        return {k: v for k, v in self.data.items() if k in self.target_types}