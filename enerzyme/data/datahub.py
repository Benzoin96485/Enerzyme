import pickle, os
from hashlib import md5
from typing import Union, List, Dict, Optional, Iterable
import h5py
import numpy as np
from addict import Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from .datatype import is_atomic, is_rounded, is_int
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


class FieldDataset(Dataset):
    def __init__(self, data: Dict[str, Iterable]) -> None:
        self.data = data
        self.compressed_keys = set()
        for k, v in self.data.items():
            if len(v) == 1:
                self.compressed_keys.add(k)

    def __getitem__(self, k) -> Iterable:
        return self.data[k]

    def __setitem__(self, k, v) -> None:
        self.data[k] = v
        if len(v) == 1:
            self.compressed_keys.add(k)

    def __contains__(self, k) -> bool:
        return k in self.data
    
    def __len__(self) -> int:
        for v in self.data.values():
            if len(v) != 1:
                return len(v)
        else:
            return 1

    def items(self):
        return self.data.items()
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def loc(self, idx) -> Dict[str, Iterable]:
        return {k: v[0 if k in self.compressed_keys else idx] for k, v in self.data.items()}

    def load_subset(self, indices):
        data = dict()
        for k, v in self.data.items():
            if k in self.compressed_keys:
                data[k] = np.array(v)
            else:
                data[k] = np.array([v[idx] for idx in indices])
        return FieldDataset(data)


class DataHub:
    def __init__(self,  
        dump_dir=".",
        data_format: Optional[str]=None,
        data_path: str="", 
        preload: bool=True,
        features: Dict[str, str]=dict(),
        targets: Dict[str, str]=dict(),
        transforms: Optional[Dict[str, Union[str, bool]]]=None,
        neighbor_list: Optional[str]=None,
        hash_length: int=16,
        compressed: bool=True,
        max_memory: int=10,
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
        self.compressed = compressed
        self.max_memory = max_memory
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
                self._load_atomic_data(k, loaded_data)
            else:
                self._load_molecular_data(k, loaded_data)
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
    
    def _expand(self, k: str, values: Iterable) -> np.ndarray:
        if isinstance(values, int) or isinstance(values, float):
            if is_int(k) and self.compressed:
                return np.array([values])
            else:
                logger.info(f"Values of {k} (data type {self.data_types[k]}) are single and repeated")
                return np.full(values, self.n_datapoint)
        else:
            if is_int(k) and self.compressed:
                return values
            else:
                logger.info(f"Values of {k} (data type {self.data_types[k]}) are single and repeated")
                return np.repeat(values, self.n_datapoint, axis=0)
    
    def _compress(self, k: str, values: Iterable) -> np.ndarray:
        # only works for equal length data
        value_array = np.array(values)
        if is_int(k) and self.compressed and (value_array == value_array[0]).all():
            logger.info(f"Values of {k} (data type {self.data_types[k]}) are all the same and compressed into a single value")
            return value_array[:1]
        else:
            return value_array

    def _load_molecular_data(self, k: str, raw_data: Dict) -> None:
        if self.data_types[k] in raw_data.keys():
            values = raw_data[self.data_types[k]]
            if isinstance(values, int) or isinstance(values, float) or len(values) == 1:
                self.data.create_dataset(k, data=self._expand(k, values))
            elif len(values) == self.n_datapoint:
                self.data.create_dataset(k, data=self._compress(k, values))
            else:
                raise IndexError(f"Length of '{k}' should be n_datapoint or 1")
        elif self.data_types[k + "a"] in raw_data.keys():
            self._load_atomic_data(k + "a", raw_data)
            # reduce atomic property into molecular property, mainly for Qa into Q
            logger.info(f"Molecular property {k} are reduced from atomic property {k + 'a'} ({self.data_types[k + 'a']})")
            if is_rounded(k):
                values = [round(sum(self.data[k + "a"][i][:self.data["N"][i % len(self.data["N"])]])) for i in tqdm(range(self.n_datapoint))]
            else:
                values = [sum(self.data[k + "a"][i][:self.data["N"][i % len(self.data["N"])]]) for i in tqdm(range(self.n_datapoint))]
            # don't compress summation of atomic property
            self.data.create_dataset(k, data=np.array(values))

    def _load_atomic_data(self, k: str, raw_data: Dict) -> None:
        if k in self.data:
            return
        values = raw_data[self.data_types[k]]
        v0 = np.array(values[0])
        if len(values) == self.n_datapoint:
            # for a datapoint, the shape of this property is (N, a, b, ...)
            # for the whole dataset, the shape of this property is (n_datapoint, max_N, a, b, ...)
            self.data.create_dataset(k, shape=(self.n_datapoint, self.max_N, *v0.shape[1:]), dtype=v0.dtype)
            logger.info(f"Storing atomic data {k} ({self.data_types[k]})")
            for i, v in tqdm(enumerate(values), total=self.n_datapoint):
                self.data[k][i,:len(v)] = v

        elif len(values) == 1:
            self.data.create_dataset(k, data=self._expand(k, values))
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
            # atomic position must be provided
            raise KeyError(f"Dataset must contain 'Ra' key (Atomic positions)")
        # number of datapoints is defined as number of different configurations
        n_datapoint = len(raw_data[self.data_types["Ra"]])
        self.n_datapoint = n_datapoint

        if "Za" not in self.data_types:
            # atomic number/type must be provided
            raise KeyError(f"Dataset must contain 'Za' key (Atomic numbers)")
        
        n_Za = len(raw_data[self.data_types["Za"]])
        Zas = parse_Za(raw_data[self.data_types["Za"]])
        if n_Za == 1:
            if self.data_types["N"] not in raw_data.keys():
                # atom count determined by length of atomic numbers
                self.data.create_dataset("N", data=self._expand("N", len(Zas)))
            else:
                self._load_molecular_data("N", raw_data)
            self.data.create_dataset("Za", data=self._expand("Za", Zas))
            self.max_N = max(self.data["N"])
        elif n_Za == n_datapoint:
            if self.data_types["N"] not in raw_data.keys():
                self.data.create_dataset("N", data=self._compress("N", [len(Za) for Za in Zas]))
            else:
                self._load_molecular_data("N", raw_data)
            self.max_N = max(self.data["N"])
            Za_compressed_flag = True
            Za0 = np.array(Zas[0])
            N0 = len(Za0)
            for Za in Zas:
                if len(Za) != N0 or (Za != Za0).any():
                    Za_compressed_flag = False
                    break
            if self.compressed and Za_compressed_flag:
                self.data.create_dataset("Za", data=[Za0])
            else:
                self.data.create_dataset("Za", shape=(n_datapoint, self.max_N), dtype=int)
                logger.info(f'Storing Za ({self.data_types["Za"]})')
                for i, Za in tqdm(enumerate(Zas), total=self.n_datapoint):
                    self.data["Za"][i,:len(Za)] = Za
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

    def _init_neighbor_list(self) -> None:
        if self.neighbor_list_type == "full":
            from .neighbor_list import full_neighbor_list
            logger.info("producing neighbor list")
            if self.compressed and len(self.data["N"]) == 1:
                idx_i, idx_j = full_neighbor_list(self.data["N"][0])
                self.data.create_dataset("idx_i", data=[idx_i])
                self.data.create_dataset("idx_j", data=[idx_j])
                self.data.create_dataset("N_pair", data=[len(idx_i)])
            else:
                max_N_pairs = self.max_N * (self.max_N - 1)
                self.data.create_dataset("idx_i", shape=(self.n_datapoint, max_N_pairs), dtype=int)
                self.data.create_dataset("idx_j", shape=(self.n_datapoint, max_N_pairs), dtype=int)
                self.data.create_dataset("N_pair", shape=self.n_datapoint, dtype=int)
                for i in tqdm(range(self.n_datapoint)):
                    idx_i, idx_j = full_neighbor_list(self.data["N"][i])
                    self.data["N_pair"][i] = len(idx_i)
                    self.data["idx_i"][i] = array_padding([idx_i], max_N_pairs, pad_value=-1)
                    self.data["idx_j"][i] = array_padding([idx_j], max_N_pairs, pad_value=-1)

    def get_handle(self, mode="r"):
        if os.path.exists(self.preload_path):
            logger.warning(f"Preload path {self.preload_path} exists and will be covered")
        else:
            os.makedirs(self.preload_path, exist_ok=True)
        self.file = h5py.File(os.path.join(self.preload_path, "pre_transformed.hdf5"), mode=mode, rdcc_nbytes=1024 ** 3 * self.max_memory)
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
    def features(self) -> FieldDataset:
        return FieldDataset({k: v for k, v in self.data.items() if k in self.feature_types.keys() | {"idx_i", "idx_j", "N_pair"}})
    
    @property
    def targets(self) -> FieldDataset:
        return FieldDataset({k: v for k, v in self.data.items() if k in self.target_types})
