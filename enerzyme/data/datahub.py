import pickle, os
from bisect import bisect
from glob import glob
from hashlib import md5
from typing import Union, List, Optional, Iterable, Literal, Any, Callable
import h5py
import numpy as np
from ase import Atoms
from ase.db import connect
from addict import Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from .datatype import is_atomic, is_rounded, is_int, register_data_type
from .transform import parse_Za, Transform
from ..utils import YamlHandler, logger


ASE_PROPERTY_METHODS: Dict[str, Callable[[Atoms], Any]] = {
    "E": lambda atoms: atoms.get_potential_energy(),
    "Fa": lambda atoms: atoms.get_forces(),
    "Qa": lambda atoms: atoms.get_charges(),
    "Sa": lambda atoms: atoms.get_magnetic_moments(),
    "Q": lambda atoms: atoms.info.get("charge", 0),
    "S": lambda atoms: atoms.info.get("spin", 1) - 1,
}


def load_from_pickle(data_path=str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        keys = set()
        for datapoint in data:
            keys.update(datapoint.keys())
        logger.info(f"Collected keys from data: {keys}")
        dd = {key: [datapoint.get(key, None) for datapoint in data] for key in keys}
        return dd
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError(f"Unknown data type in {data_path}!")


def _get_single_aselmdb_data_path(data_path=str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    elif os.path.isdir(data_path):
        return glob(os.path.join(data_path, "*"))
    else:
        return glob(data_path)


class ASELMDBSingleProperty:
    def __init__(self, aselmdb_dataset: "ASELMDBDataset", get_property_method: Callable[[Atoms], Any]):
        self.aselmdb_dataset = aselmdb_dataset
        self.get_property_method = get_property_method
    
    def __len__(self) -> int:
        return len(self.aselmdb_dataset)

    def __getitem__(self, idx: int) -> Union[int, float, np.ndarray]:
        db_idx = bisect(self.aselmdb_dataset._id_len_segments, idx)

        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self.aselmdb_dataset._id_len_segments[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.aselmdb_dataset.dbs[db_idx]._get_row(self.aselmdb_dataset.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()

        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        if "sid" not in atoms.info:
            atoms.info["sid"] = idx

        return self.get_property_method(atoms)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class ASELMDBDataset:
    def __init__(self, data_path, connect_args: Dict[str, Any]=dict(), select_args: Dict[str, Any]=dict()):
        self.data_paths = []
        self.dbs = []
        self.db_ids = []
        default_connect_args = {
            "readonly": True,
            "use_lock_file": False
        }
        default_connect_args.update(connect_args)
        if isinstance(data_path, list):
            for single_data_path in data_path:
                self.data_paths.extend(_get_single_aselmdb_data_path(single_data_path))
        else:
            self.data_paths.extend(_get_single_aselmdb_data_path(data_path))
        if not self.data_paths:
            raise ValueError(f"No data paths found in {data_path}")
        
        for single_data_path in self.data_paths:
            try:
                self.dbs.append(connect(single_data_path, **default_connect_args))
            except Exception as e:
                logger.warning(f"Failed to connect to {single_data_path}: {e}")
                continue
        if not self.dbs:
            raise ValueError(f"Failed to connect to any of the data paths in {data_path}")
        
        for db in self.dbs:
            if hasattr(db, "ids") and not select_args:
                self.db_ids.append(db.ids)
            else:
                self.db_ids.append([row.id for row in db.select(**select_args)])
        self.id_lens = [len(ids) for ids in self.db_ids]
        self._id_len_segments = np.cumsum(self.id_lens).tolist()

        first_db = self.dbs[0]
        try:
            first_row = first_db._get_row(self.db_ids[0][0])
            first_atoms = first_row.toatoms()
        except Exception as e:
            raise ValueError(f"Failed to get atoms from {first_db}: {e}")
        
        properties_from_calculator = set()
        if first_atoms.calc is not None:
            for prop, method in ASE_PROPERTY_METHODS.items():
                try:
                    method(first_atoms)
                except Exception as e:
                    logger.warning(f"Failed to get {prop} directly from calculator: {e}")
                else:
                    properties_from_calculator.add(prop)
        
        self.properties_from_info = set()
        self.quantum_numbers = set()
        print(first_row.data.keys())
        for k, v in first_row.data.items():
            if k == "charge":
                self.quantum_numbers.add("Q")
            elif k == "spin":
                self.quantum_numbers.add("S")
            elif isinstance(v, float) or isinstance(v, int) or isinstance(v, np.ndarray):
                self.properties_from_info.add(k)

        self.unique_properties_from_calculator = properties_from_calculator - self.properties_from_info
        overlapped_properties = properties_from_calculator & self.properties_from_info
        self._all_properties = self.unique_properties_from_calculator | self.properties_from_info | {"Ra", "Za", "N"} | self.quantum_numbers
        if overlapped_properties:
            logger.warning(f"Property {overlapped_properties} found in calculator will be overwritten by info")
        logger.info(f"Properties from calculator: {self.unique_properties_from_calculator}")
        logger.info(f"Properties from info: {self.properties_from_info}")

    def __len__(self) -> int:
        return sum(self.id_lens)

    def __getitem__(self, k) -> np.ndarray:
        if k == "Ra":
            get_property_method = lambda atoms: atoms.get_positions()
        elif k == "Za":
            get_property_method = lambda atoms: atoms.get_atomic_numbers()
        elif k == "N":
            get_property_method = lambda atoms: len(atoms)
        elif k in self.quantum_numbers or k in self.unique_properties_from_calculator:
            get_property_method = ASE_PROPERTY_METHODS[k]
        else:
            get_property_method = lambda atoms: atoms.info.get(k, None)
        return ASELMDBSingleProperty(self, get_property_method=get_property_method)
    
    def __contains__(self, k) -> bool:
        return k in self._all_properties

    def items(self):
        return {k: self[k] for k in self._all_properties}

    def keys(self):
        return self._all_properties

    def values(self):
        return [self[k] for k in self._all_properties]


def _collect_types(types: Optional[Union[List, Dict]]) -> Dict:
    if types is None:
        return dict()
    elif isinstance(types, list):
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

    def load_subset(self, indices: Iterable[int]) -> "FieldDataset":
        data = dict()
        for k, v in self.data.items():
            if k in self.compressed_keys:
                data[k] = np.array(v)
            else:
                data[k] = np.array([v[idx] for idx in indices])
        return FieldDataset(data)


class SingleDataHub:
    def __init__(self,  
        dump_dir=".",
        data_format: Optional[str]=None,
        data_path: str="", 
        preload: bool=True,
        features: Dict[str, str]=dict(),
        targets: Dict[str, str]=dict(),
        preprocessings: Optional[Dict[str, Union[str, bool]]]=None,
        global_transforms: Optional[Dict[str, Union[str, bool]]]=None,
        neighbor_list: Optional[str]=None,
        hash_length: int=16,
        compressed: bool=True,
        max_memory: int=10,
        connect_args: Dict[str, Any]=dict(),
        select_args: Dict[str, Any]=dict(),
        **params
    ):
        self.data_path = os.path.abspath(data_path)
        self.data_format = data_format
        self.preload = preload
        self.feature_types = _collect_types(features)
        self.target_types = _collect_types(targets)
        self.data_types = self.feature_types | self.target_types
        self.neighbor_list_type = neighbor_list
        self.compressed = compressed
        self.max_memory = max_memory
        datahub_str = data_path + str(neighbor_list) + \
            str(sorted(preprocessings.items()) if preprocessings is not None else '') + \
            str(sorted(global_transforms.items()) if global_transforms is not None else '')
        self.hash = md5(datahub_str.encode("utf-8")).hexdigest()[:hash_length]
        self.preload_path = os.path.join(dump_dir, f"processed_dataset_{self.hash}")
        logger.info(f"Preload path {self.preload_path} is created")
        self.preprocessing = Transform(preprocessings, self.preload_path)
        self.global_transform = Transform(global_transforms, self.preload_path)
        self.preprocessings = preprocessings
        self.global_transforms = global_transforms
        self.connect_args = connect_args
        self.select_args = select_args
        if not self.preload or not self.preload_data():
            self.get_handle("w")
            self._init_data()
            self._init_neighbor_list()
            self.preprocessing.transform(self.data)
            self.global_transform.transform(self.data)
            self._save_config()
            self.reset_handle()

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
        if not (isinstance(values, list) or isinstance(values, np.ndarray)):
            values = list(tqdm(values, total=len(values), desc=f"Enumerating {k} (data type {self.data_types[k]})"))
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
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path {self.data_path} doesn't exist.")
        suffix = self.data_path.split(".")[-1]
        if self.data_format == "hdf5" or suffix == "hdf5":
            self.data_format = "hdf5"
            raw_data = h5py.File(self.data_path, mode="r")["data"]
        elif self.data_format == "pickle" or suffix == "pkl" or suffix == "pickle":
            self.data_format = "pickle"
            raw_data = load_from_pickle(self.data_path)
        elif self.data_format == "npz" or suffix == "npz":
            self.data_format = "npz"
            raw_data = np.load(self.data_path, allow_pickle=True)
        elif self.data_format == "sdf" or suffix == "sdf":
            self.data_format = "sdf"
            from .supplier import SDFSupplier
            supplier = SDFSupplier(self.data_path, supplying_fields=self.data_types.keys())
            raw_data = supplier.raw_data()
        elif self.data_format == "aselmdb" or suffix == "aselmdb":
            raw_data = ASELMDBDataset(self.data_path, connect_args=self.connect_args, select_args=self.select_args)
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
        if self.data_format == "pickle":
            Zas = parse_Za(raw_data[self.data_types["Za"]])
        else:
            Zas = raw_data[self.data_types["Za"]]
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
        
        if self.data_format in ["hdf5", "npz"]:
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

    def get_handle(self, mode: Literal["r", "w"]="r") -> None:
        if mode == "w" and os.path.exists(self.preload_path):
            logger.warning(f"Preload path {self.preload_path} exists and will be overwritten")
        else:
            os.makedirs(self.preload_path, exist_ok=True)
        self.file = h5py.File(os.path.join(self.preload_path, "pre_transformed.hdf5"), mode=mode, rdcc_nbytes=1024 ** 3 * self.max_memory)
        if mode == "r":
            self.data = self.file["data"]
        else:
            self.file.clear()
            self.data = self.file.create_group("data")

    def reset_handle(self):
        self.file.close()
        self.get_handle()

    def _save_config(self):
        handler = YamlHandler(os.path.join(self.preload_path, "datahub.yaml"))
        datahub_config = Dict({
            "feature": self.feature_types,
            "target": self.target_types,
            "preprocessings": self.preprocessings,
            "global_transforms": self.global_transforms,
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


class DataHub:
    def __init__(self,  
        dump_dir=".",
        datasets: Optional[Union[List, Dict]]=None,
        fields: Optional[Dict[str, str]]=None,
        **params
    ):
        self.dump_dir = dump_dir
        if fields is not None:
            for k, v in fields.items():
                register_data_type(k, **v)
        if datasets is None:
            if "global_transforms" not in params:
                params["global_transforms"] = params.get("transforms", None)
            self.datahubs = {"default": SingleDataHub(dump_dir=dump_dir, **params)}
        elif isinstance(datasets, list):
            self.datahubs = {str(i): SingleDataHub(dump_dir=dump_dir, global_transforms=params.get("global_transforms", None), **dataset_params) for i, dataset_params in enumerate(datasets)}
        elif isinstance(datasets, dict):
            self.datahubs = {name: SingleDataHub(dump_dir=dump_dir, global_transforms=params.get("global_transforms", None), **dataset_params) for name, dataset_params in datasets.items()}
        else:
            raise ValueError(f"Unknown type of datasets: {type(datasets)}")
            
    @property
    def features(self) -> Dict[str, FieldDataset]:
        return {name: datahub.features for name, datahub in self.datahubs.items()}

    @property
    def targets(self) -> Dict[str, FieldDataset]:
        return {name: datahub.targets for name, datahub in self.datahubs.items()}
    
    @property
    def preload_path(self) -> Dict[str, str]:
        return {name: datahub.preload_path for name, datahub in self.datahubs.items()}
    
    @property
    def transform(self) -> Transform:
        return list(self.datahubs.values())[0].global_transform
