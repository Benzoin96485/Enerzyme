import os
from collections import defaultdict, OrderedDict
from typing import Union, List, Dict, Optional, Literal
from hashlib import md5
import addict
import sklearn as skl
import numpy as np
from ..utils import logger, YamlHandler
from ..data import FieldDataset

SIMPLE_PART_INFO_TYPE = Dict[Literal["name", "dataset", "ratio"], Union[str, float, int]]
FULL_PART_INFO_TYPE = Dict[Literal["name", "sources"], Union[str, List[Dict[Literal["dataset", "ratio"], Union[str, float, int]]]]]

class RandomSplit:
    def __init__(self, 
        parts: Union[
            List[str],
            List[Union[SIMPLE_PART_INFO_TYPE, FULL_PART_INFO_TYPE]]
        ],
        ratios: Optional[List[float]]=None, 
        **params
    ):
        self.parts = parts
        self.ratios = ratios.copy()
        self.mode = "old"

        # sanity check
        if all(isinstance(part, str) for part in parts):
            if ratios is None:
                raise ValueError("Ratios should be provided if parts is a list")
        elif all(isinstance(part, dict) for part in parts):
            self.mode = "full"
        else:
            raise ValueError("Parts should be a list of part names or a list of part info dictionaries")
        
        splitstr = "random" + str(parts) + str(ratios)
        self.hash = md5(splitstr.encode("utf-8")).hexdigest()[:16]

    def _get_empty_split(self, data: Dict[str, FieldDataset]) -> OrderedDict[str, OrderedDict[str, List[int]]]:
        if self.mode == "old":
            return OrderedDict(
                (part_key, OrderedDict([(data.keys()[0], [])])) for part_key in self.parts
            )
        else:
            return OrderedDict(
                (
                    part_key, (
                        OrderedDict([(part_info["source"], [])]) if "source" in part_info
                        else OrderedDict((data_key, []) for data_key in part_info["sources"])
                    )
                ) for part_key, part_info in self.parts.items()
            )


    def split(self, data: Dict[str, FieldDataset]) -> None:
        canonical_partition = defaultdict(dict)
        part_keys = []
        sorted_data_keys = defaultdict(list)
        if self.mode == "old":
            if len(data) > 1:
                raise ValueError("Source should be provided for multiple datasets")
            for i, part_key in enumerate(self.parts):
                if i >= len(self.ratios):
                    canonical_partition[part_key][data.keys()[0]] = -1
                else:
                    canonical_partition[part_key][data.keys()[0]] = self.ratios[i]
                sorted_data_keys[part_key].append(data.keys()[0])
            part_keys = self.parts
        else:
            for part_info in self.parts:
                if "name" not in part_info:
                    raise KeyError("Name should be provided for each part")
                if "dataset" in part_info:
                    canonical_partition[part_info["name"]][part_info["dataset"]] = part_info.get("ratio", -1)
                    sorted_data_keys[part_info["name"]].append(part_info["dataset"])
                elif "sources" in part_info:
                    for source_info in part_info["sources"]:
                        if "dataset" not in source_info:
                            raise KeyError("Dataset should be provided for each source")
                        canonical_partition[part_info["name"]][source_info["dataset"]] = source_info.get("ratio", -1)
                        sorted_data_keys[part_info["name"]].append(source_info["dataset"])
                else:
                    raise KeyError("One dataset or multiplesources should be provided for each part")
                part_keys.append(part_info["name"])

        l = {k: len(data[k]["Ra"]) for k in data.keys()}
        rest = {k: None for k in data.keys()}
        final_partition = defaultdict(dict)
        allocated_count = {k: 0 for k in data.keys()}

        for part_key, part_info in canonical_partition.items():
            for data_key, ratio in part_info.items():
                if isinstance(ratio, str) and ratio[-1] == "%":
                    ratio = float(ratio[:-1]) / 100
                    if ratio > 1:
                        raise ValueError("Percentage shouldn't be greater than 100%")
                elif ratio is None or ratio < 0:
                    if rest[data_key] is None:
                        rest[data_key] = part_key
                    else:
                        raise ValueError(f"Only one part can be used as the remaining part of the dataset {data_key}")
                elif ratio < 1:
                    count = min(round(ratio * l[data_key]), l[data_key] - allocated_count[data_key])
                    final_partition[part_key][data_key] = count
                    allocated_count[data_key] += count
                elif isinstance(ratio, int):
                    final_partition[part_key][data_key] = min(ratio, l[data_key] - allocated_count[data_key])
                    allocated_count[data_key] += ratio
                else:
                    raise ValueError(f"Ratio or count {ratio} not valid")

        for data_key, part_key in rest.items():
            if part_key is not None:
                final_partition[part_key][data_key] = max(0, l[data_key] - allocated_count[data_key])

        allocation = {k: [0] for k in data.keys()}
        for part_key in part_keys:
            for data_key in sorted_data_keys[part_key]:
                data_count = final_partition[part_key][data_key]
                allocation[data_key].append(data_count + allocation[data_key][-1])

        full_indices = {k: list(range(l[k])) for k in data.keys()}
        for indices in full_indices.values():
            np.random.shuffle(indices)
        self.split = OrderedDict(
            (
                part_key, 
                OrderedDict([
                    (data_key, full_indices[data_key][allocation[data_key][i]:allocation[data_key][i+1]]) 
                    for data_key in sorted_data_keys[part_key]
                ])
            ) for i, part_key in enumerate(part_keys)
        )


class Splitter:
    def __init__(self, 
        method: Literal["random"]='random', 
        seed: int=114514, 
        preload: bool=True, 
        save: bool=True, 
        **params
    ):
        self.method = method
        self.seed = seed
        self.params = params
        self.splitter = self._init_split(method, **params)
        self.preload = preload
        self.save = save
        self.split = defaultdict(list)

    def _set_seed(self):
        skl.random.seed(self.seed)
        np.random.seed(self.seed)

    def _init_split(self, method: Literal["random"], **split_params):
        if method == "random":
            return RandomSplit(**split_params)
        else:
            raise NotImplementedError

    def preload_split(self, preload_path: Optional[Dict[str, str]]) -> bool:
        if preload_path is not None:
            for data_key, path in preload_path.items():
                if os.path.isdir(path):
                    split_path = os.path.join(path, "split_" + self.splitter.hash)
                    split_file = os.path.join(split_path, "split.npz")
                    if os.path.isfile(split_file):
                        logger.info(f"Split matched and preloaded from {split_path} for dataset {data_key}")
                        for part_key, indices in np.load(split_file).items():
                            self.split[part_key][data_key] = indices.tolist()
                else:
                    logger.warning(f"Split path {path} for dataset {data_key} not found")
                    return False
            return True
        return False
    
    def _save(self, preload_path: Dict[str, str]) -> None:
        for data_key, path in preload_path.items():
            if os.path.isdir(path):
                split_path = os.path.join(path, "split_" + self.splitter.hash)
                if os.path.exists(split_path):
                    logger.warning(f"Split path {split_path} exists and will be covered")
                os.makedirs(split_path, exist_ok=True)
                split_file = os.path.join(split_path, "split.npz")
                split_dict = dict()
                for part_key, part_info in self.split.items():
                    for k, indices in part_info.items():
                        if k == data_key:
                            split_dict[part_key] = np.array(indices)
                np.savez(split_file, **split_dict)
                handler = YamlHandler(os.path.join(split_path, "splitter.yaml"))
                splitter_config = addict.Dict({
                    "method": self.method,
                    "seed": self.seed,
                    **self.params
                })
                handler.write_yaml(splitter_config)
                logger.info(f"Save split for dataset {data_key} at {split_path}")
            else:
                raise FileNotFoundError(f"Preload path {path} for dataset {data_key} not found")

    def split(self, data: Dict[str, FieldDataset], preload_path: Optional[Dict[str, str]]=None) -> Dict[str, Dict[str, List[int]]]: 
        self._set_seed()
        self.split = self.splitter._get_empty_split(data)
        if self.preload and self.preload_split(preload_path):
            return self.split
        else:
            self.split = self.splitter.split(data)
            if self.save:
                self._save(preload_path)
            return self.split