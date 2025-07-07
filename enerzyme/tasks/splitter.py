import os
from collections import defaultdict
from typing import Union, List, Dict, Optional, Literal
from hashlib import md5
import addict
import sklearn as skl
import numpy as np
from ..utils import logger, YamlHandler
from ..data import FieldDataset


class RandomSplit:
    def __init__(self, 
        parts: Union[
            List[str], 
            Dict[str, Union[float, int, str]],
            Dict[str, Union[
                Dict[Literal["source", "ratio"], Union[str, float, int]],
                List[Dict[Literal["source", "ratio"], Union[str, float, int]]]
            ]]
        ],
        ratios: Optional[List[float]]=None, 
        **params
    ):
        self.parts = parts
        self.ratios = ratios.copy()
        self.mode = "old"

        # sanity check
        if isinstance(parts, list):
            if ratios is None:
                raise ValueError("Ratios should be provided if parts is a list")
        elif isinstance(parts, dict):
            if all(isinstance(v, (dict, list)) for v in parts.values()):
                self.mode = "full"
            elif all(isinstance(v, (float, int)) for v in parts.values()):
                self.mode = "single_simple"
            elif all(isinstance(v, str) for v in parts.values()):
                self.mode = "multi_simple"
        
        splitstr = "random" + str(parts) + str(ratios)
        self.hash = md5(splitstr.encode("utf-8")).hexdigest()[:16]

    def split_single(self, data: FieldDataset):
        # convert to old style
        if self.mode == "multi_simple":
            raise ValueError("Ratio should be provided for a single dataset")
        elif self.mode == "single_simple":
            part_keys = list(self.parts.keys())
            ratios = [self.parts[k] for k in part_keys]
        elif self.mode == "full":
            part_keys = list(self.parts.keys())
            ratios = []
            for k in part_keys:
                if not isinstance(self.parts[k], dict):
                    raise ValueError(f"One part shouldn't have multiple entries")
                if "source" in self.parts[k]:
                    raise ValueError("Source shouldn't be provided for a single dataset")
            ratios = [self.parts[k].get("ratio", -1) for k in part_keys]
        elif self.mode == "old":
            part_keys = self.parts
            ratios = self.ratios + [None] * (len(part_keys) - len(self.ratios))

        l = len(data["Ra"])
        rest = None
        partition = dict()
        allocated_count = 0

        for part, ratio in zip(part_keys, ratios):
            if isinstance(ratio, str) and ratio[-1] == "%":
                ratio = float(ratio[:-1]) / 100
                if ratio >= 1:
                    raise ValueError("Percentage should be less than 100%")
            elif ratio is None or ratio < 0:
                if rest is None:
                    rest = part
                else:
                    raise ValueError("Only one part can be used as the remaining part")
            elif ratio < 1:
                count = min(round(ratio * l), l - allocated_count)
                partition[part] = count
                allocated_count += count
            elif isinstance(ratio, int):
                partition[part] = min(ratio, l - allocated_count) 
                allocated_count += ratio
            else:
                raise ValueError(f"Ratio or count {ratio} not valid")
        if rest is not None:
            partition[rest] = max(l - allocated_count, 0)
        logger.info(f"Final partition: {partition}")

        allocation = [0]
        for part in part_keys:
            allocation.append(partition[part] + allocation[-1])
        idx = np.arange(l)
        np.random.shuffle(idx)
        return {k: idx[allocation[i]:allocation[i+1]] for i, k in enumerate(part_keys)}
    
    def split_list(self, data: List[FieldDataset]):
        l = [len(data[i]["Ra"]) for i in range(len(data))]
        rest = [None for _ in range(len(data))]
        partition = dict()
        allocated_count = [0 for _ in range(len(data))]

    def split_dict(self, data: Dict[str, FieldDataset]):
        canonical_partition = defaultdict(list)
        if self.mode in ["single_simple", "old"]:
            if len(data) > 1:
                raise ValueError("Source should be provided for multiple datasets")
            else:
                if self.mode == "old":
                    for i, part_key in enumerate(self.parts):
                        if i >= len(self.ratios):
                            canonical_partition[part_key] = [{data.keys()[0]: -1}]
                        else:
                            canonical_partition[part_key] = [{data.keys()[0]: self.ratios[i]}]
                elif self.mode == "single_simple":
                    for part_key, ratio in self.parts.items():
                        canonical_partition[part_key] = [{data.keys()[0]: ratio}]
        elif self.mode == "multi_simple":
            for part_key, data_key in self.parts.items():
                if data_key in data.keys():
                    canonical_partition[part_key] = [{data_key: -1}]
                else:
                    raise ValueError(f"Data key {data_key} not found in data")
        elif self.mode == "full":
            for part_key, part_info in self.parts.items():
                if not isinstance(part_info, dict):
                    part_info_list = [part_info]
                else:
                    part_info_list = part_info
                for part_info_dict in part_info_list:
                    if "source" in part_info_dict:
                        if part_info_dict["source"] not in data.keys():
                            raise ValueError(f"Source {part_info_dict['source']} not found in data")
                        else:
                            canonical_partition[part_key] = [{part_info_dict["source"]: part_info_dict["ratio"]}]
                    else:
                        if len(data) > 1:
                            raise ValueError("Source should be provided for multipledatasets")
                        else:
                            canonical_partition[part_key].append({data.keys()[0]: part_info_dict["ratio"]})

        l = {k: len(data[k]["Ra"]) for k in data.keys()}
        rest = {k: None for k in data.keys()}
        partition = defaultdict(int)
        allocated_count = {k: 0 for k in data.keys()}

        for part_key, part_info_list in canonical_partition.items():
            for part_info in part_info_list:
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
                        partition[part_key][data_key] = count
                        allocated_count[data_key] += count
                    elif isinstance(ratio, int):
                        partition[part_key][data_key] = min(ratio, l[data_key] - allocated_count[data_key])
                        allocated_count[data_key] += ratio
                    else:
                        raise ValueError(f"Ratio or count {ratio} not valid")
        for k, v in rest.items():
            if v is not None:
                partition[v].append({k: max(0, l[k] - allocated_count[k])})

        allocation = {k: [0] for k in data.keys()}
        for part_key, part_info in partition.items():
            for data_key, data_count in part_info.items():
                allocation[data_key].append(data_count + allocation[data_key][-1])
        idx = {k: list(range(l[k])) for k in data.keys()}
        for k, v in idx.items():
            np.random.shuffle(v)
        return {
            part_key: {
                data_key: idx[data_key][allocation[data_key][i]:allocation[data_key][i+1]]
                for data_key in data.keys()
            } for i, part_key in enumerate(partition.keys())
        }
    
    def split(self, data: Union[FieldDataset, List[FieldDataset], Dict[str, FieldDataset]]):
        if isinstance(data, FieldDataset):
            return self.split_single(data)
        elif isinstance(data, list):
            return self.split_dict({str(i): v for i, v in enumerate(data)})
        elif isinstance(data, dict):
            return self.split_dict(data)


class Splitter:
    def __init__(self, method='random', seed=114514, preload=True, save=True, **params):
        self.method = method
        self.seed = seed
        self.params = params
        self.splitter = self._init_split(method, **params)
        self.preload = preload
        self.save = save

    def _set_seed(self):
        skl.random.seed(self.seed)
        np.random.seed(self.seed)

    def _init_split(self, method, **split_params):
        if method == "random":
            return RandomSplit(**split_params)
        else:
            raise NotImplementedError

    def preload_split(self, preload_path):
        if preload_path is not None and os.path.isdir(preload_path):
            split_path = os.path.join(preload_path, "split_" + self.splitter.hash)
            split_file = os.path.join(split_path, "split.npz")
            if os.path.isfile(split_file):
                logger.info(f"Split matched and preloaded from {split_path}")
                self.split = {k: v for k, v in np.load(split_file).items()}
                return True
        return False
    
    def _save(self, preload_path):
        if os.path.isdir(preload_path):
            split_path = os.path.join(preload_path, "split_" + self.splitter.hash)
            if os.path.exists(split_path):
                logger.warning(f"Split path {split_path} exists and will be covered")
            os.makedirs(split_path, exist_ok=True)
            split_file = os.path.join(split_path, "split.npz")
            np.savez(split_file, **self.split)
            handler = YamlHandler(os.path.join(split_path, "splitter.yaml"))
            splitter_config = addict.Dict({
                "method": self.method,
                "seed": self.seed,
                **self.params
            })
            handler.write_yaml(splitter_config)
            logger.info(f"Save split at {split_path}")
        else:
            raise FileNotFoundError(f"Preload path {preload_path} not found")

    def split(self, data, preload_path=None): 
        self._set_seed()
        if self.preload and self.preload_split(preload_path):
            return self.split
        else:
            self.split = self.splitter.split(data)
            if self.save:
                self._save(preload_path)
            return self.split