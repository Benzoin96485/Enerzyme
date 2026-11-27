import os
from collections import defaultdict, OrderedDict
from typing import Union, List, Dict, Optional, Literal
from hashlib import md5
import addict
from uuid import uuid4
from zipfile import BadZipFile, ZipFile
import numpy as np
from ..utils import logger, YamlHandler
from ..utils.util import atomic_output
from .distributed import is_global_zero, run_rank0_exclusive
from ..data.datahub import FieldDataset

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
        self.ratios = ratios.copy() if ratios is not None else None
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
                (part_key, OrderedDict([(list(data.keys())[0], [])])) for part_key in self.parts
            )
        else:
            return OrderedDict(
                (
                    part_info["name"], (
                        OrderedDict((source["dataset"], []) for source in part_info["sources"]) if "sources" in part_info
                        else OrderedDict([(part_info.get("dataset", list(data.keys())[0]), [])])
                    )
                ) for part_info in self.parts
            )


    def split(self, data: Dict[str, FieldDataset], rng=None):
        canonical_partition = defaultdict(dict)
        part_keys = []
        sorted_data_keys = defaultdict(list)
        if self.mode == "old":
            if len(data) > 1:
                raise ValueError("Source should be provided for multiple datasets")
            for i, part_key in enumerate(self.parts):
                if i >= len(self.ratios):
                    canonical_partition[part_key][list(data.keys())[0]] = -1
                else:
                    canonical_partition[part_key][list(data.keys())[0]] = self.ratios[i]
                sorted_data_keys[part_key].append(list(data.keys())[0])
            part_keys = self.parts
        else:
            for part_info in self.parts:
                if "name" not in part_info:
                    raise KeyError("Name should be provided for each part")
                if "sources" in part_info:
                    for source_info in part_info["sources"]:
                        if "dataset" not in source_info:
                            raise KeyError("Dataset should be provided for each source")
                        canonical_partition[part_info["name"]][source_info["dataset"]] = source_info.get("ratio", -1)
                        sorted_data_keys[part_info["name"]].append(source_info["dataset"])
                else:
                    data_key = part_info.get("dataset", list(data.keys())[0])
                    canonical_partition[part_info["name"]][data_key] = part_info.get("ratio", -1)
                    sorted_data_keys[part_info["name"]].append(data_key)
                part_keys.append(part_info["name"])

        l = {k: len(data[k]["Ra"]) for k in data.keys()}
        rest = {k: None for k in data.keys()}
        final_partition = defaultdict(dict)
        allocated_count = {k: 0 for k in data.keys()}

        for part_key, part_info in canonical_partition.items():
            for data_key, ratio in part_info.items():
                if isinstance(ratio, str) and ratio.endswith("%"):
                    ratio = float(ratio[:-1]) / 100
                    if not 0 <= ratio <= 1:
                        raise ValueError("Percentage must be between 0% and 100%")
                if ratio is None or ratio < 0:
                    if rest[data_key] is None:
                        rest[data_key] = part_key
                    else:
                        raise ValueError(f"Only one part can be used as the remaining part of the dataset {data_key}")
                    continue
                if isinstance(ratio, float) and ratio <= 1:
                    requested = round(ratio * l[data_key])
                elif isinstance(ratio, int):
                    requested = ratio
                else:
                    raise ValueError(f"Ratio or count {ratio} not valid")
                count = min(requested, l[data_key] - allocated_count[data_key])
                final_partition[part_key][data_key] = count
                allocated_count[data_key] += count

        for data_key, part_key in rest.items():
            if part_key is not None:
                final_partition[part_key][data_key] = max(0, l[data_key] - allocated_count[data_key])

        allocation = {k: [0] for k in data.keys()}
        for part_key in part_keys:
            for data_key in sorted_data_keys[part_key]:
                data_count = final_partition[part_key][data_key]
                allocation[data_key].append(data_count + allocation[data_key][-1])
            for data_key in data.keys() - sorted_data_keys[part_key]:
                allocation[data_key].append(allocation[data_key][-1])

        full_indices = {k: list(range(l[k])) for k in data.keys()}
        rng = np.random if rng is None else rng
        for indices in full_indices.values():
            rng.shuffle(indices)
        return OrderedDict(
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

    def _init_split(self, method: Literal["random"], **split_params):
        if method == "random":
            return RandomSplit(**split_params)
        else:
            raise NotImplementedError

    def preload_split(self, preload_path: Optional[Dict[str, str]]) -> bool:
        if not preload_path:
            return False
        loaded = {}
        generations = set()
        for data_key, path in preload_path.items():
            split_file = os.path.join(path, "split_" + self.splitter.hash, "split.npz")
            expected = {part for part, sources in self.split.items() if data_key in sources}
            try:
                with np.load(split_file) as parts:
                    generations.add(parts.zip.comment)
                    if set(parts.files) != expected:
                        return False
                    loaded[data_key] = {part: parts[part].tolist() for part in expected}
            except (OSError, ValueError, EOFError, BadZipFile):
                return False
        if len(generations) != 1:
            # Retry an interrupted multi-dataset publication.
            return False
        for data_key, parts in loaded.items():
            for part, indices in parts.items():
                self.split[part][data_key] = indices
        return True

    def _save(self, preload_path: Dict[str, str]) -> None:
        generation = uuid4().hex
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
                            split_dict[part_key] = np.array(indices, dtype=np.int64)
                handler = YamlHandler(os.path.join(split_path, "splitter.yaml"))
                splitter_config = addict.Dict({
                    "method": self.method,
                    "seed": self.seed,
                    **self.params
                })
                with atomic_output(handler.file_path) as path:
                    handler.write_yaml(splitter_config, path)
                with atomic_output(split_file) as path:
                    np.savez(path, **split_dict)
                    # Leave the arrays compatible with consumers iterating over every key.
                    with ZipFile(path, "a") as archive:
                        archive.comment = generation.encode("ascii")
                logger.info(f"Save split for dataset {data_key} at {split_path}")
            else:
                raise FileNotFoundError(f"Preload path {path} for dataset {data_key} not found")

    def get_split(self, data: Dict[str, FieldDataset], preload_path: Optional[Dict[str, str]]=None) -> Dict[str, Dict[str, List[int]]]: 
        self.split = self.splitter._get_empty_split(data)

        def prepare():
            if self.preload and self.preload_split(preload_path):
                return
            self.split = self.splitter.split(data, np.random.RandomState(self.seed))
            if self.save and preload_path:
                self._save(preload_path)

        if self.save and preload_path:
            sync_dir = next(iter(preload_path.values()))
            run_rank0_exclusive(prepare, sync_dir=sync_dir, name=f"split_{self.splitter.hash}")
            if not is_global_zero() and not self.preload_split(preload_path):
                raise RuntimeError(f"Published split is unavailable in {sync_dir}")
        else:
            # Each rank obtains the same partition without writing an ephemeral split.
            prepare()
        return self.split
