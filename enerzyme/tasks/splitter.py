import os
from hashlib import md5
from addict import Dict
from sklearn.model_selection import KFold
import sklearn as skl
import numpy as np
from ..utils import logger, YamlHandler


class RandomSplit:
    def __init__(self, parts, ratios, **params):
        self.parts = parts
        self.ratios = ratios.copy()
        splitstr = "random" + str(parts) + str(ratios)
        self.hash = md5(splitstr.encode("utf-8")).hexdigest()[:16]
        
    def split(self, data):
        rest = None
        l = len(data["Ra"])
        print(l)
        partition = dict()
        allocated_count = 0
        if len(self.ratios) != len(self.parts):
            if len(self.ratios) == len(self.parts - 1):
                self.ratios.append(-1)
            else:
                raise IndexError("Ratios should share the same length as the parts")
        for part, ratio in zip(self.parts, self.ratios):
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
        for part in self.parts:
            allocation.append(partition[part] + allocation[-1])
        idx = np.arange(l)
        np.random.shuffle(idx)
        return {k: idx[allocation[i]:allocation[i+1]] for i, k in enumerate(self.parts)}


class Splitter:
    def __init__(self, method='random', seed=114514, preload=True, save=True, **params):
        self.method = method
        self.seed = seed
        self.params = params
        self.splitter = self._init_split(method, **params)
        self.preload = preload
        self.save = save
        self.split

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
            splitter_config = Dict({
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