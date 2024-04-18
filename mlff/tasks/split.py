from sklearn.model_selection import KFold
import numpy as np
from numpy.random import shuffle


class TrainValTestSplit:
    def __init__(self, ratio, **split_params):
        sum_ratio = sum(ratio)
        self.train_cut = ratio[0] / sum_ratio
        self.val_cut = (ratio[0] + ratio[1]) / sum_ratio
        pass

    def split(self, data):
        l = len(data)
        idx = np.arange(l)
        shuffle(idx)
        train_cut = round(self.train_cut * l)
        val_cut = round(self.val_cut * l)
        return [(idx[:train_cut], idx[train_cut:val_cut], idx[val_cut:])]
    

class TrainValWithheldTestSplit:
    def __init__(self, ratio, **split_params):
        sum_ratio = sum(ratio)
        self.train_cut = ratio[0] / sum_ratio
        self.val_cut = (ratio[0] + ratio[1]) / sum_ratio
        self.withheld_cut = (ratio[0] + ratio[1] + ratio[2]) / sum_ratio
        pass

    def split(self, data):
        l = len(data)
        idx = np.arange(l)
        shuffle(idx)
        train_cut = round(self.train_cut * l)
        val_cut = round(self.val_cut * l)
        withheld_cut = round(self.withheld_cut * l)
        return [(idx[:train_cut], idx[train_cut:val_cut], idx[withheld_cut:])]


class Splitter:
    def __init__(self, split_method='fold_random', **split_params):
        self.split_method = split_method
        self.split_params = split_params
        self.splitter = self._init_split(self.split_method, **split_params)
        self.cv = True if "fold" in split_method else False

    def _init_split(self, split_method, **split_params):
        match split_method:
            case "fold_random":
                splitter = KFold(
                    n_splits=split_params.get("n_splits", 5), 
                    shuffle=True, 
                    random_state=split_params.get("seed", 114514)
                )
            case "train_val_test_random":
                splitter = TrainValTestSplit(
                    ratio=split_params.get("ratio", [0.7, 0.1, 0.2])
                )
            case "train_val_withheld_test_random":
                splitter = TrainValWithheldTestSplit(
                    ratio=split_params.get("ratio", [0.7, 0.1, 0, 0.2])
                )
            case _:
                splitter = KFold(
                    n_splits=split_params.get("n_splits", 5), 
                    shuffle=True, 
                    random_state=split_params.get("seed", 114514)
                )
        return splitter

    def split(self, data):
        return self.splitter.split(data)