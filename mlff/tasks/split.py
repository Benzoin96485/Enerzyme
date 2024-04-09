class Splitter(object):
    def __init__(self, split_method='5fold_random', seed=42):
        self.split_method = split_method
        self.seed = seed
        self.splitter = self._init_split(self.split_method, self.seed)
        self.n_splits = 5
        self.skf = None

    def _init_split(self, split_method, seed=42):
        pass

    def split(self, data, target=None, group=None):
        pass