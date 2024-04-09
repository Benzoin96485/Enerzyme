from sklearn.model_selection import KFold


class Splitter(object):
    def __init__(self, split_method='fold_random', **split_params):
        self.split_method = split_method
        self.split_params = split_params
        self.splitter = self._init_split(self.split_method, **split_params)

    def _init_split(self, split_method, **split_params):
        if split_method == "fold_random":
            splitter = KFold(
                n_splits=split_params.get("n_splits", 5), 
                shuffle=True, 
                random_state=split_params.get("seed", 114514)
            )
        return splitter

    def split(self, data):
        if self.split_method in ['5fold_random']:
            self.skf = self.splitter.split(data)
        return self.skf