import torch
import numpy as np


class Trainer(object):
    def __init__(self, task=None, metrics_str=None, out_dir=None, **params):
        self.task = task
        self.out_dir = out_dir
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        print(params)
        ### init common params ###
        self.split_method = params.get('Common').get('split_method','5fold_random')
        self.split_seed = params.get('Common').get('split_seed', 42)
        self.seed = params.get('Common').get('seed', 42)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, self.split_seed)
        self.logger_level = int(params.get('Common').get('logger_level'))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('NNtrainer').get('learning_rate', 1e-4))
        self.batch_size = params.get('NNtrainer').get('batch_size', 32)
        self.max_epochs = params.get('NNtrainer').get('max_epochs', 50)
        self.warmup_ratio = params.get('NNtrainer').get('warmup_ratio', 0.1)
        self.patience = params.get('NNtrainer').get('patience', 10)
        self.max_norm = params.get('NNtrainer').get('max_norm', 1.0)
        self.cuda = params.get('NNtrainer').get('cuda', False)
        self.amp = params.get('NNtrainer').get('amp', False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' and self.amp==True else None
    
    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):

    dataloader_func = NNDATALOADER_REGISTER.get(feature_name, TorchDataLoader)
    dataloader = dataloader_func(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    collate_fn=collate_fn,
                                    drop_last=drop_last)
    return dataloader