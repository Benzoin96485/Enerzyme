from .data import DataHub
from .tasks import Splitter
from .utils import YamlHandler

class FFCollect(object):
    def __init__(self, config_path=None, out_dir=None, **params):
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.config_path = config_path
        self.out_dir = out_dir
        self.datahub = DataHub(dump_dir=out_dir, **config.Datahub)
        self.splitter = Splitter(**config.Trainer.Splitter)
        self.splitter.split(self.datahub.data, preload_path=self.datahub.preload_path)