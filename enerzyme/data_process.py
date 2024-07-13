from .data import DataHub
from .utils import YamlHandler

class FFDataProcess(object):
    def __init__(self, config_path=None, out_dir=None, **params):
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.config_path = config_path
        self.out_dir = out_dir
        self.datahub = DataHub(dump_dir=out_dir, **config.Datahub)