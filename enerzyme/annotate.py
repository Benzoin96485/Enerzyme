from .data import Supplier, get_supplier
from .utils import YamlHandler

class QMAnnotate:
    def __init__(self, config_path: str, tmp_dir: str, out_dir: str):
        config = YamlHandler(config_path).read_yaml()
        self.driver_config = config["QMDriver"]
        self.supplier_config = config["Supplier"]
        supplier = get_supplier(**self.supplier_config)
        if self.driver_config["engine"].lower() == "terachem":
            from .qm.qm_driver import TeraChemDriver
            self.driver = TeraChemDriver(supplier, tmp_dir, out_dir, **self.driver_config)

    def drive(self):
        self.driver.run()
