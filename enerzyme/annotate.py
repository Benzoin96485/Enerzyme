from .data.supplier import get_supplier
from .utils import YamlHandler

_SUPPORTED_ENGINES = ("terachem", "gpu4pyscf")


class QMAnnotate:
    def __init__(self, config_path: str, tmp_dir: str, out_dir: str, start: int = 0, end: int = -1):
        config = YamlHandler(config_path).read_yaml()
        self.driver_config = config["QMDriver"]
        self.supplier_config = config["Supplier"]
        supplier = get_supplier(**self.supplier_config, start=start, end=end)
        engine = str(self.driver_config["engine"]).lower()
        if engine == "terachem":
            from .qm.qm_driver import TeraChemDriver
            self.driver = TeraChemDriver(supplier=supplier, tmp_dir=tmp_dir, output_dir=out_dir, **self.driver_config)
        elif engine == "gpu4pyscf":
            from .qm.qm_driver import GPU4PySCFDriver
            self.driver = GPU4PySCFDriver(supplier=supplier, tmp_dir=tmp_dir, output_dir=out_dir, **self.driver_config)
        else:
            raise ValueError(
                f"Unknown QM engine {self.driver_config['engine']!r}; "
                f"expected one of {_SUPPORTED_ENGINES}"
            )

    def drive(self):
        self.driver.run()
