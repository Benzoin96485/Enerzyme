from .predict import FFPredict
from .tasks.extractor import Extractor
from .utils import YamlHandler

class FFExtract:
    def __init__(self, 
        predictor: FFPredict,
        model_dir: str,
        output_dir: str,
        config_path: str,
    ) -> None:
        self.predictor = predictor
        config = YamlHandler(config_path).read_yaml()
        self.extractor = Extractor(**config.Extractor)
        self.model_dir = model_dir
        self.output_dir = output_dir

    def extract(self) -> None:
        predict_results = self.predictor._simple_predict(["Ra"])
        for ff_name, predict_result in predict_results.items():
            y_pred = predict_result["y_pred"]
            self.extractor.build_fragment(y_pred, prefix=f"{self.output_dir}/{ff_name}")
