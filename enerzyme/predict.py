import os
import pandas as pd
from .utils import YamlHandler, logger
from .data import DataHub
from .tasks import Trainer
from .models import ModelHub


class FFPredict(object):
    def __init__(self, model_dir=None, output_dir=None, config_path=None):
        if model_dir is None:
            raise ValueError("model_dir is None")
        self.model_dir = model_dir
        self.output_dir = model_dir if output_dir is None else model_dir

        model_config_path = os.path.join(model_dir, 'config.yaml')
        config = YamlHandler(model_config_path).read_yaml()
        new_config = YamlHandler(config_path).read_yaml()
        if config_path is not None:
            for k, v in new_config.Datahub.items():
                if k not in ["transforms", "neighbor_list"]:
                    config.Datahub[k] = v
            config.Metric = new_config.Metric

        logger.info('Config: {}'.format(config))
        
        ### load from ckp will initialize the datahub and the modelhub
        self.load_from_ckp(**config)

    def load_from_ckp(self, **params):
        ## load test data
        self.datahub = DataHub(
            dump_dir=self.output_dir, 
            **params['Datahub']
        )
        self.trainer = Trainer(
            out_dir=self.model_dir, 
            metric_config=params['Metric'], 
            **params['Trainer']
        )
        self.modelhub = ModelHub(self.datahub, self.trainer, **params['Modelhub'])
        self.metrics = self.trainer.metrics
    
    # def save_predict(self, data, dir, name="predict"):
    #     prefix = self.data_path.split('/')[-1].split('.')[0]
    #     run_id = 0
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     else:
    #         folders = [x for x in os.listdir(dir)]
    #         while name + f'_{run_id}' + '.csv' in folders:
    #             run_id += 1
    #     name = prefix + '.' + name + f'_{run_id}' + '.csv'
    #     path = os.path.join(dir, name)
    #     data.to_csv(path)
    #     logger.info("save predict result to {}".format(path))

    def predict(self):
        FFs = self.modelhub.models.get('FF', None)
        metrics = []
        for ff_name, ff in FFs.items():
            result = dict()
            #ff.evaluate(checkpoints_path=os.path.join(self.model_dir, ff_name))
            y_pred, metric_score = ff.evaluate()
            
            for k, v in self.datahub.targets.items():
                result[f"predict_{k}"] = y_pred[k]
                result[k] = v
            os.makedirs(self.output_dir, exist_ok=True)
            pd.DataFrame({k: [vi for vi in v] for k, v in result.items()}).to_pickle(os.path.join(self.datahub.preload_path, f"{ff_name}-prediction.pkl"))
            metrics.append(metric_score)
        metrics_df = pd.concat(metrics)
        metrics_df.to_csv(os.path.join(self.output_dir, 'metric.csv'))
        logger.info(f"final predict metrics score: \n{metrics_df.T}")
        logger.info("pipeline finish!")

    # def update_config(self, config, **params):
    #     for key, value in params.items():
    #         config[key] = value
    #     return config


# def get_parser():
#     # Before creating the true parser, we need to import optional user module
#     # in order to eagerly import custom tasks, optimizers, architectures, etc.
#     parser = argparse.ArgumentParser()

#     # fmt: off
#     parser.add_argument('--data_path', type=str, help='test data path')
#     parser.add_argument('--model_dir', type=str,
#                     help='the output directory for saving artifact')
#     parser.add_argument('--save_dir', type=str,
#                 help='the output directory for saving artifact')    
#     parser.add_argument('--metric_str', nargs="+", type=str,
#                 help='the metric names separated by commas')    
#     args = parser.parse_args()
#     return args