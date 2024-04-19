import os
import joblib
import argparse
import pandas as pd
from .utils import YamlHandler, logger
from .data import DataHub
from .tasks import Trainer
from .models import ModelHub


class FFPredict(object):
    def __init__(self, model_dir=None, data_path=None, save_dir=None, metric_str=None):
        if not model_dir:
            raise ValueError("model_dir is None")
        if not data_path:
            raise ValueError("data_path is None")
        if not save_dir:
            raise ValueError("save_dir is None")
        self.model_dir = model_dir
        self.data_path = data_path
        self.save_dir = save_dir

        config_path = os.path.join(model_dir, 'config.yaml')
        config = YamlHandler(config_path).read_yaml()
        logger.info('Config: {}'.format(config))
        self.task = config.Base.task
        self.metric_str = metric_str
        
        # ### load from ckp will initialize the datahub, featurehub, modelhub, ensembler
        self.load_from_ckp(**config)

    def load_from_ckp(self, **params):
        params['Datahub'].pop("data_path")
        ## load test data
        self.datahub = DataHub(
            data_path=self.data_path, 
            task=self.task, 
            is_train=False, 
            dump_dir=self.model_dir, 
            **params['Datahub']
        )
        self.trainer = Trainer(
            task=self.task, 
            out_dir=self.model_dir, 
            metrics_str=self.metric_str, 
            **params['Trainer']
        )
        self.modelhub = ModelHub(self.datahub, self.trainer, task=self.task, **params['Modelhub'])
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
        metrics = {k: [] for k in self.metrics.metrics_str}
        for ff_name, ff in FFs.items():
            #ff.evaluate(checkpoints_path=os.path.join(self.model_dir, ff_name))
            ff.evaluate()
            data = self.datahub.data
            result = dict()
            pred = data["target_scaler"].inverse_transform(ff.cv["test_pred"])
            target = data["target_scaler"].inverse_transform(data["target"])
            for k, v in target.items():
                if k != "atom_type":
                    result[f"predict_{k}"] = pred[k]
            for k, v in data.items():
                if k not in ["target", "target_scaler"]:
                    result[k] = v
            df = pd.DataFrame(result)
            os.makedirs(self.save_dir, exist_ok=True)
            df.to_pickle(os.path.join(self.save_dir, f'{ff_name}-result.pkl'))

            for k, v in self.metrics.cal_metric(pd.DataFrame(target), pred).items():
                metrics[k].append(v)
        metrics_df = pd.DataFrame(metrics, index=FFs.keys())
        metrics_df.to_csv(os.path.join(self.save_dir, f'metric.csv'))
        logger.info(f"final predict metrics score: \n{metrics_df.T}")
        logger.info("pipeline finish!")

    def update_config(self, config, **params):
        for key, value in params.items():
            config[key] = value
        return config


def get_parser():
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument('--data_path', type=str, help='test data path')
    parser.add_argument('--model_dir', type=str,
                    help='the output directory for saving artifact')
    parser.add_argument('--save_dir', type=str,
                help='the output directory for saving artifact')    
    parser.add_argument('--metric_str', nargs="+", type=str,
                help='the metric names separated by commas')    
    args = parser.parse_args()
    return args