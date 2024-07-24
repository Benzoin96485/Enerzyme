import argparse
from .train import FFTrain
from .predict import FFPredict
from .simulate import FFSimulate
from .data_process import FFDataProcess
from .utils.base_logger import logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Enerzyme cli",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    parser_train = subparsers.add_parser(
        "train",
        help="train Enerzyme command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument('-c', '--config_path', type=str, 
        help='training config'
    )
    parser_train.add_argument('-o', '--output_dir', type=str, default='../results',
                    help='the output directory for saving artifact')

    parser_predict = subparsers.add_parser(
        "predict",
        help="predict Enerzyme command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_predict.add_argument('-d', '--data_path', type=str, help='test data path')
    parser_predict.add_argument('-m', '--model_dir', type=str,
                    help='the directory of models')
    parser_predict.add_argument('-o', '--output_dir', type=str, default='../results',
                help='the output directory for saving artifact')    
    parser_predict.add_argument('-c', '--config_path', type=str, 
        help='prediction config'
    )

    parser_simulate = subparsers.add_parser(
        "simulate",
        help="simulate Enerzyme command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_simulate.add_argument('-c', '--config_path', type=str, default='', 
        help='simulation config'
    )
    parser_simulate.add_argument('-m', '--model_dir', type=str,
                    help='the directory of models')

    parser_data_process = subparsers.add_parser(
        "data_process",
        help="Process and save preloaded data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_data_process.add_argument('-c', '--config_path', type=str, default='', 
        help='data process config'
    )
    parser_data_process.add_argument('-o', '--output_dir', type=str, default='../results',
        help='the output directory for saving artifact')

    args = parser.parse_args()
    return args


def train(args):
    moltrain = FFTrain(
        out_dir=args.output_dir,
        config_path=args.config_path
    )
    moltrain.train_all()


def predict(args):
    molpredict = FFPredict(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_path=args.config_path
    )
    molpredict.predict()


def simulate(args):
    molcalculate = FFSimulate(
        model_dir=args.model_dir,
        config_path=args.config_path
    )
    molcalculate.run()


def data_process(args):
    FFDataProcess(
        out_dir=args.output_dir,
        config_path=args.config_path
    )


def main():
    args = get_parser()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)   
    elif args.command == 'simulate':
        simulate(args)
    elif args.command == 'data_process':
        data_process(args)
    else:
        raise NotImplementedError(f"Command {args.command} is not supported now.")
    logger.info("job complete")


if __name__ == '__main__':
    main()