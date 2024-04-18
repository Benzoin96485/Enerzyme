import argparse
from mlff.train import FFTrain
from mlff.predict import FFPredict
from mlff.utils.base_logger import logger


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
    parser_train.add_argument('-c', '--config_path', type=str, default='', 
        help='training config'
    )
    parser_train.add_argument('-o', '--output_dir', type=str, default='../results',
                    help='the output directory for saving artifact')

    parser_predict = subparsers.add_parser(
        "predict",
        help="predict Enerzyme command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_predict.add_argument('--data_path', type=str, help='test data path')
    parser_predict.add_argument('--model_dir', type=str,
                    help='the output directory for saving artifact')
    parser_predict.add_argument('--save_dir', type=str,
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
        save_dir=args.save_dir
    )
    molpredict.predict()


if __name__ == '__main__':
    args = get_parser()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    
    logger.info("job complete")