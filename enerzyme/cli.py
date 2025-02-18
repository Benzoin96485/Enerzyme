import argparse
from .utils.base_logger import logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Enerzyme cli",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    parser_train = subparsers.add_parser(
        "train",
        help="train a machine learning force field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.add_argument('-c', '--config_path', type=str, 
        help='training config'
    )
    parser_train.add_argument('-o', '--output_dir', type=str, default='../results',
                    help='the output directory for saving artifact')

    parser_predict = subparsers.add_parser(
        "predict",
        help="predict properties of molecules with machine learning force fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_predict.add_argument('-m', '--model_dir', type=str,
                    help='the directory of models')
    parser_predict.add_argument('-o', '--output_dir', type=str, default='../results',
                help='the output directory for saving artifact')    
    parser_predict.add_argument('-c', '--config_path', type=str, 
        help='prediction config'
    )

    parser_simulate = subparsers.add_parser(
        "simulate",
        help="simulate molecules with machine learning force fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_simulate.add_argument('-c', '--config_path', type=str, default='', 
        help='simulation config'
    )
    parser_simulate.add_argument('-m', '--model_dir', type=str,
                    help='the directory of models')
    parser_simulate.add_argument('-o', '--output_dir', type=str, default='../results',
        help='the output directory for saving artifact')
    
    parser_extract = subparsers.add_parser(
        "extract",
        help="extract substructures from molecules with machine learning force fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_extract.add_argument('-c', '--config_path', type=str, default='', 
        help='extraction config'
    )
    parser_extract.add_argument('-m', '--model_dir', type=str,
                    help='the directory of models')
    parser_extract.add_argument('-o', '--output_dir', type=str, default='../results',
        help='the output directory for saving artifact')
    parser_extract.add_argument('-s', '--skip_prediction', default=False, action='store_true')

    parser_collect = subparsers.add_parser(
        "collect",
        help="Process and save preloaded data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_collect.add_argument('-c', '--config_path', type=str, default='', 
        help='data process config'
    )
    parser_collect.add_argument('-o', '--output_dir', type=str, default='../results',
        help='the output directory for saving artifact')

    parser_annotate = subparsers.add_parser(
        "annotate",
        help="Drive QM calculations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_annotate.add_argument('-c', '--config_path', type=str, default='', 
        help='QM annotate config'
    )
    parser_annotate.add_argument('-t', '--tmp_dir', type=str, default='.', 
        help='QM annotate tmp directory'
    )
    parser_annotate.add_argument('-o', '--output_dir', type=str, default='.', 
        help='QM annotate output directory'
    )
    parser_annotate.add_argument('-s', '--start', type=int, default=0, 
        help='QM annotate start index'
    )
    parser_annotate.add_argument('-e', '--end', type=int, default=-1, 
        help='QM annotate end index'
    )

    parser_bond = subparsers.add_parser(
        "bond",
        help="Give bonds to molecules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_bond.add_argument('-p', '--pdb_path', type=str, default='', 
        help='input pdb file path'
    )
    parser_bond.add_argument('-m', '--mol_path', type=str, default='', 
        help='output mol file path'
    )
    parser_bond.add_argument('-i', '--img_path', type=str, default='', 
        help='output img file path'
    )
    parser_bond.add_argument('-t', '--template_path', type=str, default='', 
        help='template sdf file path'
    )
    args = parser.parse_args()
    return args


def train(args):
    from .train import FFTrain
    moltrain = FFTrain(
        out_dir=args.output_dir,
        config_path=args.config_path
    )
    moltrain.train_all()


def predict(args):
    from .predict import FFPredict
    molpredict = FFPredict(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config_path
    )
    molpredict.predict()


def simulate(args):
    from .simulate import FFSimulate
    molsimulate = FFSimulate(
        model_dir=args.model_dir,
        config_path=args.config_path,
        out_dir=args.output_dir
    )
    molsimulate.run()


def extract(args):
    from .predict import FFPredict
    from .extract import FFExtract
    molpredict = FFPredict(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config_path
    )
    molextract = FFExtract(
        predictor=molpredict,
        model_dir=args.model_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        skip_prediction=args.skip_prediction
    )
    molextract.extract()


def collect(args):
    from .collect import FFCollect
    FFCollect(
        out_dir=args.output_dir,
        config_path=args.config_path
    )


def annotate(args):
    from .annotate import QMAnnotate
    molannotate = QMAnnotate(
        config_path=args.config_path,
        tmp_dir=args.tmp_dir,
        out_dir=args.output_dir,
        start=args.start,
        end=args.end
    )
    molannotate.drive()


def bond(args):
    from .bond import pdb2mol
    pdb2mol(args.pdb_path, args.mol_path, args.img_path, args.template_path)


def main():
    args = get_parser()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)   
    elif args.command == 'simulate':
        simulate(args)
    elif args.command == 'collect':
        collect(args)
    elif args.command == 'extract':
        extract(args)
    elif args.command == 'annotate':
        annotate(args)
    elif args.command == 'bond':
        logger.disabled = True
        bond(args)
    else:
        raise NotImplementedError(f"Command {args.command} is not supported now.")
    logger.info("job complete")


if __name__ == '__main__':
    main()