from mlff.train import get_parser, FFTrain
from mlff.utils.base_logger import logger


if __name__ == '__main__':
    args = get_parser()
    
    moltrain = FFTrain(
        out_dir=args.output_dir,
        config_path=args.config_path
    )

    moltrain.train_all()

    logger.info("train complete")