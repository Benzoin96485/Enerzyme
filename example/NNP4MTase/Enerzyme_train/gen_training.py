from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_type", required=True, help="Type of the model", type=str, choices=["spookynet", "physnet", "mace"])
    parser.add_argument("-d", "--data_path", required=True, help="Path to the data file", type=str)
    parser.add_argument("-e", "--dielectric_constant", required=False, help="Dielectric constant", type=float, default=10.0)
    parser.add_argument("-q", "--qa_weight", required=False, help="Weight of the charge attribute", type=float, default=100.0)
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory", type=str)
    parser.add_argument("-b", "--sbatch_header", required=False, help="GPU sbatch header", type=str)
    args = parser.parse_args()

    config_template = os.path.join(os.path.dirname(__file__), "config_template", f"train_{args.model_type}.yaml")
    with open(config_template, "r") as f:
        config_template = f.read()

    config_template = config_template.replace("__DATA_PATH__", args.data_path)
    config_template = config_template.replace("__DIELECTRIC_CONSTANT__", str(args.dielectric_constant))
    config_template = config_template.replace("__QA_WEIGHT__", str(args.qa_weight))
    config_template = config_template.replace("__ATOMIC_ENERGY_PATH__", os.path.join(os.path.dirname(__file__), "atomic_energy.csv"))

    with open(os.path.join(args.output, f"train.yaml"), "w") as f:
        f.write(config_template)

    with open(os.path.join(args.output, f"train.sh"), "w") as f:
        if args.sbatch_header is not None:
            with open(args.sbatch_header, "r") as h:
                f.write(h.read() + "\n")
        f.write(f"enerzyme train -c train.yaml -o {args.output}")
