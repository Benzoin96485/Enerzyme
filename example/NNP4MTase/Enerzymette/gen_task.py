import os
from argparse import ArgumentParser
import shutil

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", required=True, help="The directory that contains the model folder with checkpoints and the config.yaml generated during model training", type=str)
    parser.add_argument("-o", "--output_dir", required=True, help="The output directory", type=str)
    parser.add_argument("-t", "--task", required=True, help="The task to be performed", type=str, choices=["scan", "neb"])
    parser.add_argument("-s", "--scan_dir", required=True, help="The DFT scan directory that contains the optimized reactant, product (for NEB), and scan.in terachem inputfile.", type=str)
    parser.add_argument("-b", "--sbatch_header", required=False, help="The sbatch header file", type=str)
    args = parser.parse_args()

    if args.task == "neb":
        with open(os.path.join(args.output_dir, "neb.sh"), "r") as f:
            if args.sbatch_header is not None:
                with open(args.sbatch_header, "r") as h:
                    f.write(h.read() + "\n")
            f.write(f"""enerzymette enerzyme_neb \
    -r {os.path.join(args.scan_dir, "reactant.xyz")} \
    -p {os.path.join(args.scan_dir, "product.xyz")} \
    -o {args.output_dir} \
    -m {args.model_dir} \
    -q {os.path.join(args.scan_dir, "scan.in")} \
    -c server.yaml \
    -n 8 \
    -b 5001 \
    -i stdout
""")
        shutil.copy(os.path.join(os.path.dirname(__file__), "server.yaml"), os.path.join(args.output_dir, "server.yaml"))
    elif args.task == "scan":
        with open(os.path.join(args.output_dir, "scan.sh"), "r") as f:
            if args.sbatch_header is not None:
                with open(args.sbatch_header, "r") as h:
                    f.write(h.read() + "\n")
            f.write(f"""enerzymette enerzyme_scan \
    -r {os.path.join(args.scan_dir, "reactant.xyz")} \
    -o {args.output_dir} \
    -m {args.model_dir} \
    -q {os.path.join(args.scan_dir, "scan.in")} \
    -n 25
""")