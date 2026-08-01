import yaml
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the input pkl file", type=str)
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory", type=str)
    parser.add_argument("-t", "--tmp_dir", required=True, help="Path to the temporary directory", type=str)
    parser.add_argument("-n", "--dataset_name", required=True, help="Name of the labeled dataset", type=str)
    parser.add_argument("-np", "--n_processes", required=True, help="Number of processes", type=int, default=2)
    parser.add_argument("-cb", "--cpu_sbatch_header", required=False, help="CPU sbatch header", type=str)
    parser.add_argument("-gb", "--gpu_sbatch_header", required=False, help="GPU sbatch header", type=str)
    parser.add_argument("-ms", "--multiwfn_settings", required=True, help="MultiWfn settings.ini file", type=str)
    args = parser.parse_args()

    config = {
        "Supplier": {
            "path": args.input,
        },
        "QMDriver": {
            "engine": "TeraChem",
            "bs": "6-31gs",
            "xc": "b3lyp",
            "pcm": "cosmo",
            "dftd": "d3",
            "pcm_radii_file": os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "QM_cluster/utils/pcm_radii"))),
            "epsilon": 10,
            "keep_molden": True,
            "keep_output": False,
            "clean_tmp": True,
            "pickle_name": f"{args.dataset_name}.pkl",
            "dump_single_run": False,
            "n_processes": args.n_processes,
        },
    }

    with open(os.path.join(args.output_dir, "annotate.yaml"), "w") as f:
        yaml.dump(config, f)
    
    with open(os.path.join(args.output_dir, "qm_annotation.sh"), "w") as f:
        if args.gpu_sbatch_header is not None:
            with open(args.gpu_sbatch_header, "r") as h:
                f.write(h.read() + "\n")

        f.write(f"enerzyme annotate -c annotate.yaml -o {args.output_dir} -t {args.tmp_dir}")
    
    with open(os.path.join(args.output_dir, "multiwfn_annotation.sh"), "w") as f:
        if args.cpu_sbatch_header is not None:
            with open(args.cpu_sbatch_header, "r") as h:
                f.write(h.read() + "\n")

        f.write(f'''cp {args.multiwfn_settings} settings.ini
cd {os.path.splitext(os.path.basename(args.input))[0]}
mkdir chrg-12CM5
for i in `seq 0 1249`;
do 
molden=moldens/$i.molden
while [ ! -f "$molden" ]; do
    echo "$molden not found, waiting 2 minutes..."
    sleep 120
done
echo "calculate $molden"
Multiwfn << EOF
$molden
7
-16
1
y
0
q
EOF
mv *.chg chrg-12CM5
done
''')