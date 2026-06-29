import pickle
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the labeled pkl file without atomic charges", type=str)
    parser.add_argument("-d", "--working_dir", required=True, help="Path to the working directory that contains the chrg-12CM5 directory", type=str)
    parser.add_argument("-o", "--output", required=True, help="Path to the output pkl file with atomic charges", type=str)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    for i in range(1250):
        chg_file = f"{args.working_dir}/chrg-12CM5/{i}.chg"
        chg = np.loadtxt(chg_file, usecols=4)
        data[i]["chrg"] = chg

    with open(args.output, "wb") as f:
        pickle.dump(data, f)
