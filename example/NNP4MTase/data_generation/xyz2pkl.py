import pickle
import random
import ase.io
from argparse import ArgumentParser
from ase.atoms import Atoms
from typing import List

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the input md-traj.xyz file", type=str)
    parser.add_argument("-o", "--output", required=True, help="Path to the output pkl file", type=str)
    args = parser.parse_args()

    frames: List[Atoms] = ase.io.read(args.input, index=slice(50, 1300))
    n_atoms = len(frames[0])

    datapoints = []
    for frame in frames:
        datapoint = {
            "Za": frame.get_atomic_numbers(),
            "Ra": frame.get_positions(),
            "Q": frame.info.get("charge", 0),
        }
        datapoints.append(datapoint)

    random.shuffle(datapoints)

    with open(args.output, "wb") as f:
        pickle.dump(datapoints, f)