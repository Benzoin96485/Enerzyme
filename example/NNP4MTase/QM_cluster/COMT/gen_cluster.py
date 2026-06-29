import pandas as pd
import os
import shutil
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--master_list", type=str, required=True)
    parser.add_argument("-i", "--input_csv", type=str, required=True)
    parser.add_argument("-c", "--config_yaml", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-q", "--qp_output_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.master_list)
    df_COMT = df[df["enzyme"] == "COMT"]
    
    qp_df = pd.DataFrame(
        {  
            "pdb_id": df_COMT["pdb_id"],
            "center": df_COMT["substrate"].apply(lambda x: f'SAM_{x}_MG'),
            "multiplicity": None,
            "oxidation": None
        }
    )
    qp_df.to_csv(args.input_csv, index=False)

    # subprocess.run(["qp", "run", "-c", args.config_yaml])

    for _, df_row in df_COMT.iterrows():
        pdb_id = df_row["pdb_id"]
        center = df_row["center"]
        cluster_dir = os.path.join(args.output_dir, f"COMT_{pdb_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        raw_structure_dir = os.path.join(cluster_dir, "raw")
        os.makedirs(raw_structure_dir, exist_ok=True)
        qp_dir = os.path.join(args.qp_output_dir, f"{pdb_id}")
        reference_pdb = os.path.join(qp_dir, center, f"{center}.pdb")
        reference_sdf = os.path.join(qp_dir, "Protoss", f"{pdb_id}_ligands.sdf")
        shutil.copy(reference_pdb, raw_structure_dir)
        shutil.copy(reference_sdf, raw_structure_dir)

        with open(os.path.join(cluster_dir, "gen_scan.sh"), "w") as f:
            f.write(f"python ../../utils/gen_scan.py -m ../../master_list.csv -e COMT -p {pdb_id}\n")