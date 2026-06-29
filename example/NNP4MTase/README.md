# NNP4MTase

This example is for the paper *Enerzyme: A Framework for Efficient Training of Reactive Neural Network Potentials for Enzyme Catalysis with Application to Methyltransferases* in submission.

## Environment

Create a conda environment from :code:`requirements.yaml`:

```bash
    conda env create -f requirements.yaml
    conda activate nnp4mtase
```

This will install the core dependencies. Besides, we need to install the following additional dependencies:

- `enerzymette`: Install from the `NNP4MTase` release of the [Enerzymette repository](https://github.com/Benzoin96485/Enerzymette) with `pip install .`.
- `torch-scatter`: Based on the Python and PyTorch version in the `requirements.yaml` and the CUDA 12.9 in our environment, we installed the wheel of the [`2.1.2+pt28cu129` version](https://data.pyg.org/whl/torch-2.8.0%2Bcu129/torch_scatter-2.1.2%2Bpt28cu129-cp313-cp313-linux_x86_64.whl).
- `fairchem-core`: Although the package is already installed in the conda environment, we need to get a license to access the [UMA models](https://huggingface.co/facebook/UMA).
- ORCA: We use the ORCA v6.1.0 obtained from the [ORCA Forum](https://orcaforum.kofo.mpg.de/app.php/dlext/?cat=26).
- TeraChem: We use the TeraChem v1.9.
- QuantumPDB: We use the QuantumPDB obtained from the [QuantumPDB repository](https://github.com/hjkgrp/quantumPDB).
- MultiWfn: We use the MultiWfn v3.8 obtained from the [MultiWfn official website](http://sobereva.com/multiwfn/).

## Overview

The workflow proceeds in four stages: (1) build QM cluster models and run reference DFT (`QM_cluster/`), (2) generate and label training data from steered MD (`data_generation/`), (3) train reactive NNP models (`Enerzyme_train/`), and (4) run NNP-driven scan/NEB with Enerzymette (`Enerzymette/`). All enzyme-specific parameters are centralized in `QM_cluster/master_list.csv`.

## master_list.csv

Central registry of all methyltransferase systems. Columns:

| Column | Description |
|--------|-------------|
| `enzyme` | Enzyme family name (`COMT`, `PfPMT`, `HcgC`) |
| `pdb_id` | PDB identifier for COMT structures; empty for PfPMT and HcgC systems |
| `center` | Residue cluster center used by QuantumPDB / `raw/` layout |
| `substrate` | Residue name of the methyl acceptor |
| `nucleophile` | Atom name of the nucleophile on the substrate |
| `target_x1` | Target C–N bond length (Å) for the DFT scan endpoint |
| `pdb_file` | Optional explicit path to the cluster PDB (relative to this file); if empty, `raw/{center}.pdb` is used |

Most scripts accept `-m master_list.csv -e ENZYME [-p PDB_ID]` to look up a row. When `-p` is omitted, the row with an empty `pdb_id` is selected (PfPMT and HcgC). For COMT rows, `-p` must be supplied (e.g. `-p 1VID`).

## QM_cluster

### utils/gen_scan.py

Generates TeraChem input files and a `job.sh` for the three-step DFT workflow: reactant optimization → constrained bond scan → product optimization. From `master_list.csv` it resolves the cluster PDB, computes the total charge, and sets up backbone frozen atoms and scan constraints.

```bash
cd QM_cluster/COMT/COMT_1VID          # or Yang-case/{PfPMT, HcgC}
python ../../utils/gen_scan.py -m ../../master_list.csv -e COMT -p 1VID
bash job.sh
```

Optional flags: `-b` (SLURM header), `-i` (initial reactant XYZ), `-t` (scratch directory), `-g` (MO guess from previous step).

### utils/gen_neb.py

Generates ORCA NEB input (`neb.inp`), a TeraChem wrapper template, and `job.sh`. Charge and backbone constraints are derived from `master_list.csv` (same lookup as `gen_scan.py`).

```bash
python ../../utils/gen_neb.py -m ../../master_list.csv -e COMT -p 1VID \
    --reactant reactant.xyz --product product.xyz
bash job.sh
```

Optional flags: `-n` (number of images, default 8), `-b` (SLURM header), `-t` (TS guess XYZ).

### utils/pcm_radii

PCM cavity radii file copied into each TeraChem working directory by `gen_scan.py` and referenced in annotation configs.

### COMT/

Scripts and templates for building COMT cluster models with [QuantumPDB](https://github.com/hjkgrp/quantumPDB).

- **`qp_config_template.yaml`**: QuantumPDB configuration template. Placeholders `__INPUT_CSV__` and `__OUTPUT_DIR__` are substituted by `gen_cluster.sh`.
- **`gen_cluster.sh`**: Prepares `qp_config.yaml`, then calls `gen_cluster.py`.
- **`gen_cluster.py`**: Reads COMT rows from `master_list.csv`, writes a QuantumPDB input CSV, copies cluster PDB/SDF from QuantumPDB output into `COMT_{pdb_id}/raw/`, and writes a per-system `gen_scan.sh`.

```bash
cd QM_cluster/COMT
# Run QuantumPDB first (qp run -c qp_config.yaml), then:
bash gen_cluster.sh
```

Each `COMT_{pdb_id}/` directory is created with `raw/{center}.pdb`, `raw/{pdb_id}_ligands.sdf`, and a ready-to-run `gen_scan.sh`.

### Yang-case/

Reference cluster models from Yang et al. (J. Phys. Chem. Lett. 2019, 10, 3779–3787). Each subdirectory (`PfPMT/`, `HcgC/`) contains:

- **`raw/`**: Capped cluster structures (`cluster-capped.pdb`, etc.); see `raw/readme.md` for provenance.
- **`gen_scan.sh`**: One-liner invoking `gen_scan.py` with the corresponding `-e` flag.

```bash
cd QM_cluster/Yang-case/PfPMT
bash gen_scan.sh
```

## data_generation

Scripts for building the NNP training dataset from metadynamics trajectories.

### sampling.py

Runs constrained steered MD (xTB + PLUMED) on a QM cluster. Reads initial structure, charge, substrate, and nucleophile from `master_list.csv`. Outputs `metad-traj.xyz`, `metad.log`, and `plumed.log` under `-o`.

```bash
python data_generation/sampling.py -o ./output \
    -m QM_cluster/master_list.csv -e PfPMT
```

### xyz2pkl.py

Converts an MD trajectory (`metad-traj.xyz`) into an unlabeled Enerzyme pickle (frames 50–1299, shuffled). Each frame stores `Za`, `Ra`, and total charge `Q`.

```bash
python data_generation/xyz2pkl.py -i output/metad-traj.xyz -o unlabeled.pkl
```

### gen_annotation.py

Generates `annotate.yaml`, `qm_annotation.sh`, and `multiwfn_annotation.sh` for DFT labeling (TeraChem energies/gradients via `enerzyme annotate`) and atomic charge calculation (MultiWfn 1.2*CM5).

```bash
python data_generation/gen_annotation.py \
    -i unlabeled.pkl -o ./annotate -t /scratch/tmp \
    -n labeled -np 4 -ms settings.ini
```

### collect.py

Merges MultiWfn 1.2*CM5 charges from `chrg-12CM5/{i}.chg` into the labeled pickle produced by `enerzyme annotate`.

```bash
python data_generation/collect.py \
    -i labeled.pkl -d ./annotate -o labeled_with_chrg.pkl
```

## Enerzyme_train

### gen_training.py

Fills a model-specific training config template and writes `train.yaml` + `train.sh`. Supports `spookynet`, `physnet`, and `mace`.

```bash
python Enerzyme_train/gen_training.py \
    -m physnet -d ../data_generation/labeled_with_chrg.pkl \
    -o ./train_physnet -e 10.0 -q 100.0
bash train_physnet/train.sh
```

### config_template/

YAML templates (`train_physnet.yaml`, `train_mace.yaml`, `train_spookynet.yaml`) with placeholders for data path, dielectric constant, atomic charge loss weight, and atomic energy reference.

### atomic_energy.csv

Per-element reference energies (H, C, N, O, Mg, P, S, Cl, …) used by the training config for energy decomposition.

## Enerzymette

### gen_task.py

Generates SLURM scripts for ML-driven scan or NEB using a trained Enerzyme model. Requires a DFT scan directory containing optimized `reactant.xyz`, `product.xyz` (for NEB), and `scan.in`.

```bash
python Enerzymette/gen_task.py \
    -m ./train_physnet/checkpoints -o ./ml_neb \
    -t neb -s ../QM_cluster/COMT/COMT_1VID
bash ml_neb/neb.sh
```

For scan: `-t scan`. Copies `server.yaml` into the output directory for NEB jobs.

### server.yaml

Enerzyme inference server settings passed to `enerzymette enerzyme_neb`.
