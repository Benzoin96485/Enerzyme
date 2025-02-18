# Enerzyme

Towards next-generation machine learning force fields for enzymatic catalysis.

Currently supported model architectures:

|   Model   |   Type   | Energy and force prediction | Charge and dipole prediction | Fully modulized | Shallow ensemble |                                           Reference paper                                           |               Reference code               |
| :-------: | :------: | :-------------------------: | :--------------------------: | :-------------: | :--------------: | :-------------------------------------------------------------------------------------------------: | :----------------------------------------: |
|  PhysNet  | internal |             ✅             |              ✅              |       ✅       |        ✅        | [J. Chem. Theory Comput. 2019, 15, 3678–3693](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00181) | [Github](https://github.com/MMunibas/PhysNet) |
| SpookyNet | internal |             ✅             |              ✅              |       ✅       |        ✅        |          [Nat. Commun. 2021, 12(1), 7273](https://www.nature.com/articles/s41467-021-27504-0)          | [Github](https://github.com/OUnke/SpookyNet) |
|  LEFTNet  | internal |             ✅             |              ✅              |       ❌       |        ❌        |                   [NeurIPS 2023, arXiv:2304.04757](https://arxiv.org/abs/2304.04757)                   |  [Github](https://github.com/yuanqidu/M2Hub)  |
|   MACE   | external |             ✅             |              ❌              |       ❌       |        ❌        |                   [NeurIPS 2022, arXiv:2206.07697](https://arxiv.org/abs/2206.07697)                   |   [Github](https://github.com/ACEsuit/mace)   |
|  NequIP  | external |             ✅             |              ❌              |       ❌       |        ❌        |          [Nat. Commun. 2022, 13(1), 2453](https://www.nature.com/articles/s41467-022-29939-5)          | [Github](https://github.com/mir-group/nequip) |
|  XPaiNN  | external |             ✅             |              ❌              |       ❌       |        ❌        | [J. Chem. Theory Comput. 2024, 20, 21, 9500–9511](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01151) | [Github](https://github.com/X1X1010/XequiNet) |

# Usage

## Installation

Recommended environment for internal force fields

```
python==3.12.8
pip==25.0
setuptools==75.8.0
h5py==3.12.1
numpy==2.2.2
addict==2.4.0
tqdm==4.67.1
joblib==1.4.2
pandas==2.2.3
pytorch==2.5.1
scikit-learn==1.6.1
ase==3.24.0
transformers==4.48.2
torch-ema==0.3
pyyaml==6.0.2
torch-scatter==2.1.2
e3nn==0.5.5
```

To test PhysNet, you also need

```
tensorflow==2.13.0
```

To invoke MACE, you need

```
mace-torch==0.3.6
```

To invoke NequIP, you need

```
nequip==0.6.1
```

To invoke XPaiNN, you need

```
XequiNet==0.3.6
scipy==1.11.2
pyscf==2.7.0
torch_geometric==2.5.3
pytorch-warmup==0.1.1
pydantic==1.10.12
```

Then install the package

```bash
pip install -e .
```

## Training

Energy (force) / Atomic Charge / Dipole moment fitting.

```bash
enerzyme train -c <configuration yaml file> -o <output directory>
```

Please see `enerzyme/config/train.yaml` for details and recommended configurations.

Enerzyme saves the preprocessed dataset, split indices, final `<configuration yaml file>`, and the best/last model to the `<output directory>`.

### Active Learning Training

Please see `enerzyme/config/active_learning_train.yaml` for details and recommended configurations.

## Evaluation

Energy (force) / Atomic Charge / Dipole moment prediction.

```bash
enerzyme predict -c <configuration yaml file> -o <output directory> -m <model directory>
```

Please see `enerzyme/config/predict.yaml` for details.

Enerzyme reads the `<model directory>` for the model configuration, load the models, predict the results from all active models, save the predicted values as a pickle in the corresponding model subfolders, and report the results as a csv file in the `<output directory>`.

## Simulation

Supported simulation types:

- Constrained optimization. See `enerzyme/config/opt.yaml`
- Constrained flexible scan on the distance between two atoms. See `enerzyme/config/scan.yaml`
- Constrained Langevin MD. See `enerzyme/config/nvt_md.yaml`

```bash
enerzyme simulate -c <configuration yaml file> -o <output directory> -m <model directory>
```

Enerzyme reads the `<model directory>` for the model configuration, load the models, do simulation, and report the results in the `<output directory>`.

## Fragment Extraction

Extract fragments based on local uncertainty from the prediction

```bash
enerzyme extract -c <configuration yaml file> -o <output directory> -m <model directory>
```

Please see `enerzyme/config/extract.yaml` for details.

## Data annotation

Label molecules with energies, forces and dipoles from QM calculation. Please get your QM engine's environment prepared first.

```bash
enerzyme annotate -c <configuration yaml file> -o <output directory> -t <temporary directory> -s <start index> -e <end index>
```

Please see `enerzyme/config/annotate.yaml` for details.

## Bond order assignment

Guess bond orders from pdb file. Compatible with [QuantumPDB](https://github.com/davidkastner/quantumPDB), where the pdb file should be the output of cluster building and the template sdf file should be ligands.sdf.

```bash
enerzyme bond -p <pdb file> -m <output mol file> -i <output image file> -t <template sdf file>
```
