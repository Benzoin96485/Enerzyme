# Enerzyme
Towards next generation machine learning force field on enzymatic catalysis.

Current model architectures:
- PhysNet: [J. Chem. Theory Comput. 2019, 15, 3678−3693](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00181)
- SpookyNet: [Nat. Commun. 2021, 12(1), 7273](https://www.nature.com/articles/s41467-021-27504-0)

# Usage
## Installation

Recommended environment
```
python==3.10.12
pip==23.2.1
setuptools==68.1.2
h5py==3.9.0
numpy==1.24.3
addict==2.4.0
tqdm==4.66.1
joblib==1.3.2
pandas==2.1.0
pytorch==2.0.1
scikit-learn==1.3.0
ase==3.22.1
transformers==4.33.1
torch-ema==0.3
pyyaml==6.0.1
```

```bash
pip install -e .
```

## Training

Energy (force) / Atomic Charge / Dipole moment fitting.

```bash
enerzyme train -c <configuration yaml file> -o <output directory>
```
Please see `enerzyme/config/train.yaml` for details and recommended configurations.

Enerzyme saves the preprocessed dataset, split indices, final `<configuration yaml file>`, and the best model to the `<output directory>`.

## Evaluation

Energy (force) / Atomic Charge / Dipole moment prediction.

```bash
enerzyme predict -c <configuration yaml file> -o <output directory> -m <model directory>
```

Please see `enerzyme/config/predict.yaml` for details.

Enerzyme reads the `<model directory>` for the model configuration, load the models, predict the results from all active models, save the predicted values as a pickle in the corresponding model subfolders, and report the results as a csv file in the `<output directory>`.

## Simulation

Supported simulation types:
- Flexible scanning on the distance between two atoms.
- Constrained Langevin MD

```bash
enerzyme simulate -c <configuration yaml file> -o <output directory> -m <model directory>
```

Enerzyme reads the `<model directory>` for the model configuration, load the models, do simulation, and report the results in the `<output directory>`.