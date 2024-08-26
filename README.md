# Enerzyme
Towards next generation machine learning force field on enzymatic catalysis.

Current model architectures:
- PhysNet: [J. Chem. Theory Comput. 2019, 15, 3678−3693](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00181)
- SpookyNet: [Nat. Commun. 2021, 12(1), 7273](https://www.nature.com/articles/s41467-021-27504-0)

# Usage
## Installation
```bash
pip install -e .
```

## Training

Energy (force) / Atomic / Charge Dipole moment fitting.

```bash
enerzyme train -c <configuration yaml file> -o <output directory>
```
Please see `enerzyme/config/train.yaml` for details and recommended configurations.

Enerzyme saves the preprocessed dataset, split indices, final `<configuration yaml file>`, and the best model to the `<output directory>`.

## Evaluation

Energy (force) / Atomic / Charge Dipole moment prediction.

```bash
enerzyme predict -c <configuration yaml file> -o <output directory> -m <model directory>
```

Please see `enerzyme/config/predict.yaml` for details.

Enerzyme reads the `<model directory>` for the model configuration, load the models, predict the results from all active models, save the predicted values as a pickle in the corresponding model subfolders, and report the results as a csv file in the `<output directory>`.

## Simulation

Scanning on the distance between two atoms is supported.

```bash
enerzyme simulate -c <configuration yaml file> -o <output directory> -m <model directory>
```

Enerzyme reads the `<model directory>` for the model configuration, load the models, do simulation, and report the results in the `<output directory>`.