# Enerzyme
Next generation machine learning force field on enzymatic catalysis

# Usage
## Training
```bash
python main.py -c config.yaml -o output_dir
```
Charge / Energy (force) / Dipole moment fitting are all supported now. The config.yaml template is currently at `mlff/config/q_default.yaml` for charge fitting and `mlff/config/qe_default.yaml`for charge and energy fitting.

The dataset in `data_path` should be a python pickle file containing a list of each frame's dictionary like
```python
{
    "chrg": [...] # np.ndarray of shape (N,): atomic charges in e
    "energy": ... # float: energy in Ha
    "grad": [...] # np.ndarray of shape (N,3): energy gradient in Ha/Angstrom
    "coord": [] # np.ndarray of shape (N,3): coordinates
    "atom_type": [] # list of length N: atom types (element symbols in upper case)
    "dipole": [] # np.ndarray of shape (3,): dipole in e·Angstrom
}
```

# Test
The `test_physnet.py` tests if the behavior of our torch implementation of PhysNet is the same as the [official tensorflow repo](https://github.com/MMunibas/PhysNet?tab=readme-ov-file) and the [paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00181). 
```
cd test
pytest -v
```