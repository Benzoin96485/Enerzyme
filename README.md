# Enerzyme
Next generation machine learning force field on enzymatic catalysis

# Usage
## Training
```bash
python main.py -c config.yaml -o output_dir
```
Only charge fitting is supported now. The config.yaml template is currently at `mlff/config/q_default.yaml`.

The dataset in `data_path` should be a python pickle file containing a list of each frame's dictionary like
```python
{
    "chrg": [...] # np.ndarray of shape (N,): atomic charges
    "coord": [] # np.ndarray of shape (N,3): coordinates
    "atom_type": [] # list of length N: atom types (element symbols in upper case)
}
