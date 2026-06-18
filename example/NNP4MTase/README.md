# NNP4MTase

This example is for a paper in preparation about the application of Enerzyme-NNPs to the methyltransferase QM cluster models.

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

## QM cluster models

The data is from the QM cluster models of the methyltransferase.