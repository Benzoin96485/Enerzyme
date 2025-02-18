from setuptools import setup, find_packages

setup(
    name='enerzyme',
    version='0.3.1',
    install_requires=['numpy', 'h5py', 'tqdm', 'ase', 'joblib', 'addict', 
                      'pandas', 'torch', 'scikit-learn', 'transformers',
                      'torch-ema', 'pyyaml', 'torch-scatter', 'rdkit'],
    entry_points={'console_scripts': ['enerzyme=enerzyme.cli:main']},
    packages=find_packages(include=["enerzyme", "enerzyme.*"]),
    package_data={"enerzyme": [
        "data/periodic-table.csv", 
        "models/layers/dispersion/grimme_d3_tables/*.npy", 
        "models/layers/dispersion/grimme_d4_tables/*.pth"
    ]},
    auth='Benzoin96485',
    author_email='luowl7@mit.edu',
    description='Next generation machine learning force field on enzymatic catalysis',
    url='https://github.com/Benzoin96485/Enerzyme',
    zip_safe=True,
)
