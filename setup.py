from setuptools import setup, find_packages

setup(
    name='enerzyme',
    version='0.0.1',
    install_requires=['ase', 'joblib', 'addict'],
    entry_points={'console_scripts': ['enerzyme=enerzyme.cli:main']},
    packages=find_packages(include=["enerzyme", "enerzyme.*"]),
    package_data={"enerzyme": [
        "data/periodic-table.csv", 
        "models/physnet/tables/*.npy", 
        "models/spookynet/modules/d4data/*.pth"
    ]},
    auth='Benzoin96485',
    author_email='luowl7@mit.edu',
    description='Next generation machine learning force field on enzymatic catalysis',
    url='https://github.com/Benzoin96485/Enerzyme',
    zip_safe=True,
)
