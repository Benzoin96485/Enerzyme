from rdkit import Chem
from typing import Iterator, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
import pickle


class Supplier(ABC):
    name: str
    def __init__(self, supplying_fields: List[str]=["atom_type", "Ra", "Q", "mol"]):
        self.supplying_fields = set(supplying_fields)
        self.field_to_value = {
            "atom_type": lambda mol: np.array([atom.GetSymbol() for atom in mol.GetAtoms()]),
            "Ra": lambda mol: np.array(mol.GetConformer().GetPositions()),
            "Q": lambda mol: Chem.GetFormalCharge(mol),
            "mol": lambda mol: mol,
            "Za": lambda mol: np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
            "N": lambda mol: mol.GetNumAtoms(),
        }

    @abstractmethod
    def suppl(self) -> Iterator[Dict[str, Any]]:
        ...

    def get_package(self, mol: Chem.Mol) -> Dict[str, Any]:
        package = {}
        for field in self.supplying_fields:
            package[field] = self.field_to_value[field](mol)
        return package

    def raw_data(self) -> Dict[str, Any]:
        raw_data = {field: [] for field in self.supplying_fields}
        for package in self.suppl():
            for field in self.supplying_fields:
                raw_data[field].append(package[field])
        return raw_data


class SDFSupplier(Supplier):
    def __init__(self, sdf_file: str, start: int = 0, end: int = -1, supplying_fields: List[str]=["atom_type", "Ra", "Q", "mol"], **kwargs):
        super().__init__(supplying_fields)
        self.supplier = list(Chem.SDMolSupplier(sdf_file, removeHs=False))
        self.start = start
        to_end = True
        if end >= 0:
            self.end = end
            to_end = False
        else:
            self.end = len(self.supplier)
        self.name = sdf_file.split("/")[-1].split(".")[0] + (
            f"_{start}_{end}" if (start != 0 or not to_end) else ""
        )

    def suppl(self):
        i = self.start
        for mol in self.supplier[self.start:self.end]:
            package = self.get_package(mol)
            package.update({"index": i})
            yield package
            i += 1


class PickleSupplier:
    def __init__(self, pkl_file: str, start: int = 0, end: int = -1, **kwargs):
        with open(pkl_file, "rb") as f:
            self.supplier = pickle.load(f)
        self.start = start
        to_end = True
        if end >= 0:
            self.end = end
            to_end = False
        else:
            self.end = len(self.supplier)
        self.name = pkl_file.split("/")[-1].split(".")[0] + (
            f"_{start}_{end}" if (start != 0 or not to_end) else ""
        )
    
    def suppl(self):
        i = self.start
        for item in self.supplier[self.start:self.end]:
            yield {
                "atom_type": item["Za"],
                "Ra": item["Ra"],
                "Q": item["Q"],
                "index": i,
            }
            i += 1

def get_supplier(path: str, start: int = 0, end: int = -1, **kwargs) -> Supplier:
    if path.endswith(".sdf"):
        return SDFSupplier(path, start, end, **kwargs)
    elif path.endswith(".pkl"):
        return PickleSupplier(path, start, end, **kwargs)
    else:
        raise ValueError(f"File type of {path} not supported")