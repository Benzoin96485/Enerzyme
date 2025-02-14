from rdkit import Chem
from typing import Iterator, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class Supplier(ABC):
    name: str
    @abstractmethod
    def suppl(self) -> Iterator[Dict[str, Any]]:
        ...

    def get_package(self, mol: Chem.Mol) -> Dict[str, Any]:
        atom_type = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        Ra = np.array(mol.GetConformer().GetPositions())
        Q = Chem.GetFormalCharge(mol)
        return {
            "atom_type": atom_type,
            "Ra": Ra,
            "Q": Q,
            "mol": mol,
        }


class SDFSupplier(Supplier):
    def __init__(self, sdf_file: str, start: int = 0, end: int = -1, **kwargs):
        super().__init__()
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


def get_supplier(path: str, start: int = 0, end: int = -1, **kwargs) -> Supplier:
    if path.endswith(".sdf"):
        return SDFSupplier(path, start, end, **kwargs)
    else:
        raise ValueError(f"File type of {path} not supported")