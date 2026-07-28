from rdkit import Chem
from typing import Iterator, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
from ase import Atoms
import pickle
import ase.io


class Supplier(ABC):
    def __init__(self, input_file, start: int = 0, end: int = -1):
        self.start = start
        to_end = True
        if end >= 0:
            self.end = end
            to_end = False
        else:
            self.end = None
        self.name = input_file.split("/")[-1].split(".")[0] + (
            f"_{start}_{end}" if (start != 0 or not to_end) else ""
        )
        
    @abstractmethod
    def suppl(self) -> Iterator[Atoms]:
        ...


class SDFSupplier(Supplier):
    def __init__(self, input_file, **kwargs):
        super().__init__(input_file, **kwargs)
        self.supplier = Chem.SDMolSupplier(input_file, removeHs=False)

    def suppl(self):
        i = self.start
        for i, mol in enumerate(self.supplier):
            if i < self.start:
                continue
            if i >= self.end:
                break
            atoms = Atoms(
                symbols=np.array([atom.GetSymbol() for atom in mol.GetAtoms()]),
                positions=np.array(mol.GetConformer().GetPositions()),
                pbc=False,
                info={
                    "index": i,
                    "charge": Chem.GetFormalCharge(mol),
                    "spin": 1
                }
            )
            yield atoms
            i += 1


class PickleSupplier(Supplier):
    def __init__(self, input_file, features: Dict[str, str], **kwargs):
        super().__init__(input_file, **kwargs)
        with open(input_file, "rb") as f:
            self.supplier = pickle.load(f)
        self.features = features

    def suppl(self):
        for i, data in enumerate(self.supplier[self.start:self.end]):
            atoms = Atoms(
                symbols=data[self.features["Za"]],
                positions=data[self.features["Ra"]],
                pbc=False,
                info={
                    "index": i + self.start,
                    "charge": data[self.features["Q"]] if "Q" in self.features else 0,
                    "spin": data[self.features["S"]] + 1 if "S" in self.features else 1,
                }
            )
            yield atoms


class XYZSupplier(Supplier):
    def __init__(self, input_file, Q: int=0, S: int=0, **kwargs):
        super().__init__(input_file, **kwargs)
        self.Q = Q
        self.S = S
        self.supplier = ase.io.read(input_file, index=slice(self.start, self.end))

    def suppl(self):
        for i, atoms in enumerate(self.supplier):
            if "charge" not in atoms.info:
                atoms.info["charge"] = self.Q
            if "spin" not in atoms.info:
                atoms.info["spin"] = self.S + 1
            if "index" not in atoms.info:
                atoms.info["index"] = i + self.start
            yield atoms


def get_supplier(path: str, start: int = 0, end: int = -1, **kwargs) -> Supplier:
    if path.endswith(".sdf"):
        return SDFSupplier(input_file=path, start=start, end=end, **kwargs)
    elif path.endswith(".pkl"):
        return PickleSupplier(input_file=path, start=start, end=end, **kwargs)
    elif path.endswith(".xyz"):
        return XYZSupplier(input_file=path, start=start, end=end, **kwargs)
    else:
        raise ValueError(f"File type of {path} not supported")
