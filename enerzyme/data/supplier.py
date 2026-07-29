from rdkit import Chem
from typing import Iterator, Dict, Any, List, Optional
from abc import ABC, abstractmethod
import numpy as np
from ase import Atoms
import pickle
import ase.io


# Identity mapping for unlabeled annotate pickles (e.g. xyz2pkl: Ra/Za/Q).
# Override via Supplier.features in YAML for remapped keys (coord/atom_type/…).
_DEFAULT_PICKLE_FEATURES: Dict[str, str] = {
    "Ra": "Ra",
    "Za": "Za",
    "Q": "Q",
    "S": "S",
}


class Supplier(ABC):
    def __init__(self, input_file, start: int = 0, end: int = -1):
        self.input_file = input_file
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
        self._open_supplier()

    def _open_supplier(self) -> None:
        # RDKit SDMolSupplier is not picklable; recreate from path after unpickle.
        self.supplier = Chem.SDMolSupplier(self.input_file, removeHs=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("supplier", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_supplier()

    def suppl(self):
        for i, mol in enumerate(self.supplier):
            if i < self.start:
                continue
            if self.end is not None and i >= self.end:
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


class PickleSupplier(Supplier):
    def __init__(
        self,
        input_file,
        features: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(input_file, **kwargs)
        with open(input_file, "rb") as f:
            self.supplier = pickle.load(f)
        self.features = features if features is not None else dict(_DEFAULT_PICKLE_FEATURES)

    def suppl(self):
        for i, data in enumerate(self.supplier[self.start:self.end]):
            q_key = self.features.get("Q")
            s_key = self.features.get("S")
            atoms = Atoms(
                symbols=data[self.features["Za"]],
                positions=data[self.features["Ra"]],
                pbc=False,
                info={
                    "index": i + self.start,
                    # Defaults when the mapped key is absent from the frame
                    # (e.g. xyz2pkl unlabeled pickles omit S): charge=0, spin=1 (S=0).
                    "charge": data[q_key] if q_key is not None and q_key in data else 0,
                    "spin": data[s_key] + 1 if s_key is not None and s_key in data else 1,
                }
            )
            yield atoms


class XYZSupplier(Supplier):
    def __init__(self, input_file, Q: int=0, S: int=0, **kwargs):
        super().__init__(input_file, **kwargs)
        self.Q = Q
        self.S = S
        # ase.io.read returns Atoms for an int index / single image, list for a slice.
        frames = ase.io.read(input_file, index=slice(self.start, self.end))
        self.supplier = [frames] if isinstance(frames, Atoms) else list(frames)

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
