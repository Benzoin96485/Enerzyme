import os, pathlib
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Union, List, Optional
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils import logger
from . import PERIODIC_TABLE_PATH


PERIODIC_TABLE = pd.read_csv(PERIODIC_TABLE_PATH, index_col="atom_type")
REVERSED_PERIODIC_TABLE = pd.read_csv(PERIODIC_TABLE_PATH, index_col="Za")


def parse_Za(atom_types: Iterable[Union[str, int]]) -> Union[np.ndarray, List[int]]:
    if isinstance(atom_types[0], str):
        # numpy.str_ is an instance of str
        return PERIODIC_TABLE.loc[atom_types]["Za"].to_numpy()
    elif isinstance(atom_types, np.ndarray):
        # numpy.int is not an instance of int
        return atom_types.astype(int)
    elif isinstance(atom_types[0], int):
        return np.array(atom_types)
    else:
        logger.info("Parsing atom type")
        Zas = []
        for atom_types_ in tqdm(atom_types):
            Zas.append(parse_Za(atom_types_))
        return Zas


def load_atomic_energy(atomic_energy_path: str) -> pd.DataFrame:
    if os.path.exists(atomic_energy_path):
        atomic_energies = pd.read_csv(atomic_energy_path)
        atomic_energies["Za"] = parse_Za(atomic_energies["atom_type"])
        atomic_energies.set_index("Za", inplace=True)
        atomic_energies.loc[0] = {"atom_type": "", "atomic_energy": 0}
        return atomic_energies
    else:
        raise FileNotFoundError(f"Atomic energy file {atomic_energy_path} not found!")


class BaseTransform(ABC):
    def __init__(self, major_key: str, *args, **kwargs) -> None:
        self.major_key = major_key
    
    @abstractmethod
    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        ...

    def inverse_transform(self, new_output: Dict[str, Iterable], selected_indices: Optional[Iterable[int]]=None) -> None:
        if selected_indices is None:
            for i in range(len(new_output[self.major_key])):
                self.single_inverse_transform(new_output, i)
        else:
            for i in selected_indices:
                self.single_inverse_transform(new_output, i)
    

class AtomicEnergyTransform(BaseTransform):
    def __init__(self, atomic_energy_path: str, simulation_mode=False, *args, **kwargs) -> None:
        super().__init__(major_key="E")
        self.atomic_energies = load_atomic_energy(atomic_energy_path)
        self.transform_type = "shift"

    def transform(self, new_input: Dict[str, Iterable]) -> None:
        if "E" not in new_input:
            return
        logger.info("Calculating total atomic energy offset")
        if len(new_input["Za"]) == 1:
            for i in tqdm(range(len(new_input["E"]))):
                new_input["E"][i] -= sum(self.atomic_energies.loc[new_input["Za"][0]]["atomic_energy"])
        else:
            for i in tqdm(range(len(new_input["E"]))):
                new_input["E"][i] -= sum(self.atomic_energies.loc[new_input["Za"][i]]["atomic_energy"])
    
    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        if len(new_output["Za"]) == 1:
            new_output["E"][idx] += sum(self.atomic_energies.loc[new_output["Za"][0]]["atomic_energy"])
        else:
            new_output["E"][idx] += sum(self.atomic_energies.loc[new_output["Za"][idx]]["atomic_energy"])
    

class NegativeGradientTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(major_key="Fa")
        self.transform_type = "scale"

    def transform(self, new_input):
        if "Fa" in new_input:
            for i in range(len(new_input["Fa"])):
                new_input["Fa"][i] = -new_input["Fa"][i]
    
    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        new_output["Fa"][idx] = -new_output["Fa"][idx]


class TotalEnergyNormalization(BaseTransform):
    def __init__(self, preload_path=".", scale=None, shift=None):
        super().__init__(major_key="E")
        self.transform_type = "normalization"
        self.scale = 1
        self.shift = 0
        self.loaded = True
        self.statistics = os.path.join(preload_path, "statistics.data")
        if scale is not None:
            self.scale = scale  
        if shift is not None:
            self.shift = shift
        if scale is None and shift is None:
            self.loaded = False
            if os.path.isfile(self.statistics):
                stat = joblib.load(self.statistics)
                self.scale = stat["scale"]
                self.shift = stat["shift"]
                self.loaded = True
        else:
            joblib.dump({"shift": self.shift, "scale": self.scale}, self.statistics)

    def transform(self, new_input):
        if "E" not in new_input:
            return
        if not self.loaded:
            logger.info("Calculating total energy normalization statistics")    
            self.shift = np.mean(new_input["E"])
            self.scale = np.std(new_input["E"])
            joblib.dump({"shift": self.shift, "scale": self.scale}, self.statistics)
            self.loaded = True
        logger.info(f"Total energy normalization: mean {self.shift}, std {self.scale}")
        for i in range(len(new_input["E"])):
            new_input["E"][i] = (new_input["E"][i] - self.shift) / self.scale
            new_input["Fa"][i] /= self.scale

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        if not self.loaded:
            raise RuntimeError("Shift and scale parameters not loaded")
        new_output["E"][idx] = new_output["E"][idx] * self.scale + self.shift
        new_output["Fa"][idx] *= self.scale


class Transform:
    def __init__(self, transform_args: Optional[Dict]=None, preload_path: Optional[str]=None, simulation_mode: bool=False) -> None:
        self.transform_args = transform_args
        self.backup_keys = set()
        self.shifts = []
        self.scales = []
        self.normalizations = []
        if transform_args is None:
            return
        for k, v in transform_args.items():
            if k == "atomic_energy":
                self.shifts.append(AtomicEnergyTransform(v))
                self.backup_keys.add("E")
            if k == "negative_gradient" and v and (not simulation_mode):
                self.scales.append(NegativeGradientTransform())
                self.backup_keys.add("Fa")
            if k == "total_energy_normalization" and v:
                if v is None:
                    v = preload_path
                if isinstance(v, str):
                    self.normalizations.append(TotalEnergyNormalization(v))
                elif isinstance(v, dict):
                    self.normalizations.append(TotalEnergyNormalization(**v))
                else:
                    raise ValueError(f"Invalid total energy normalization: {v}")
                self.backup_keys.add("E")

    def transform(self, raw_input: Dict):
        for shift in self.shifts:
            shift.transform(raw_input)
        for scale in self.scales:
            scale.transform(raw_input)
        for normalization in self.normalizations:
            normalization.transform(raw_input)

    def inverse_transform(self, raw_output: Dict, selected_indices: Optional[Iterable[int]]=None):
        for normalization in self.normalizations:
            normalization.inverse_transform(raw_output, selected_indices)
        for scale in self.scales:
            scale.inverse_transform(raw_output, selected_indices)
        for shift in self.shifts:
            shift.inverse_transform(raw_output, selected_indices)


