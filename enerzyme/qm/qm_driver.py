import numpy as np
from shutil import copy, rmtree
import os
from typing import Any, Dict, Literal, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from pickle import dump
from rdkit import Chem
from ..utils import logger
from ..data import Supplier


class QMDriver(ABC):
    def __init__(self, 
        supplier: Supplier, tmp_dir: str, output_dir: str, pickle_name: str,
        bs: str, xc: str,  
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True
    ):
        self.supplier = supplier
        self.tmp_dir = Path(tmp_dir).absolute() / self.supplier.name / "tmp"
        self.output_dir = Path(output_dir).absolute() / self.supplier.name
        self.output_path = (self.output_dir / pickle_name).with_suffix(".pkl")
        self.keep_molden = keep_molden
        if keep_molden:
            os.makedirs(self.output_dir / "moldens", exist_ok=True)
        self.keep_stdout = keep_stdout
        if keep_stdout:
            os.makedirs(self.output_dir / "stdout", exist_ok=True)
        self.bs = bs
        self.xc = xc
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.clean_tmp = clean_tmp

    @abstractmethod
    def make_input(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]) -> None:
        ...

    @abstractmethod
    def invoke_qm(self, input_file: str) -> str:
        ...

    @abstractmethod
    def collect_results(self, input_file: Path, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]) -> Dict[str, Any]:
        ...

    def copy_files(self, output_file: Path, molden_file: Optional[Path]) -> None:
        if self.keep_stdout:
            if output_file.exists():
                copy(output_file, self.output_dir / "stdout")
            else:
                raise FileNotFoundError(f"Output file {output_file} not found")
        if self.keep_molden and molden_file is not None:
            if molden_file.exists():
                copy(molden_file, self.output_dir / "moldens")
            else:
                raise FileNotFoundError(f"Molden file {molden_file} not found")

    def single_run(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]):
        input_file = self.make_input(package=package)
        output_file = self.invoke_qm(input_file)
        try:
            results = self.collect_results(input_file, package)
        except FileNotFoundError as e:
            logger.warning(f"Calculation of {input_file} failed: {e}")
            results = {}
        self.copy_files(output_file, results.get("molden_file", None))
        if self.clean_tmp:
            rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        return results

    def run(self):
        datapoints = []
        for package in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
            results = self.single_run(package)
            if not results:
                continue
            datapoint = {}
            datapoint["grad"] = results["grad"]
            datapoint["dipole"] = results["dipole"]
            datapoint["energy"] = results["energy"]
            datapoint["atom_type"] = package["atom_type"]
            datapoint["coord"] = package["Ra"]
            datapoint["total_spin"] = package.get("spin", 0)
            datapoint["total_chrg"] = package["Q"]
            datapoint["index"] = package["index"]
            datapoints.append(datapoint)
        dump(datapoints, open(self.output_path, "wb"))
        logger.info(f"QM calculations finished. Pickle saved to {self.output_path}")


class TeraChemDriver(QMDriver):
    def __init__(self, 
        supplier: Supplier, tmp_dir: str, output_dir: str, pickle_name: str,
        bs: str, xc: str,  
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True,
        dftd: Optional[str] = None, 
        pcm: Optional[str] = None,
        epsilon: Optional[float] = None,
        pcm_radii_file: Optional[str] = None,
        *args, **kwargs
    ):
        super().__init__(supplier, tmp_dir, output_dir, pickle_name, bs, xc, keep_molden, keep_stdout, clean_tmp)
        self.dftd = dftd
        self.pcm = pcm
        self.epsilon = epsilon
        self.pcm_radii_file = pcm_radii_file

    def make_input(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]) -> str:
        Q = package["Q"]
        S = package.get("S", 0)
        index = package["index"]
        base_input = f'''
run gradient
coordinates {self.tmp_dir / f"{index}.xyz"}
basis {self.bs}
method {self.xc}
charge {Q}
spinmult {S + 1}
maxit 1000
scf diis+a
scrdir ./scr_{index}
'''
        if self.dftd:
            base_input += f"dftd {self.dftd}\n"
        if self.pcm:
            assert self.epsilon is not None
            assert self.pcm_radii_file is not None
            base_input += f"pcm {self.pcm}\n"
            base_input += f"epsilon {self.epsilon}\n"
            base_input += f"pcm_radii read\n"
            base_input += f"pcm_radii_file {self.pcm_radii_file}\n"
        base_input += "end\n"
        input_file = self.tmp_dir / f"{index}.in"
        with open(input_file, "w") as f:
            f.write(base_input)
        Chem.MolToXYZFile(package["mol"], self.tmp_dir / f"{index}.xyz")
        return input_file
    
    def invoke_qm(self, input_file: Path):
        output_file = input_file.with_suffix(".out")
        current_dir = os.getcwd()
        os.chdir(self.tmp_dir)
        os.system(f"terachem {input_file} > {output_file}")
        os.chdir(current_dir)
        return output_file

    def collect_results(self, input_file: Path, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]):
        scr_dir = self.tmp_dir / f"scr_{input_file.stem}"
        if not (scr_dir / "results.dat").exists():
            raise FileNotFoundError(f"Results file {scr_dir / 'results.dat'} not found")
        if not (scr_dir / "grad.xyz").exists():
            raise FileNotFoundError(f"Gradients file {scr_dir / 'grad.xyz'} not found")
        with open(scr_dir / "results.dat", "r") as f:
            lines = f.readlines()
            com = np.array(list(map(float, lines[2].split())))
            dipole = np.array(list(map(float, lines[5].split()))) * 0.20819434 # Debye to e Angstrom
            dipole = dipole + com * package["Q"]
        with open(scr_dir / "grad.xyz", "r") as f:
            _ = f.readline()
            title = f.readline()
            energy = float(title.split()[6])
        grad = np.loadtxt(scr_dir / "grad.xyz", skiprows=2, usecols=(1,2,3)) / 0.5291772108 # Bohr to Angstrom
        return {"dipole": dipole, "grad": grad, "energy": energy, "molden_file": scr_dir / (input_file.stem + ".molden")}

class ORCADriver(QMDriver):
    pass

class PySCFDriver(QMDriver):
    pass

class Psi4Driver(QMDriver):
    pass
