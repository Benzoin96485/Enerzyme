import numpy as np
from shutil import copy, rmtree
import os
from functools import partial
from typing import Any, Dict, Literal, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from pickle import dump
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from ..utils import logger
from ..data.supplier import Supplier


class QMDriver(ABC):
    def __init__(self, 
        supplier: Supplier, tmp_dir: str, output_dir: str, pickle_name: str,
        bs: str, xc: str,  
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True,
        n_processes: int = 1,
        dump_single_run: bool = True
    ):
        '''
        Base class for QM drivers.

        Params:
        -------
        supplier: Supplier
            The supplier of molecular data.
        tmp_dir: str
            The directory to store temporary files.
        output_dir: str
            The directory to store output files.
        pickle_name: str
            The name of the pickle file to store the results.
        bs: str
            The basis set to use.
        xc: str
            The exchange-correlation functional to use.
        keep_molden: bool
            Whether to keep the Molden files.
        keep_stdout: bool
            Whether to keep the stdout files.
        clean_tmp: bool
            Whether to clean the temporary files.
        n_processes: int
            The number of processes to use.
        dump_single_run: bool
            Whether to dump the single run results.
        '''
        self.supplier = supplier
        self.tmp_dir_base = Path(tmp_dir).absolute() / self.supplier.name / "tmp"
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
        self.clean_tmp = clean_tmp
        self.dump_single_run = dump_single_run
        if self.dump_single_run:
            os.makedirs(self.output_dir / "single_run", exist_ok=True)
        self.n_processes = n_processes if n_processes > 0 else cpu_count()

    @abstractmethod
    def make_input(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any], tmp_dir: Path) -> None:
        ...

    @abstractmethod
    def invoke_qm(self, input_file: str, tmp_dir: Path) -> str:
        ...

    @abstractmethod
    def collect_results(self, input_file: Path, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any], tmp_dir: Path) -> Dict[str, Any]:
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

    def dump_results(self, result_package: Dict[str, Any]) -> None:
        with open(self.output_dir / "single_run" / f"{result_package['index']}.pkl", "wb") as f:
            dump(result_package, f)

    def single_run(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any]):
        tmp_dir = Path(str(self.tmp_dir_base) + f".{package['index']}")
        os.makedirs(tmp_dir, exist_ok=True)
        input_file = self.make_input(package=package, tmp_dir=tmp_dir)
        output_file = self.invoke_qm(input_file, tmp_dir)
        try:
            result_package = self.collect_results(input_file, package, tmp_dir)
        except FileNotFoundError as e:
            logger.warning(f"Calculation of {input_file} failed: {e}")
            result_package = {}
            return result_package
        
        self.copy_files(output_file, result_package.get("molden_file", None))
        if self.clean_tmp:
            rmtree(tmp_dir)
        result_package["atom_type"] = package["atom_type"]
        result_package["coord"] = package["Ra"]
        result_package["total_spin"] = package.get("spin", 0)
        result_package["total_chrg"] = package["Q"]
        result_package["index"] = package["index"]
        if self.dump_single_run:
            self.dump_results(result_package)
        return result_package

    def run(self):
        datapoints = []
        if self.n_processes == 1:
            result_packages = []
            for package in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                result_package = self.single_run(package)
                result_packages.append(result_package)
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            with Pool(self.n_processes) as p:
                result_packages = list(tqdm(
                    p.imap(self.single_run, self.supplier.suppl()),
                    desc="Running QM",
                    dynamic_ncols=True,
                    leave=False,
                    position=0
                ))
        for result_package in result_packages:
            if not result_package:
                continue
            datapoint = {k: result_package[k] for k in [
                "energy", "grad", "dipole", "index", "atom_type", "coord", "total_spin", "total_chrg"
            ]}
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
        n_processes: int = 1,
        dump_single_run: bool = True,
        *args, **kwargs
    ):
        super().__init__(supplier, tmp_dir, output_dir, pickle_name, bs, xc, keep_molden, keep_stdout, clean_tmp, n_processes, dump_single_run)
        self.dftd = dftd
        self.pcm = pcm
        self.epsilon = epsilon
        self.pcm_radii_file = pcm_radii_file

    def make_input(self, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any], tmp_dir: Path) -> str:
        Q = package["Q"]
        S = package.get("S", 0)
        index = package["index"]
        base_input = f'''
run gradient
coordinates {tmp_dir / f"{index}.xyz"}
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
        input_file = tmp_dir / f"{index}.in"
        with open(input_file, "w") as f:
            f.write(base_input)
        if "mol" in package:
            Chem.MolToXYZFile(package["mol"], tmp_dir / f"{index}.xyz")
        elif "Ra" in package:
            with open(tmp_dir / f"{index}.xyz", "w") as f:
                f.write(f"{len(package['Ra'])}\n")
                f.write("\n")
                for i, (x, y, z) in enumerate(package["Ra"]):
                    f.write(f"{package['atom_type'][i]} {x} {y} {z}\n")
        else:
            raise ValueError(f"Invalid package: {package}")
        return input_file
    
    def invoke_qm(self, input_file: Path, tmp_dir: Path):
        output_file = input_file.with_suffix(".out")
        current_dir = os.getcwd()
        os.chdir(tmp_dir)
        os.system(f"terachem {input_file} > {output_file}")
        os.chdir(current_dir)
        return output_file

    def collect_results(self, input_file: Path, package: Dict[Literal["index", "atom_type", "Ra", "Q", "mol"], Any], tmp_dir: Path):
        scr_dir = tmp_dir / f"scr_{input_file.stem}"
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
