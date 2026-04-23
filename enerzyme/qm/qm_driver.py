import numpy as np
from shutil import copy, rmtree
import os
import subprocess
import queue # TODO: 
from typing import Any, Dict, Literal, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import ase.io
from ase.units import Bohr, Ha, Debye
from ase import Atoms
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ..utils import logger
from ..data.supplier import Supplier


QM_CALCULATED_TO_ASE_PROPERTY = {
    "E": "energy",
    "Fa": "forces",
    "M2": "dipole",
    "Qa": "charges",
    "Sa": "magmoms",
}

class QMDriver(ABC):
    def __init__(self, 
        supplier: Supplier, tmp_dir: str, output_dir: str, output_file: str, 
        template_input_file: str,
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True,
        n_processes: int = 1,
        timeout: Optional[float] = None,
        **kwargs
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
        output_file: str
            The name of the output file to store the results.
        keep_molden: bool
            Whether to keep the Molden files.
        keep_stdout: bool
            Whether to keep the stdout files.
        clean_tmp: bool
            Whether to clean the temporary files.
        n_processes: int
            The number of processes to use.
        '''
        self.supplier = supplier
        self.tmp_dir_base = Path(tmp_dir).absolute() / self.supplier.name / "tmp"
        self.output_dir = Path(output_dir).absolute() / self.supplier.name
        self.output_path = (self.output_dir / output_file)
        self.default_connect_args = {
            "use_lock_file": False
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_input_file = template_input_file
        self.keep_molden = keep_molden
        if keep_molden:
            os.makedirs(self.output_dir / "moldens", exist_ok=True)
        self.keep_stdout = keep_stdout
        if keep_stdout:
            os.makedirs(self.output_dir / "stdout", exist_ok=True)
        self.clean_tmp = clean_tmp
        self.n_processes = n_processes if n_processes > 0 else cpu_count()
        self.timeout = timeout

    @abstractmethod
    def make_input(self, atoms: Atoms, tmp_dir: Path) -> None:
        ...

    @abstractmethod
    def invoke_qm(self, input_file: str, atoms: Atoms, tmp_dir: Path) -> str:
        ...

    @abstractmethod
    def collect_results(self, input_file: Path, atoms: Atoms, tmp_dir: Path) -> Dict[str, Any]:
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

    def single_run(self, atoms: Atoms) -> None:
        db = connect(self.output_path, **self.default_connect_args)
        index = atoms.info["index"]
        try:
            db.get(index=index)
        except KeyError:
            system_id = db.reserve()
        else:
            logger.warning(f"System {index} already exists in {self.output_path}. Skipping...")
            return
        
        tmp_dir = Path(str(self.tmp_dir_base) + f".{index}")
        os.makedirs(tmp_dir, exist_ok=True)

        atom_info = {
            "charge": atoms.info.get("charge", 0),
            "spin": atoms.info.get("spin", 1),
            "index": index,
        }

        input_file = self.make_input(atoms, tmp_dir)
        output_file = self.invoke_qm(input_file, atoms, tmp_dir)
        try:
            result_package = self.collect_results(input_file, atoms, tmp_dir)
        except FileNotFoundError as e:
            logger.warning(f"Calculation of {input_file} failed: {e}")
            db.delete(system_id)
            return
        
        self.copy_files(output_file, result_package.get("molden_file", None))
        if self.clean_tmp:
            rmtree(tmp_dir)

        results = {}
        for qm_property, ase_property in QM_CALCULATED_TO_ASE_PROPERTY.items():
            if qm_property in result_package:
                results[ase_property] = result_package[qm_property]
        atoms.calc = SinglePointCalculator(
            atoms=atoms,
            **results
        )
        db.write(atoms, id=system_id, data=atom_info, index=index)

    def run(self):
        if self.n_processes == 1:
            for atoms in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                self.single_run(atoms)
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            with Pool(self.n_processes) as p:
                list(tqdm(
                    p.imap(self.single_run, self.supplier.suppl()),
                    desc="Running QM",
                    dynamic_ncols=True,
                    leave=False,
                    position=0
                ))
        logger.info(f"QM calculations finished. ASE LMDB saved to {self.output_path}")


class TeraChemDriver(QMDriver):
    def __init__(self, 
        terachem_args: list[str]=["terachem"],
        n_gpus: int=1,
        **kwargs
    ):
        self.terachem_args = terachem_args
        if n_gpus < 1:
            import torch
            self.n_gpus = torch.cuda.device_count()
            logger.info(f"Using all {self.n_gpus} GPUs")
        else:
            self.n_gpus = n_gpus
        super().__init__(**kwargs)

    def make_input(self, atoms: Atoms, tmp_dir: Path) -> str:
        index = atoms.info["index"]
        input_file = tmp_dir / f"{index}.in"
        Q = atoms.info.get("charge", 0)
        spinmult = atoms.info.get("spin", 1)
        
        input_lines = [
            "run gradient\n"
            f"coordinates {tmp_dir / f"{index}.xyz"}\n"
            f"charge {Q}\n"
            f"spinmult {spinmult}\n"
            f"scrdir ./scr_{index}\n"
        ]
        with open(self.template_input_file, "r") as f:
            for line in f:
                if line.startswith("run"):
                    continue
                elif line.startswith("coordinates"):
                    continue
                elif line.startswith("charge"):
                    continue
                elif line.startswith("spinmult"):
                    continue
                elif line.startswith("scrdir"):
                    continue
                elif line.strip() == "end":
                    break
                input_lines.append(line)

        input_lines.append("end\n")
        
        with open(input_file, "w") as f:
            f.writelines(input_lines)

        ase.io.write(tmp_dir / f"{index}.xyz", atoms, format="xyz")
        return input_file
    
    def invoke_qm(self, input_file: Path, atoms: Atoms, tmp_dir: Path):
        output_file = input_file.with_suffix(".out")
        current_dir = os.getcwd()
        os.chdir(tmp_dir)
        if self.n_gpus > 1:
            gpu_binding_flag = [f"-g{atoms.info['index'] % self.n_gpus}"]
        else:
            gpu_binding_flag = []
        try:
            with open(output_file, 'w') as f:
                subprocess.run(
                    self.terachem_args + gpu_binding_flag + [str(input_file)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    check=False
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"TeraChem calculation for {input_file} timed out after {self.timeout} seconds")
        finally:
            os.chdir(current_dir)
        return output_file

    def collect_results(self, input_file: Path, atoms: Atoms, tmp_dir: Path) -> Dict[
        Literal["M2", "Fa", "E", "molden_file"], Any
    ]:
        scr_dir = tmp_dir / f"scr_{input_file.stem}"
        if not (scr_dir / "results.dat").exists():
            raise FileNotFoundError(f"Results file {scr_dir / 'results.dat'} not found")
        if not (scr_dir / "grad.xyz").exists():
            raise FileNotFoundError(f"Gradients file {scr_dir / 'grad.xyz'} not found")
        with open(scr_dir / "results.dat", "r") as f:
            lines = f.readlines()
            com_line_index = -1
            dipole_line_index = -1
            for i, line in enumerate(lines):
                if line.startswith("Center of Mass (Angs):"):
                    com_line_index = i + 2
                if line.startswith("Ground state dipole moment (Debye):"):
                    dipole_line_index = i + 2
            if com_line_index == -1 or dipole_line_index == -1:
                raise FileNotFoundError(f"Center of Mass or dipole moment line not found in {scr_dir / 'results.dat'}")
            com = np.array(list(map(float, lines[com_line_index].split())))
            dipole = np.array(list(map(float, lines[dipole_line_index].split()))) * ase.units.Debye # Debye to e Angstrom
            dipole = dipole + com * atoms.info["charge"]
        with open(scr_dir / "grad.xyz", "r") as f:
            _ = f.readline()
            title = f.readline()
            energy = float(title.split()[6]) * ase.units.Ha # Ha to eV
        grad = np.loadtxt(scr_dir / "grad.xyz", skiprows=2, usecols=(1,2,3)) * ase.units.Ha / ase.units.Bohr # Ha/Bohr to eV/Angstrom
        return {"M2": dipole, "Fa": -grad, "E": energy, "molden_file": scr_dir / (input_file.stem + ".molden")}

class ORCADriver(QMDriver):
    pass

class PySCFDriver(QMDriver):
    pass

class Psi4Driver(QMDriver):
    pass
