import numpy as np
from shutil import copy, rmtree
import os
import subprocess
from pickle import dump
from typing import Any, Dict, Literal, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import ase.io
import ase.units
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

_PICKLE_SUFFIXES = {".pkl", ".pickle"}


def _resolve_annotate_output(
    output_file: Optional[str],
    pickle_name: Optional[str],
) -> tuple[Literal["pickle", "aselmdb"], str]:
    """Pickle if ``pickle_name`` is set or ``output_file`` looks like a pickle; else ASE DB."""
    if pickle_name:
        name = str(pickle_name)
        if Path(name).suffix.lower() not in _PICKLE_SUFFIXES:
            name = f"{Path(name).stem}.pkl"
        return "pickle", name
    if output_file and Path(output_file).suffix.lower() in _PICKLE_SUFFIXES:
        return "pickle", str(output_file)
    return "aselmdb", (output_file or "dataset.aselmdb")


def _result_package_to_pickle_datapoint(
    atoms: Atoms,
    result_package: Dict[str, Any],
) -> Dict[str, Any]:
    """Map driver results (E/Fa in eV) to legacy Enerzymette pickle fields (energy/grad in Ha)."""
    energy_ev = float(result_package["E"])
    forces_ev = np.asarray(result_package["Fa"], dtype=float)
    # Campaign / devel pickle: energy [Ha], grad [Ha/Å] with train ``negative_gradient: true``
    return {
        "energy": energy_ev / Ha,
        "grad": (-forces_ev) / Ha,
        "dipole": np.asarray(result_package.get("M2", np.zeros(3)), dtype=float),
        "index": int(atoms.info["index"]),
        "atom_type": np.asarray(atoms.get_chemical_symbols()),
        "coord": np.asarray(atoms.get_positions(), dtype=float),
        "total_spin": int(atoms.info.get("spin", 1)) - 1,
        "total_chrg": int(atoms.info.get("charge", 0)),
    }


class QMDriver(ABC):
    def __init__(
        self,
        supplier: Supplier,
        tmp_dir: str,
        output_dir: str,
        template_input_file: str,
        output_file: Optional[str] = None,
        pickle_name: Optional[str] = None,
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True,
        n_processes: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        '''
        Base class for QM drivers.

        Output format (mutually exclusive):
        - pickle when ``pickle_name`` is set, or ``output_file`` ends with ``.pkl`` / ``.pickle``
        - ASE DB (default) otherwise (``output_file`` defaults to ``dataset.aselmdb``)
        '''
        self.supplier = supplier
        self.tmp_dir_base = Path(tmp_dir).absolute() / self.supplier.name / "tmp"
        self.output_dir = Path(output_dir).absolute() / self.supplier.name
        self.output_format, out_name = _resolve_annotate_output(output_file, pickle_name)
        self.output_path = self.output_dir / out_name
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
        logger.info(f"Annotate output format={self.output_format}, path={self.output_path}")

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

    def _run_qm(self, atoms: Atoms) -> Optional[Dict[str, Any]]:
        index = atoms.info["index"]
        tmp_dir = Path(str(self.tmp_dir_base) + f".{index}")
        os.makedirs(tmp_dir, exist_ok=True)
        input_file = self.make_input(atoms, tmp_dir)
        output_file = self.invoke_qm(input_file, atoms, tmp_dir)
        try:
            result_package = self.collect_results(input_file, atoms, tmp_dir)
        except FileNotFoundError as e:
            logger.warning(f"Calculation of {input_file} failed: {e}")
            if self.clean_tmp and tmp_dir.exists():
                rmtree(tmp_dir)
            return None
        self.copy_files(output_file, result_package.get("molden_file", None))
        if self.clean_tmp:
            rmtree(tmp_dir)
        return result_package

    def single_run_aselmdb(self, atoms: Atoms) -> None:
        db = connect(self.output_path, **self.default_connect_args)
        index = atoms.info["index"]
        try:
            db.get(index=index)
        except KeyError:
            system_id = db.reserve()
        else:
            logger.warning(f"System {index} already exists in {self.output_path}. Skipping...")
            return

        result_package = self._run_qm(atoms)
        if result_package is None:
            db.delete(system_id)
            return

        atom_info = {
            "charge": atoms.info.get("charge", 0),
            "spin": atoms.info.get("spin", 1),
            "index": index,
        }
        results = {}
        for qm_property, ase_property in QM_CALCULATED_TO_ASE_PROPERTY.items():
            if qm_property in result_package:
                results[ase_property] = result_package[qm_property]
        atoms.calc = SinglePointCalculator(atoms=atoms, **results)
        db.write(atoms, id=system_id, data=atom_info, index=index)

    def single_run_pickle(self, atoms: Atoms) -> Optional[Dict[str, Any]]:
        result_package = self._run_qm(atoms)
        if result_package is None:
            return None
        return _result_package_to_pickle_datapoint(atoms, result_package)

    def single_run(self, atoms: Atoms):
        if self.output_format == "pickle":
            return self.single_run_pickle(atoms)
        return self.single_run_aselmdb(atoms)

    def run(self):
        if self.output_format == "pickle":
            self._run_pickle()
        else:
            self._run_aselmdb()

    def _run_aselmdb(self) -> None:
        if self.n_processes == 1:
            for atoms in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                self.single_run_aselmdb(atoms)
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            with Pool(self.n_processes) as p:
                list(tqdm(
                    p.imap(self.single_run_aselmdb, self.supplier.suppl()),
                    desc="Running QM",
                    dynamic_ncols=True,
                    leave=False,
                    position=0
                ))
        logger.info(f"QM calculations finished. ASE LMDB saved to {self.output_path}")

    def _run_pickle(self) -> None:
        if self.n_processes == 1:
            result_packages: List[Optional[Dict[str, Any]]] = []
            for atoms in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                result_packages.append(self.single_run_pickle(atoms))
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            with Pool(self.n_processes) as p:
                result_packages = list(tqdm(
                    p.imap(self.single_run_pickle, self.supplier.suppl()),
                    desc="Running QM",
                    dynamic_ncols=True,
                    leave=False,
                    position=0
                ))
        datapoints = [r for r in result_packages if r]
        with open(self.output_path, "wb") as f:
            dump(datapoints, f)
        logger.info(f"QM calculations finished. Pickle saved to {self.output_path} ({len(datapoints)} structures)")


class TeraChemDriver(QMDriver):
    def __init__(
        self,
        terachem_args: list[str] = ["terachem"],
        n_gpus: int = 1,
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
            f"coordinates {tmp_dir / f'{index}.xyz'}\n"
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
            dipole = np.array(list(map(float, lines[dipole_line_index].split()))) * Debye  # Debye to e Angstrom
            dipole = dipole + com * atoms.info["charge"]
        with open(scr_dir / "grad.xyz", "r") as f:
            _ = f.readline()
            title = f.readline()
            energy = float(title.split()[6]) * Ha  # Ha to eV
        grad = np.loadtxt(scr_dir / "grad.xyz", skiprows=2, usecols=(1, 2, 3)) * Ha / Bohr  # Ha/Bohr to eV/Angstrom
        return {"M2": dipole, "Fa": -grad, "E": energy, "molden_file": scr_dir / (input_file.stem + ".molden")}


class ORCADriver(QMDriver):
    pass


class PySCFDriver(QMDriver):
    pass


class Psi4Driver(QMDriver):
    pass
