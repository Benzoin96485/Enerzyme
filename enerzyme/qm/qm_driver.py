import numpy as np
from shutil import copy, rmtree
import os
import subprocess
from pickle import dump, load
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

# Geometry / charge fields always present on ASE LMDB rows written by annotate.
_ASELMDB_SCHEMA_GEOMETRY = ("Ra", "Za", "N", "Q", "S")

# Optional annotate pickle rename: standard Enerzyme name → custom key in the .pkl dict.
# Values stay standard (E/Fa in Ha / Ha·Å⁻¹, M2 in e·Å, Za atomic numbers, Q, S=2S+1−1).
# Enerzymette AL historically expects the names below (and Fa stored as ∇E under ``grad``).
ENERZYMETTE_PICKLE_FIELDS: Dict[str, str] = {
    "E": "energy",
    "Fa": "grad",
    "M2": "dipole",
    "Ra": "coord",
    "Za": "atom_type",
    "Q": "total_chrg",
    "S": "total_spin",
}

_PICKLE_SUFFIXES = {".pkl", ".pickle"}


# YAML keys once accepted by TeraChemDriver; now live in ``template_input_file``.
_LEGACY_TC_TEMPLATE_KEYS = frozenset({
    "bs", "xc", "pcm", "dftd", "epsilon", "pcm_radii_file",
    "scf_method", "scf_maxit", "scf_guess",
})


def _resolve_keep_stdout(keep_stdout: bool, kwargs: Dict[str, Any]) -> bool:
    """Accept legacy ``keep_output`` from YAML kwargs as an alias for ``keep_stdout``."""
    if "keep_output" not in kwargs:
        return keep_stdout
    keep_output = bool(kwargs.pop("keep_output"))
    logger.warning(
        "QMDriver option `keep_output` is deprecated; use `keep_stdout` instead. "
        f"Interpreting keep_output={keep_output} as keep_stdout."
    )
    # Prefer explicit keep_stdout when it is True; otherwise honor the legacy key
    # (covers default keep_stdout=False + keep_output=True from old YAML).
    if keep_stdout:
        if keep_stdout != keep_output:
            logger.warning(
                f"Both keep_stdout={keep_stdout} and keep_output={keep_output} were set; "
                "using keep_stdout."
            )
        return keep_stdout
    return keep_output


def _warn_unused_qm_kwargs(kwargs: Dict[str, Any]) -> None:
    """Pop / warn on leftover YAML keys so silent no-ops are visible."""
    # Selected by annotate.py before constructing the concrete driver.
    kwargs.pop("engine", None)
    legacy = sorted(k for k in kwargs if k in _LEGACY_TC_TEMPLATE_KEYS)
    for key in legacy:
        kwargs.pop(key)
    if legacy:
        logger.warning(
            "Ignoring legacy QMDriver keys now owned by `template_input_file`: "
            f"{legacy}. Put basis / XC / PCM / DFTD / SCF settings in the template."
        )
    if kwargs:
        leftover = sorted(kwargs)
        for key in leftover:
            kwargs.pop(key)
        logger.warning(f"Ignoring unrecognized QMDriver options: {leftover}")


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


def _aselmdb_row_is_complete(row) -> bool:
    """True when a reserved/written ASE DB row has calculator energy (usable resume hit)."""
    try:
        atoms = row.toatoms()
    except Exception:
        return False
    if atoms.calc is None:
        return False
    try:
        energy = atoms.calc.results.get("energy", None)
    except Exception:
        return False
    return energy is not None


def _build_standard_pickle_datapoint(
    atoms: Atoms,
    result_package: Dict[str, Any],
) -> Dict[str, Any]:
    """Driver results (E/Fa in eV) → standard Enerzyme pickle fields (E/Fa in Ha / Ha·Å⁻¹)."""
    energy_ev = float(result_package["E"])
    forces_ev = np.asarray(result_package["Fa"], dtype=float)
    return {
        "E": energy_ev / Ha,
        "Fa": forces_ev / Ha,
        "M2": np.asarray(result_package.get("M2", np.zeros(3)), dtype=float),
        "Ra": np.asarray(atoms.get_positions(), dtype=float),
        "Za": np.asarray(atoms.get_atomic_numbers(), dtype=int),
        "N": len(atoms),
        "Q": int(atoms.info.get("charge", 0)),
        "S": int(atoms.info.get("spin", 1)) - 1,
        "index": int(atoms.info["index"]),
    }


def _apply_pickle_fields(
    datapoint: Dict[str, Any],
    pickle_fields: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    """Rename standard keys to custom names. ``None`` / missing map → identity (all standard keys).

    Special case for Enerzymette: when ``Fa`` is renamed to ``grad``, store ``−Fa`` (∇E) so
    existing ``negative_gradient: true`` train YAMLs keep working.
    """
    if not pickle_fields:
        return datapoint
    out: Dict[str, Any] = {}
    for std_key, value in datapoint.items():
        if std_key == "index":
            out["index"] = value
            continue
        if std_key not in pickle_fields:
            continue
        custom = pickle_fields[std_key]
        name = std_key if custom is None else str(custom)
        if std_key == "Fa" and name == "grad":
            value = -np.asarray(value, dtype=float)
        out[name] = value
    return out


def _result_package_to_pickle_datapoint(
    atoms: Atoms,
    result_package: Dict[str, Any],
    pickle_fields: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    return _apply_pickle_fields(
        _build_standard_pickle_datapoint(atoms, result_package),
        pickle_fields,
    )


class QMDriver(ABC):
    def __init__(
        self,
        supplier: Supplier,
        tmp_dir: str,
        output_dir: str,
        template_input_file: str,
        output_file: Optional[str] = None,
        pickle_name: Optional[str] = None,
        pickle_fields: Optional[Dict[str, Optional[str]]] = None,
        keep_molden: bool = False,
        keep_stdout: bool = False,
        clean_tmp: bool = True,
        n_processes: int = 1,
        timeout: Optional[float] = None,
        dump_single_run: bool = True,
        **kwargs,
    ):
        '''
        Base class for QM drivers.

        Output format (mutually exclusive):
        - pickle when ``pickle_name`` is set, or ``output_file`` ends with ``.pkl`` / ``.pickle``
        - ASE DB (default) otherwise (``output_file`` defaults to ``dataset.aselmdb``)

        Pickle records use standard Enerzyme names (``E``, ``Fa``, ``M2``, ``Ra``, ``Za``, …)
        unless ``pickle_fields`` renames them (e.g. :data:`ENERZYMETTE_PICKLE_FIELDS`).

        Resume / skip completed structures:
        - pickle: when ``dump_single_run`` is True (default), each success is cached under
          ``output_dir/<supplier>/single_run/<index>.pkl`` and reloaded on later runs
        - aselmdb: rows keyed by structure ``index``; completed rows are skipped (incomplete
          reservations are deleted and retried). ``dump_single_run`` is ignored for ASE DB.
        '''
        self.supplier = supplier
        self.tmp_dir_base = Path(tmp_dir).absolute() / self.supplier.name / "tmp"
        self.output_dir = Path(output_dir).absolute() / self.supplier.name
        self.output_format, out_name = _resolve_annotate_output(output_file, pickle_name)
        self.output_path = self.output_dir / out_name
        self.pickle_fields = pickle_fields
        self.dump_single_run = bool(dump_single_run)
        self.default_connect_args = {
            "use_lock_file": False
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_input_file = template_input_file
        self.keep_molden = keep_molden
        if keep_molden:
            os.makedirs(self.output_dir / "moldens", exist_ok=True)
        self.keep_stdout = _resolve_keep_stdout(keep_stdout, kwargs)
        if self.keep_stdout:
            os.makedirs(self.output_dir / "stdout", exist_ok=True)
        self.clean_tmp = clean_tmp
        self.n_processes = n_processes if n_processes > 0 else cpu_count()
        self.timeout = timeout
        if self.output_format == "pickle" and self.dump_single_run:
            os.makedirs(self.single_run_dir, exist_ok=True)
        elif self.output_format == "aselmdb" and not self.dump_single_run:
            logger.info(
                "dump_single_run=False has no effect for aselmdb; "
                "completed rows keyed by structure index are always skipped on resume."
            )
        _warn_unused_qm_kwargs(kwargs)
        logger.info(f"Annotate output format={self.output_format}, path={self.output_path}")
        if self.output_format == "pickle":
            logger.info(f"dump_single_run={self.dump_single_run} (per-structure resume cache)")
            if self.pickle_fields:
                logger.info(f"Pickle field map: {self.pickle_fields}")

    @property
    def single_run_dir(self) -> Path:
        return self.output_dir / "single_run"

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

    def dump_pickle_single_run(self, datapoint: Dict[str, Any]) -> None:
        """Cache one pickle datapoint for resume (``single_run/<index>.pkl``)."""
        index = int(datapoint["index"])
        path = self.single_run_dir / f"{index}.pkl"
        with open(path, "wb") as f:
            dump(datapoint, f)

    def load_pickle_single_run(self, index: int) -> Optional[Dict[str, Any]]:
        """Load a cached pickle datapoint if ``dump_single_run`` left one on disk."""
        if not self.dump_single_run:
            return None
        path = self.single_run_dir / f"{index}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            datapoint = load(f)
        logger.info(
            f"Single run results for {index} already exist. Loading from {path}"
        )
        return datapoint

    def aselmdb_schema_properties(self) -> List[str]:
        """Standard names written into ASE LMDB metadata for Datahub discovery."""
        # Geometry/charge always; E/Fa/M2 from successful QM (Qa/Sa only if a driver adds them).
        return list(_ASELMDB_SCHEMA_GEOMETRY) + ["E", "Fa", "M2"]

    def _ensure_aselmdb_schema(self) -> None:
        """Persist property schema so readers need not rely on the first row alone."""
        from ..data.datahub import ASELMDB_METADATA_PROPERTIES_KEY

        props = self.aselmdb_schema_properties()
        db = connect(self.output_path, **self.default_connect_args)
        meta = dict(db.metadata or {})
        if meta.get(ASELMDB_METADATA_PROPERTIES_KEY) == props:
            return
        meta[ASELMDB_METADATA_PROPERTIES_KEY] = props
        db.metadata = meta
        logger.info(
            f"Wrote ASE LMDB schema {ASELMDB_METADATA_PROPERTIES_KEY}={props} "
            f"to {self.output_path}"
        )

    def _claim_aselmdb_row_id(self, db, index: int) -> Optional[int]:
        """Claim or skip a structure ``index`` in the ASE DB for resume-safe writes.

        Returns an ASE primary key to overwrite, or ``None`` when a completed row
        already exists for ``index`` (skip QM). Incomplete reserved rows are deleted
        and re-claimed so crashed jobs can resume.
        """
        existing = list(db.select(index=index))
        if existing:
            complete = [row for row in existing if _aselmdb_row_is_complete(row)]
            if complete:
                logger.info(
                    f"System {index} already completed in {self.output_path}. Skipping..."
                )
                return None
            # Orphaned reservations / incomplete writes: free the index and retry.
            ids = [row.id for row in existing]
            logger.warning(
                f"System {index} has incomplete ASE LMDB row(s) {ids}; "
                "deleting and recomputing."
            )
            db.delete(ids)

        system_id = db.reserve(index=index)
        if system_id is None:
            # Race with another writer that finished between select and reserve.
            logger.info(
                f"System {index} already exists in {self.output_path}. Skipping..."
            )
            return None
        return system_id

    def _run_qm(self, atoms: Atoms) -> Optional[Dict[str, Any]]:
        index = atoms.info["index"]
        tmp_dir = Path(str(self.tmp_dir_base) + f".{index}")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            input_file = self.make_input(atoms, tmp_dir)
            output_file = self.invoke_qm(input_file, atoms, tmp_dir)
            result_package = self.collect_results(input_file, atoms, tmp_dir)
            self.copy_files(output_file, result_package.get("molden_file", None))
        except Exception as e:
            logger.warning(f"Calculation of structure {index} failed: {e}")
            if self.clean_tmp and tmp_dir.exists():
                rmtree(tmp_dir)
            return None
        if self.clean_tmp:
            rmtree(tmp_dir)
        return result_package

    def single_run_aselmdb(self, atoms: Atoms) -> None:
        db = connect(self.output_path, **self.default_connect_args)
        index = int(atoms.info["index"])
        # Prefer a parent-assigned id (multiprocess); otherwise claim by structure index.
        # Bare reserve() is wrong: with no key-value pairs ASE treats any existing row as a
        # hit and returns None after the first insert.
        if "aselmdb_row_id" in atoms.info:
            system_id = atoms.info["aselmdb_row_id"]
            if system_id is None:
                return
        else:
            system_id = self._claim_aselmdb_row_id(db, index)
            if system_id is None:
                return

        # Always release the reservation unless the row is successfully written;
        # otherwise orphaned reserved rows block later runs of the same index.
        wrote = False
        try:
            result_package = self._run_qm(atoms)
            if result_package is None:
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
            wrote = True
        except Exception as e:
            logger.warning(
                f"Failed to store ASE LMDB row for structure {index} (id={system_id}): {e}"
            )
        finally:
            if not wrote:
                try:
                    db.delete([system_id])
                except Exception as e:
                    logger.warning(
                        f"Could not delete reserved ASE LMDB id {system_id} "
                        f"for structure {index}: {e}"
                    )

    def single_run_pickle(self, atoms: Atoms) -> Optional[Dict[str, Any]]:
        index = int(atoms.info["index"])
        cached = self.load_pickle_single_run(index)
        if cached is not None:
            return cached
        result_package = self._run_qm(atoms)
        if result_package is None:
            return None
        datapoint = _result_package_to_pickle_datapoint(
            atoms, result_package, pickle_fields=self.pickle_fields
        )
        if self.dump_single_run:
            self.dump_pickle_single_run(datapoint)
        return datapoint

    def single_run(self, atoms: Atoms):
        if self.output_format == "pickle":
            return self.single_run_pickle(atoms)
        return self.single_run_aselmdb(atoms)

    def run(self):
        if self.output_format == "pickle":
            self._run_pickle()
        else:
            self._run_aselmdb()

    def _reserve_aselmdb_ids(self, atoms_list: List[Atoms]) -> List[Atoms]:
        """Serially claim unique ASE row ids keyed by structure ``index``.

        ASE LMDB only inserts with auto-increment ids; concurrent ``reserve()`` /
        ``nextid`` updates are unsafe without a working lock file (aselmdb passes
        a ``Path`` so ASE's lock is never enabled). Parent-side reservation gives
        each structure a distinct primary key; workers only overwrite that row.
        Completed indices are skipped for resume.
        """
        db = connect(self.output_path, **self.default_connect_args)
        to_run: List[Atoms] = []
        for atoms in atoms_list:
            index = int(atoms.info["index"])
            system_id = self._claim_aselmdb_row_id(db, index)
            if system_id is None:
                continue
            atoms.info["aselmdb_row_id"] = system_id
            to_run.append(atoms)
        return to_run

    def _pool_imap(self, worker, items, *, total: Optional[int] = None):
        """``Pool.imap`` on a bound method pickles ``self``; drop Supplier first.

        Workers only need QM I/O state. Keeping ``supplier`` (esp. RDKit
        ``SDMolSupplier``) in the pickled driver fails for SDF inputs.
        """
        supplier = self.supplier
        self.supplier = None
        try:
            with Pool(self.n_processes) as p:
                return list(tqdm(
                    p.imap(worker, items),
                    desc="Running QM",
                    dynamic_ncols=True,
                    leave=False,
                    position=0,
                    total=total,
                ))
        finally:
            self.supplier = supplier

    def _run_aselmdb(self) -> None:
        self._ensure_aselmdb_schema()
        if self.n_processes == 1:
            for atoms in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                self.single_run_aselmdb(atoms)
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            atoms_list = self._reserve_aselmdb_ids(list(self.supplier.suppl()))
            self._pool_imap(
                self.single_run_aselmdb,
                atoms_list,
                total=len(atoms_list),
            )
        logger.info(f"QM calculations finished. ASE LMDB saved to {self.output_path}")

    def _run_pickle(self) -> None:
        if self.n_processes == 1:
            result_packages: List[Optional[Dict[str, Any]]] = []
            for atoms in tqdm(self.supplier.suppl(), desc="Running QM", dynamic_ncols=True, leave=False, position=0):
                result_packages.append(self.single_run_pickle(atoms))
        else:
            logger.info(f"Running QM calculations with {self.n_processes} processes")
            # Materialize before Pool: generator + detached supplier would be empty.
            atoms_list = list(self.supplier.suppl())
            result_packages = self._pool_imap(
                self.single_run_pickle,
                atoms_list,
                total=len(atoms_list),
            )
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
        template_path = Path(self.template_input_file).resolve()
        template_dir = template_path.parent

        input_lines = [
            "run gradient\n"
            f"coordinates {tmp_dir / f'{index}.xyz'}\n"
            f"charge {Q}\n"
            f"spinmult {spinmult}\n"
            f"scrdir ./scr_{index}\n"
        ]
        with open(template_path, "r") as f:
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
                elif line.startswith("pcm_radii_file"):
                    # Resolve relative to the template; copy into tmp so TeraChem
                    # can use a basename path (no host absolute paths required).
                    parts = line.split()
                    if len(parts) >= 2:
                        radii_src = Path(parts[1])
                        if not radii_src.is_absolute():
                            radii_src = template_dir / radii_src
                        radii_src = radii_src.resolve()
                        radii_dst = tmp_dir / radii_src.name
                        if not radii_dst.exists():
                            copy(radii_src, radii_dst)
                        input_lines.append(f"pcm_radii_file {radii_src.name}\n")
                    else:
                        input_lines.append(line)
                    continue
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
