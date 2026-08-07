import os, pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from functools import partial
from multiprocessing import get_context
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import ase.units
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
        if "Fa" in new_output:
            new_output["Fa"][idx] = -new_output["Fa"][idx]



class EnergyUnitConversionTransform(BaseTransform):
    def __init__(self, old_unit: str, new_unit: str):
        super().__init__(major_key="E")
        self.transform_type = "scale"
        self.conversion_factor = getattr(ase.units, old_unit) / getattr(ase.units, new_unit)

    def transform(self, new_input):
        if "E" in new_input:
            for i in range(len(new_input["E"])):
                new_input["E"][i] *= self.conversion_factor
        
        if "Fa" in new_input:
            for i in range(len(new_input["Fa"])):
                new_input["Fa"][i] *= self.conversion_factor

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        if "E" in new_output:
            new_output["E"][idx] /= self.conversion_factor

        if "Fa" in new_output:
            new_output["Fa"][idx] /= self.conversion_factor


def wants_uniform_qs_init(transform_args: Optional[Dict]) -> bool:
    """True when ``uniform_qs_init`` is enabled in a transforms dict.

    Accepts either ``global_transforms`` or ``preprocessings`` (or any dict that
    may contain the YAML hook).
    """
    if not transform_args or "uniform_qs_init" not in transform_args:
        return False
    v = transform_args["uniform_qs_init"]
    if v is False or v is None:
        return False
    if isinstance(v, dict) and v.get("enabled") is False:
        return False
    return True



def wants_xtb_qs_prior(transform_args: Optional[Dict]) -> bool:
    """True when ``xtb_qs_prior`` is enabled in a transforms dict."""
    if not transform_args or "xtb_qs_prior" not in transform_args:
        return False
    v = transform_args["xtb_qs_prior"]
    if v is False or v is None:
        return False
    if isinstance(v, dict) and v.get("enabled") is False:
        return False
    return True


def wants_pyscf_nao_qs_prior(transform_args: Optional[Dict]) -> bool:
    """True when ``pyscf_nao_qs_prior`` is enabled in a transforms dict."""
    if not transform_args or "pyscf_nao_qs_prior" not in transform_args:
        return False
    v = transform_args["pyscf_nao_qs_prior"]
    if v is False or v is None:
        return False
    if isinstance(v, dict) and v.get("enabled") is False:
        return False
    return True


def wants_qs_delta(transform_args: Optional[Dict]) -> bool:
    """True when ``qs_delta`` is enabled in a transforms dict."""
    if not transform_args or "qs_delta" not in transform_args:
        return False
    v = transform_args["qs_delta"]
    if v is False or v is None:
        return False
    if isinstance(v, dict) and v.get("enabled") is False:
        return False
    return True


def _conserve_atomic_totals(
    q_atom: np.ndarray,
    s_atom: np.ndarray,
    n_atoms: int,
    q_tot: float,
    s_tot: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_atoms <= 0:
        return q_atom, s_atom
    q = np.asarray(q_atom[:n_atoms], dtype=np.float64).copy()
    s = np.asarray(s_atom[:n_atoms], dtype=np.float64).copy()
    dq = float(q_tot) - float(q.sum())
    ds = float(s_tot) - float(s.sum())
    inv = 1.0 / n_atoms
    q += dq * inv
    s += ds * inv
    return q, s


def _multiplicity_from_enerzyme_S(s_val: float) -> int:
    """Dataset ``S`` is (multiplicity - 1); ASE/tblite use multiplicity in ``atoms.info['spin']``."""
    return max(1, int(round(float(s_val) + 1.0)))


def _pyscf_nao_qs_prior_worker(
    task: Tuple[int, np.ndarray, np.ndarray, int, float, float, int, Callable[..., Tuple[Any, Any]]]
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Run one PySCF NAO prior task without touching parent HDF5 handles."""
    from ase import Atoms

    i, z_i, r_i, n_atoms, q_tot, s_tot, max_scf_iter, atomic_qs_fn = task
    if n_atoms <= 0:
        return i, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), None
    try:
        atoms = Atoms(numbers=np.asarray(z_i, dtype=int), positions=np.asarray(r_i, dtype=float))
        atoms.info["charge"] = float(q_tot)
        atoms.info["spin"] = _multiplicity_from_enerzyme_S(float(s_tot))
        qa, sa = atomic_qs_fn(atoms, int(max_scf_iter))
        qa = np.asarray(qa, dtype=np.float64).reshape(-1)
        sa = np.asarray(sa, dtype=np.float64).reshape(-1)
        if qa.size != n_atoms or sa.size != n_atoms:
            raise ValueError(f"expected length {n_atoms}, got Qa {qa.size}, Sa {sa.size}")
        qa, sa = _conserve_atomic_totals(qa, sa, n_atoms, q_tot, s_tot)
        return i, qa, sa, None
    except Exception as exc:
        return i, None, None, repr(exc)


class XTBQSPriorTransform(BaseTransform):
    """xTB Mulliken-style per-atom Q/S prior into ``Q_init_a`` / ``S_init_a`` (HDF5 cache only)."""

    POPULATED_KEYS = frozenset({"Q_init_a", "S_init_a"})

    def __init__(
        self,
        q_key: str = "Q",
        s_key: str = "S",
        n_key: str = "N",
        out_q: str = "Q_init_a",
        out_s: str = "S_init_a",
        max_scf_iter: int = 1,
        atomic_qs_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("module_path", None)
        if kwargs:
            logger.warning("XTBQSPriorTransform: ignoring unrecognized keys %s", sorted(kwargs))
        super().__init__(major_key=out_q)
        self.q_key = q_key
        self.s_key = s_key
        self.n_key = n_key
        self.out_q = out_q
        self.out_s = out_s
        self.max_scf_iter = int(max_scf_iter)
        if atomic_qs_fn is not None:
            self._atomic_qs_fn = atomic_qs_fn
        else:
            from ..qm.xtb_population.deps import check_xtbml_dependencies
            from ..qm.xtb_population.atomic_populations import atomic_Q_and_S_from_xtbml

            check_xtbml_dependencies()
            self._atomic_qs_fn = atomic_Q_and_S_from_xtbml
        self.transform_type = "feature"

    def transform(self, new_input: Dict[str, Iterable]) -> None:
        from ase import Atoms

        required = ("Ra", "Za", self.n_key, self.q_key, self.s_key)
        for rk in required:
            if rk not in new_input:
                raise KeyError(f"xtb_qs_prior: missing required dataset '{rk}'")
        za = np.asarray(new_input["Za"])
        # Compressed datasets may store a single stoichiometry row (1-D or 2-D with one frame).
        if za.ndim == 1:
            za = za.reshape(1, -1)
        elif za.ndim != 2:
            raise ValueError(f"xtb_qs_prior: Za must be 1-D or 2-D, got shape {za.shape}")
        # Ra is never compressed; N/Q/S may be shared across frames when compressed.
        n_frames = len(new_input["Ra"])
        max_n = int(za.shape[1])
        n_arr = np.asarray(new_input[self.n_key][:], dtype=np.int64).ravel()
        q_flat = np.asarray(new_input[self.q_key][:], dtype=np.float64).ravel()
        s_flat = np.asarray(new_input[self.s_key][:], dtype=np.float64).ravel()
        q_len = max(len(q_flat), 1)
        s_len = max(len(s_flat), 1)
        q_block = np.zeros((n_frames, max_n), dtype=np.float64)
        s_block = np.zeros((n_frames, max_n), dtype=np.float64)
        failures: List[int] = []
        for i in tqdm(range(n_frames), desc="xtb_qs_prior"):
            n_atoms = int(n_arr[i % len(n_arr)])
            if n_atoms <= 0:
                continue
            # Za may be compressed to a single stoichiometry row.
            za_row = za[i % len(za)]
            z_i = np.asarray(za_row[:n_atoms], dtype=int)
            r_i = np.asarray(new_input["Ra"][i, :n_atoms], dtype=float)
            q_tot = float(q_flat[i % q_len])
            s_tot = float(s_flat[i % s_len])
            atoms = Atoms(numbers=z_i, positions=r_i)
            atoms.info["charge"] = q_tot
            atoms.info["spin"] = _multiplicity_from_enerzyme_S(s_tot)
            try:
                qa, sa = self._atomic_qs_fn(atoms, self.max_scf_iter)
                qa = np.asarray(qa, dtype=np.float64).reshape(-1)
                sa = np.asarray(sa, dtype=np.float64).reshape(-1)
                if qa.size != n_atoms or sa.size != n_atoms:
                    raise ValueError(f"expected length {n_atoms}, got Qa {qa.size}, Sa {sa.size}")
                qa, sa = _conserve_atomic_totals(qa, sa, n_atoms, q_tot, s_tot)
                q_block[i, :n_atoms] = qa
                s_block[i, :n_atoms] = sa
            except Exception as e:
                failures.append(i)
                logger.error(f"xtb_qs_prior: frame {i} failed: {e}")
        if failures:
            raise RuntimeError(
                f"xtb_qs_prior: {len(failures)} / {n_frames} frames failed (indices {failures[:20]}"
                f"{'...' if len(failures) > 20 else ''}); no silent fallback."
            )
        if self.out_q in new_input:
            del new_input[self.out_q]
        if self.out_s in new_input:
            del new_input[self.out_s]
        new_input.create_dataset(self.out_q, data=q_block)
        new_input.create_dataset(self.out_s, data=s_block)
        logger.info("xtb_qs_prior: wrote Q_init_a and S_init_a from xTB")

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        pass


class PySCFNAOQSPriorTransform(BaseTransform):
    """GPU4PySCF/PySCF NAO per-atom Q/S prior into ``Q_init_a`` / ``S_init_a`` (HDF5 cache only).

    DFT settings are taken from YAML / constructor kwargs (no built-in xc or basis).
    Required when not injecting ``atomic_qs_fn``: ``xc``, ``basis``.
    Optional: ``max_scf_iter``, ``conv_tol``, ``density_fit``, ``use_gpu``, ``verbose``, ``n_processes``.
    """

    POPULATED_KEYS = frozenset({"Q_init_a", "S_init_a"})

    def __init__(
        self,
        q_key: str = "Q",
        s_key: str = "S",
        n_key: str = "N",
        out_q: str = "Q_init_a",
        out_s: str = "S_init_a",
        max_scf_iter: int = 1,
        xc: Optional[str] = None,
        basis: Optional[str] = None,
        conv_tol: float = 1e-6,
        density_fit: bool = True,
        use_gpu: bool = True,
        verbose: int = 0,
        n_processes: int = 1,
        atomic_qs_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("module_path", None)
        if kwargs:
            logger.warning("PySCFNAOQSPriorTransform: ignoring unrecognized keys %s", sorted(kwargs))
        super().__init__(major_key=out_q)
        self.q_key = q_key
        self.s_key = s_key
        self.n_key = n_key
        self.out_q = out_q
        self.out_s = out_s
        self.max_scf_iter = int(max_scf_iter)
        self.xc = xc
        self.basis = basis
        self.conv_tol = float(conv_tol)
        self.density_fit = bool(density_fit)
        self.use_gpu = bool(use_gpu)
        self.verbose = int(verbose)
        self.n_processes = int(n_processes)
        if atomic_qs_fn is not None:
            self._atomic_qs_fn = atomic_qs_fn
        else:
            if not xc or not basis:
                raise ValueError(
                    "pyscf_nao_qs_prior: 'xc' and 'basis' are required in YAML when "
                    "atomic_qs_fn is not set (no default functional or basis group)."
                )
            from ..qm.pyscf_nao_population.deps import check_pyscf_nao_dependencies
            from ..qm.pyscf_nao_population.atomic_populations import atomic_Q_and_S_from_pyscf_nao

            check_pyscf_nao_dependencies(use_gpu=self.use_gpu)
            self._atomic_qs_fn = partial(
                atomic_Q_and_S_from_pyscf_nao,
                xc=str(xc),
                basis=str(basis),
                conv_tol=self.conv_tol,
                density_fit=self.density_fit,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
            )
        self.transform_type = "feature"

    def transform(self, new_input: Dict[str, Iterable]) -> None:
        required = ("Ra", "Za", self.n_key, self.q_key, self.s_key)
        for rk in required:
            if rk not in new_input:
                raise KeyError(f"pyscf_nao_qs_prior: missing required dataset '{rk}'")
        za = np.asarray(new_input["Za"])
        if za.ndim == 1:
            za = za.reshape(1, -1)
        elif za.ndim != 2:
            raise ValueError(f"pyscf_nao_qs_prior: Za must be 1-D or 2-D, got shape {za.shape}")
        # Ra is never compressed; N/Q/S may be shared across frames when compressed.
        n_frames = len(new_input["Ra"])
        max_n = int(za.shape[1])
        n_arr = np.asarray(new_input[self.n_key][:], dtype=np.int64).ravel()
        q_flat = np.asarray(new_input[self.q_key][:], dtype=np.float64).ravel()
        s_flat = np.asarray(new_input[self.s_key][:], dtype=np.float64).ravel()
        q_len = max(len(q_flat), 1)
        s_len = max(len(s_flat), 1)
        q_block = np.zeros((n_frames, max_n), dtype=np.float64)
        s_block = np.zeros((n_frames, max_n), dtype=np.float64)
        failures: List[int] = []
        tasks = []
        za_len = len(za)
        for i in range(n_frames):
            n_atoms = int(n_arr[i % len(n_arr)])
            if n_atoms <= 0:
                tasks.append(
                    (
                        i,
                        np.zeros(0, dtype=int),
                        np.zeros((0, 3), dtype=np.float64),
                        n_atoms,
                        0.0,
                        0.0,
                        self.max_scf_iter,
                        self._atomic_qs_fn,
                    )
                )
                continue
            za_row = za[i % za_len]
            z_i = np.asarray(za_row[:n_atoms], dtype=int)
            r_i = np.asarray(new_input["Ra"][i, :n_atoms], dtype=float)
            q_tot = float(q_flat[i % q_len])
            s_tot = float(s_flat[i % s_len])
            tasks.append((i, z_i, r_i, n_atoms, q_tot, s_tot, self.max_scf_iter, self._atomic_qs_fn))

        n_processes = max(1, self.n_processes)
        if n_processes == 1:
            results = (
                _pyscf_nao_qs_prior_worker(task)
                for task in tqdm(tasks, desc="pyscf_nao_qs_prior")
            )
            for i, qa, sa, error in results:
                if error is not None:
                    failures.append(i)
                    logger.error(f"pyscf_nao_qs_prior: frame {i} failed: {error}")
                    continue
                n_atoms = int(n_arr[i % len(n_arr)])
                q_block[i, :n_atoms] = qa
                s_block[i, :n_atoms] = sa
        else:
            logger.info(f"pyscf_nao_qs_prior: running with {n_processes} processes (spawn)")
            # CUDA contexts are not fork-safe. Use spawn so each worker initializes
            # GPU4PySCF/CuPy from a clean interpreter instead of inheriting parent GPU state.
            ctx = get_context("spawn")
            with ctx.Pool(n_processes) as pool:
                results = pool.imap(_pyscf_nao_qs_prior_worker, tasks)
                for i, qa, sa, error in tqdm(results, total=len(tasks), desc="pyscf_nao_qs_prior"):
                    if error is not None:
                        failures.append(i)
                        logger.error(f"pyscf_nao_qs_prior: frame {i} failed: {error}")
                        continue
                    n_atoms = int(n_arr[i % len(n_arr)])
                    q_block[i, :n_atoms] = qa
                    s_block[i, :n_atoms] = sa
        if failures:
            raise RuntimeError(
                f"pyscf_nao_qs_prior: {len(failures)} / {n_frames} frames failed (indices {failures[:20]}"
                f"{'...' if len(failures) > 20 else ''}); no silent fallback."
            )
        if self.out_q in new_input:
            del new_input[self.out_q]
        if self.out_s in new_input:
            del new_input[self.out_s]
        new_input.create_dataset(self.out_q, data=q_block)
        new_input.create_dataset(self.out_s, data=s_block)
        logger.info("pyscf_nao_qs_prior: wrote Q_init_a and S_init_a from PySCF NAO")

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        pass


class QSDeltaTransform(BaseTransform):
    """In-place atomic delta transform: ``Qa`` / ``Sa`` = target minus prior (HDF5 cache only).

    Forward runs only after ``Q_init_a`` / ``S_init_a`` exist (``Transform.transform`` order:
    all ``uniform_qs_init``, then all ``xtb_qs_prior`` / ``pyscf_nao_qs_prior``, then all
    ``qs_delta`` — YAML key order within each group does not change this).

    NSE heads still output ``Qa`` / ``Sa``. To keep loss computation aligned with those output
    names, this transform overwrites the loaded ``Qa`` / ``Sa`` targets with residual labels.
    ``single_inverse_transform`` maps predicted residuals back to full ``Qa`` / ``Sa`` in place by
    adding the prior for metrics and comparison to NBO references. Requires ``Q_init_a`` /
    ``S_init_a`` on the same dict (see ``_decorate_batch_output`` copying from batch features).
    """

    POPULATED_KEYS = frozenset()

    def __init__(
        self,
        qa_key: str = "Qa",
        sa_key: str = "Sa",
        q_init_key: str = "Q_init_a",
        s_init_key: str = "S_init_a",
        n_key: str = "N",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("out_q", None)
        kwargs.pop("out_s", None)
        if kwargs:
            logger.warning("QSDeltaTransform: ignoring unrecognized keys %s", sorted(kwargs))
        super().__init__(major_key=qa_key)
        self.qa_key = qa_key
        self.sa_key = sa_key
        self.q_init_key = q_init_key
        self.s_init_key = s_init_key
        self.n_key = n_key
        self.transform_type = "target"

    def transform(self, new_input: Dict[str, Iterable]) -> None:
        need = (self.qa_key, self.sa_key, self.q_init_key, self.s_init_key, self.n_key, "Ra", "Za")
        for k in need:
            if k not in new_input:
                raise KeyError(f"qs_delta: missing '{k}'")
        # Ra is never compressed; N may be shared across frames when compressed.
        n_frames = len(new_input["Ra"])
        za = np.asarray(new_input["Za"])
        if za.ndim == 1:
            max_n = int(za.shape[0])
        else:
            max_n = int(za.shape[1])
        n_arr = np.asarray(new_input[self.n_key][:], dtype=np.int64).ravel()
        dq = np.zeros((n_frames, max_n), dtype=np.float64)
        ds = np.zeros((n_frames, max_n), dtype=np.float64)
        for i in range(n_frames):
            n_atoms = int(n_arr[i % len(n_arr)])
            if n_atoms <= 0:
                continue
            qa = np.asarray(new_input[self.qa_key][i, :n_atoms], dtype=np.float64)
            sa = np.asarray(new_input[self.sa_key][i, :n_atoms], dtype=np.float64)
            q0 = np.asarray(new_input[self.q_init_key][i, :n_atoms], dtype=np.float64)
            s0 = np.asarray(new_input[self.s_init_key][i, :n_atoms], dtype=np.float64)
            dq[i, :n_atoms] = qa - q0
            ds[i, :n_atoms] = sa - s0
        del new_input[self.qa_key]
        del new_input[self.sa_key]
        new_input.create_dataset(self.qa_key, data=dq)
        new_input.create_dataset(self.sa_key, data=ds)
        logger.info("qs_delta: overwrote Qa and Sa with residual targets")

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        if self.qa_key not in new_output or self.sa_key not in new_output:
            return
        if self.q_init_key not in new_output or self.s_init_key not in new_output:
            raise KeyError(
                f"qs_delta inverse_transform: need '{self.q_init_key}' and '{self.s_init_key}' "
                f"alongside '{self.qa_key}' / '{self.sa_key}' (they are copied from batch features in "
                "_decorate_batch_output when present)."
            )
        dq = np.asarray(new_output[self.qa_key][idx], dtype=np.float64).reshape(-1)
        ds = np.asarray(new_output[self.sa_key][idx], dtype=np.float64).reshape(-1)
        q0 = np.asarray(new_output[self.q_init_key][idx], dtype=np.float64).reshape(-1)
        s0 = np.asarray(new_output[self.s_init_key][idx], dtype=np.float64).reshape(-1)
        n = dq.shape[0]
        if q0.shape[0] < n or s0.shape[0] < n:
            raise ValueError(
                f"qs_delta inverse: prior length {q0.shape[0]} / {s0.shape[0]} < delta length {n}"
            )
        new_output[self.qa_key][idx] = dq + q0[:n]
        new_output[self.sa_key][idx] = ds + s0[:n]



class UniformSplitQSTransform(BaseTransform):
    """Per-frame uniform split of total charge Q and spin S onto atoms: Q_init_a = Q/N, S_init_a = S/N.

    S uses the same convention as elsewhere (multiplicity minus one). Missing Q or S default to 0.
    """

    POPULATED_KEYS = frozenset({"Q_init_a", "S_init_a"})

    def __init__(
        self,
        q_key: str = "Q",
        s_key: str = "S",
        n_key: str = "N",
        out_q: str = "Q_init_a",
        out_s: str = "S_init_a",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(major_key=out_q)
        self.q_key = q_key
        self.s_key = s_key
        self.n_key = n_key
        self.out_q = out_q
        self.out_s = out_s
        self.transform_type = "feature"

    def transform(self, new_input: Dict[str, Iterable]) -> None:
        if self.n_key not in new_input:
            logger.warning("uniform_qs_init: missing N; skip Q_init_a / S_init_a")
            return
        if "Ra" not in new_input:
            logger.warning("uniform_qs_init: missing Ra; skip Q_init_a / S_init_a")
            return
        # Ra is never compressed; N/Q/S may be shared across frames when compressed.
        n_frames = len(new_input["Ra"])
        za = new_input["Za"]
        if len(za.shape) < 2:
            logger.warning("uniform_qs_init: unexpected Za shape; skip")
            return
        max_n = int(za.shape[1])
        if self.q_key in new_input:
            q_flat = np.asarray(new_input[self.q_key][:], dtype=np.float64).ravel()
        else:
            q_flat = np.zeros(1, dtype=np.float64)
        if self.s_key in new_input:
            s_flat = np.asarray(new_input[self.s_key][:], dtype=np.float64).ravel()
        else:
            s_flat = np.zeros(1, dtype=np.float64)
        q_len = max(len(q_flat), 1)
        s_len = max(len(s_flat), 1)
        q_block = np.zeros((n_frames, max_n), dtype=np.float64)
        s_block = np.zeros((n_frames, max_n), dtype=np.float64)
        n_arr = np.asarray(new_input[self.n_key][:], dtype=np.int64).ravel()
        for i in range(n_frames):
            n_atoms = int(n_arr[i % len(n_arr)])
            q_val = float(q_flat[i % q_len])
            s_val = float(s_flat[i % s_len])
            if n_atoms <= 0:
                continue
            inv = 1.0 / n_atoms
            q_block[i, :n_atoms] = q_val * inv
            s_block[i, :n_atoms] = s_val * inv
        if self.out_q in new_input:
            del new_input[self.out_q]
        if self.out_s in new_input:
            del new_input[self.out_s]
        new_input.create_dataset(self.out_q, data=q_block)
        new_input.create_dataset(self.out_s, data=s_block)
        logger.info("uniform_qs_init: wrote Q_init_a and S_init_a (uniform Q/N, S/N per frame)")

    def single_inverse_transform(self, new_output: Dict[str, Iterable], idx: int) -> None:
        pass


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
        self.uniform_qs_inits: List[UniformSplitQSTransform] = []
        self.xtb_qs_priors: List[XTBQSPriorTransform] = []
        self.pyscf_nao_qs_priors: List[PySCFNAOQSPriorTransform] = []
        self.qs_deltas: List[QSDeltaTransform] = []
        if transform_args is None:
            return
        for k, v in transform_args.items():
            if k == "atomic_energy":
                self.shifts.append(AtomicEnergyTransform(v))
                self.backup_keys.add("E")
            if k == "negative_gradient" and v and (not simulation_mode):
                self.scales.append(NegativeGradientTransform())
                self.backup_keys.add("Fa")
            if k == "energy_unit_conversion" and v:
                kwargs = v if isinstance(v, dict) else {}
                self.scales.append(EnergyUnitConversionTransform(**kwargs))
                self.backup_keys.update({"E", "Fa"})
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
            if k == "uniform_qs_init":
                if v is False or v is None:
                    continue
                if isinstance(v, dict) and v.get("enabled") is False:
                    continue
                kwargs = {k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else {}
                kwargs.pop("enabled", None)
                self.uniform_qs_inits.append(UniformSplitQSTransform(**kwargs))
            if k == "xtb_qs_prior":
                if v is False or v is None:
                    continue
                if isinstance(v, dict) and v.get("enabled") is False:
                    continue
                kwargs = {k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else {}
                kwargs.pop("enabled", None)
                self.xtb_qs_priors.append(XTBQSPriorTransform(**kwargs))
            if k == "pyscf_nao_qs_prior":
                if v is False or v is None:
                    continue
                if isinstance(v, dict) and v.get("enabled") is False:
                    continue
                kwargs = {k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else {}
                kwargs.pop("enabled", None)
                self.pyscf_nao_qs_priors.append(PySCFNAOQSPriorTransform(**kwargs))
            if k == "qs_delta":
                if v is False or v is None:
                    continue
                if isinstance(v, dict) and v.get("enabled") is False:
                    continue
                kwargs = {k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else {}
                kwargs.pop("enabled", None)
                self.qs_deltas.append(QSDeltaTransform(**kwargs))

        _prior_count = (
            bool(self.uniform_qs_inits)
            + bool(self.xtb_qs_priors)
            + bool(self.pyscf_nao_qs_priors)
        )
        if _prior_count > 1:
            raise ValueError(
                "Transform: enable only one of uniform_qs_init, xtb_qs_prior, or "
                "pyscf_nao_qs_prior; all write Q_init_a / S_init_a and a later stage would "
                "overwrite the earlier without warning."
            )

    def transform(self, raw_input: Dict):
        for shift in self.shifts:
            shift.transform(raw_input)
        for scale in self.scales:
            scale.transform(raw_input)
        for normalization in self.normalizations:
            normalization.transform(raw_input)
        # Q/S priors before deltas so Q_init_a / S_init_a exist when computing residuals
        for u in self.uniform_qs_inits:
            u.transform(raw_input)
        for x in self.xtb_qs_priors:
            x.transform(raw_input)
        for p in self.pyscf_nao_qs_priors:
            p.transform(raw_input)
        for d in self.qs_deltas:
            d.transform(raw_input)

    def inverse_transform(self, raw_output: Dict, selected_indices: Optional[Iterable[int]]=None):
        # Reverse of forward: undo deltas first, then energy-related inverses
        for d in reversed(self.qs_deltas):
            d.inverse_transform(raw_output, selected_indices)
        for x in reversed(self.xtb_qs_priors):
            x.inverse_transform(raw_output, selected_indices)
        for p in reversed(self.pyscf_nao_qs_priors):
            p.inverse_transform(raw_output, selected_indices)
        for u in reversed(self.uniform_qs_inits):
            u.inverse_transform(raw_output, selected_indices)
        for normalization in self.normalizations:
            normalization.inverse_transform(raw_output, selected_indices)
        for scale in self.scales:
            scale.inverse_transform(raw_output, selected_indices)
        for shift in self.shifts:
            shift.inverse_transform(raw_output, selected_indices)


