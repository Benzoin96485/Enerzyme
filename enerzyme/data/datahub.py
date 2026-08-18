import pickle, os
from bisect import bisect
from glob import glob
from hashlib import md5
from typing import Union, List, Optional, Iterable, Literal, Any, Callable
import h5py
import numpy as np
from ase import Atoms
import ase.units
from ase.db import connect
from addict import Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from .datatype import is_atomic, is_rounded, is_int, register_data_type
from .transform import (
    EnergyUnitConversionTransform,
    NegativeGradientTransform,
    PySCFNAOQSPriorTransform,
    Transform,
    UniformSplitQSTransform,
    XTBQSPriorTransform,
    parse_Za,
    wants_pyscf_nao_qs_prior,
    wants_qs_delta,
    wants_uniform_qs_init,
    wants_xtb_qs_prior,
)
from ..utils import YamlHandler, logger


def _atomic_charges_from_atoms(atoms: Atoms) -> Any:
    """Qa from calculator ``charges`` when present, else ASE ``initial_charges``."""
    if atoms.calc is not None:
        results = getattr(atoms.calc, "results", None) or {}
        if "charges" in results:
            return atoms.get_charges()
        try:
            return atoms.get_charges()
        except Exception:
            pass
    if atoms.has("initial_charges"):
        return atoms.get_initial_charges()
    raise AttributeError(
        "No atomic charges: calculator charges / get_charges() and "
        "initial_charges are all unavailable"
    )


def _atomic_magmoms_from_atoms(atoms: Atoms) -> Any:
    """Sa from calculator ``magmoms`` when present, else ASE ``initial_magmoms``."""
    if atoms.calc is not None:
        results = getattr(atoms.calc, "results", None) or {}
        if "magmoms" in results:
            return atoms.get_magnetic_moments()
        try:
            return atoms.get_magnetic_moments()
        except Exception:
            pass
    if atoms.has("initial_magmoms"):
        return atoms.get_initial_magnetic_moments()
    raise AttributeError(
        "No atomic magmoms: calculator magmoms / get_magnetic_moments() and "
        "initial_magmoms are all unavailable"
    )


# Standard Enerzyme names → ASE Atoms accessors (calculator results / geometry).
# ASE itself still stores calculator keys as energy/forces/dipole/charges/magmoms
# and fairchem-style info keys charge/spin; this adapter only exposes standard names.
# Qa/Sa prefer calculator populations, then fall back to ASE initial_* arrays.
ASE_PROPERTY_METHODS: Dict[str, Callable[[Atoms], Any]] = {
    "E": lambda atoms: atoms.get_potential_energy(),
    "Fa": lambda atoms: atoms.get_forces(),
    "Qa": _atomic_charges_from_atoms,
    "Sa": _atomic_magmoms_from_atoms,
    "M2": lambda atoms: atoms.get_dipole_moment(),
}

# ASE SinglePointCalculator.results keys → standard Enerzyme names.
_ASE_RESULTS_KEY_TO_STANDARD = {
    "energy": "E",
    "forces": "Fa",
    "dipole": "M2",
    "charges": "Qa",
    "magmoms": "Sa",
}

# Written by annotate into ase.db metadata; preferred over first-row probing.
ASELMDB_METADATA_PROPERTIES_KEY = "enerzyme_properties"

# ASE info keys that are remapped to standard names (not exposed as Datahub keys).
_ASE_INFO_STANDARD_KEYS = frozenset({"charge", "spin"})

# Always available geometry / charge fields (no calculator required).
_ASELMDB_GEOMETRY_KEYS = frozenset({"Ra", "Za", "N", "Q", "S"})

# ASELMDBDataset exposes these under standard Enerzyme names only (not pickle aliases
# like energy/coord/grad). Custom row-data fields may still use non-identity maps.
_ASELMDB_FIXED_KEYS = _ASELMDB_GEOMETRY_KEYS | frozenset(ASE_PROPERTY_METHODS)


def _probe_calculator_properties(atoms: Atoms) -> set:
    """Discover ASE-backed fields from ``calc.results``, accessors, or initial_*."""
    found: set = set()
    if atoms.calc is not None:
        results = getattr(atoms.calc, "results", None) or {}
        for ase_key, std_key in _ASE_RESULTS_KEY_TO_STANDARD.items():
            if ase_key in results:
                found.add(std_key)
        for prop, method in ASE_PROPERTY_METHODS.items():
            if prop in found:
                continue
            try:
                method(atoms)
            except Exception as e:
                logger.warning(f"Failed to get {prop} directly from calculator: {e}")
            else:
                found.add(prop)
    # DBs without a calculator (or lacking charges/magmoms) can still expose
    # populations via ASE initial arrays, e.g. BOS-TMC aselmdb.
    if "Qa" not in found and atoms.has("initial_charges"):
        found.add("Qa")
    if "Sa" not in found and atoms.has("initial_magmoms"):
        found.add("Sa")
    return found


def _read_aselmdb_schema_properties(dbs) -> set:
    """Union ``enerzyme_properties`` metadata across connected ASE databases."""
    schema: set = set()
    for db in dbs:
        try:
            meta = dict(getattr(db, "metadata", None) or {})
        except Exception as e:
            logger.warning(f"Failed to read ASE DB metadata: {e}")
            continue
        props = meta.get(ASELMDB_METADATA_PROPERTIES_KEY)
        if not props:
            continue
        if isinstance(props, str):
            props = [props]
        schema.update(str(p) for p in props)
    return schema


def load_from_pickle(data_path=str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        keys = set()
        for datapoint in data:
            keys.update(datapoint.keys())
        logger.info(f"Collected keys from data: {keys}")
        dd = {key: [datapoint.get(key, None) for datapoint in data] for key in keys}
        return dd
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError(f"Unknown data type in {data_path}!")


def load_from_sdf(data_path: str) -> Dict[str, list]:
    """Load an SDF into a column-oriented dict for Datahub.

    Uses the Atoms-based :class:`SDFSupplier` and exposes standard Enerzyme
    geometry / charge fields (``Ra``, ``Za``, ``N``, ``Q``, ``S``). Feature
    maps should use those names as source attributes (identity mapping).
    """
    from .supplier import SDFSupplier

    raw_data: Dict[str, list] = {"Ra": [], "Za": [], "N": [], "Q": [], "S": []}
    for atoms in SDFSupplier(input_file=data_path).suppl():
        raw_data["Ra"].append(np.asarray(atoms.get_positions()))
        raw_data["Za"].append(np.asarray(atoms.get_atomic_numbers()))
        raw_data["N"].append(len(atoms))
        raw_data["Q"].append(int(atoms.info.get("charge", 0)))
        raw_data["S"].append(int(atoms.info.get("spin", 1)) - 1)
    logger.info(f"Collected keys from SDF: {set(raw_data)}")
    return raw_data


# Directory / glob expansion only considers these ASE DB suffixes.
_ASELMDB_FILE_SUFFIXES = (".aselmdb", ".db")


def _is_aselmdb_lock_path(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.endswith(".lock") or name.endswith("-lock")


def _is_aselmdb_db_path(path: str) -> bool:
    """True for regular files that look like ASE / ASE LMDB databases."""
    if not os.path.isfile(path) or _is_aselmdb_lock_path(path):
        return False
    name = os.path.basename(path).lower()
    return any(name.endswith(suffix) for suffix in _ASELMDB_FILE_SUFFIXES)


def _get_single_aselmdb_data_path(data_path=str) -> List[str]:
    """Resolve a file, directory, or glob to ASE DB file paths.

    Directories only include ``*.aselmdb`` / ``*.db`` regular files (lock files
    and other non-DB paths are ignored). Explicit globs still expand as given,
    but lock files are filtered out.
    """
    if os.path.isfile(data_path):
        return [data_path]
    elif os.path.isdir(data_path):
        candidates = [
            p for p in glob(os.path.join(data_path, "*")) if _is_aselmdb_db_path(p)
        ]
        return sorted(candidates)
    else:
        candidates = [
            p for p in glob(data_path)
            if os.path.isfile(p) and not _is_aselmdb_lock_path(p)
        ]
        return sorted(candidates)


class ASELMDBSingleProperty:
    def __init__(self, aselmdb_dataset: "ASELMDBDataset", get_property_method: Callable[[Atoms], Any]):
        self.aselmdb_dataset = aselmdb_dataset
        self.get_property_method = get_property_method
    
    def __len__(self) -> int:
        return len(self.aselmdb_dataset)

    def __getitem__(self, idx: int) -> Union[int, float, np.ndarray]:
        db_idx = bisect(self.aselmdb_dataset._id_len_segments, idx)

        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self.aselmdb_dataset._id_len_segments[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.aselmdb_dataset.dbs[db_idx]._get_row(self.aselmdb_dataset.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()

        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        if "sid" not in atoms.info:
            atoms.info["sid"] = idx

        return self.get_property_method(atoms)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class ASELMDBDataset:
    def __init__(
        self,
        data_path,
        new_energy_unit: str="Ha",
        connect_args: Dict[str, Any]=dict(),
        select_args: Dict[str, Any]=dict(),
        transforms: Optional[Dict[str, Any]]=None,
        declared_properties: Optional[Iterable[str]]=None,
    ):
        if transforms and transforms.get("energy_unit_conversion"):
            logger.warning(
                "energy_unit_conversion transform is disabled for ASELMDB; "
                f"energies/forces are already converted via new_energy_unit={new_energy_unit}"
            )
        if transforms and transforms.get("negative_gradient"):
            logger.warning(
                "negative_gradient transform is disabled for ASELMDB; "
                "Fa already comes from ASE get_forces() (physical forces, not ∇E)"
            )
        if new_energy_unit != "eV":
            logger.info(f"Loading ASE energy in {new_energy_unit}")
        self.energy_unit_conversion_factor = 1 / getattr(ase.units, new_energy_unit)
        self.data_paths = []
        self.dbs = []
        self.db_ids = []
        default_connect_args = {
            "readonly": True,
            "use_lock_file": False
        }
        default_connect_args.update(connect_args)
        if isinstance(data_path, list):
            for single_data_path in data_path:
                self.data_paths.extend(_get_single_aselmdb_data_path(single_data_path))
        else:
            self.data_paths.extend(_get_single_aselmdb_data_path(data_path))
        if not self.data_paths:
            raise ValueError(f"No ASE LMDB / ASE DB files found in {data_path}")

        connected_paths: List[str] = []
        for single_data_path in self.data_paths:
            try:
                self.dbs.append(connect(single_data_path, **default_connect_args))
            except Exception as e:
                raise ValueError(
                    f"Failed to connect to ASE LMDB path {single_data_path}: {e}"
                ) from e
            connected_paths.append(single_data_path)
        self.data_paths = connected_paths

        for db in self.dbs:
            if hasattr(db, "ids") and not select_args:
                self.db_ids.append(db.ids)
            else:
                self.db_ids.append([row.id for row in db.select(**select_args)])
        self.id_lens = [len(ids) for ids in self.db_ids]
        self._id_len_segments = np.cumsum(self.id_lens).tolist()
        if not self._id_len_segments or self._id_len_segments[-1] == 0:
            raise ValueError(f"No rows found in ASE LMDB path(s): {data_path}")

        first_db = self.dbs[0]
        try:
            first_row = first_db._get_row(self.db_ids[0][0])
            first_atoms = first_row.toatoms()
        except Exception as e:
            raise ValueError(f"Failed to get atoms from {first_db}: {e}")

        # Discovery priority: DB schema metadata → Datahub-declared fields → first-row probe.
        # First-row probing alone misses properties present only on later frames.
        properties_from_schema = _read_aselmdb_schema_properties(self.dbs)
        declared = {str(p) for p in (declared_properties or [])}
        properties_from_calculator = _probe_calculator_properties(first_atoms)

        # ASE may store data as null / omit it; AtomsRow.data then returns None
        # or raises when wrapping None in FancyDict.
        try:
            row_data = first_row.data
        except TypeError:
            row_data = None

        # Row ``data`` / ``atoms.info`` hold charge/spin/index and custom fields.
        # Numeric values alone are not enough: fixed calculator / geometry names
        # (E, Fa, M2, …) must stay calculator-backed even if duplicated in data,
        # otherwise ``atoms.info.get`` skips unit conversion and can return stale
        # copies. Annotate writes calculator results + charge/spin/index only.
        raw_info_keys: set = set()
        if row_data:
            for k, v in row_data.items():
                if k in _ASE_INFO_STANDARD_KEYS or k in _ASELMDB_FIXED_KEYS:
                    continue
                if isinstance(v, (float, int, np.floating, np.integer, np.ndarray)):
                    raw_info_keys.add(k)

        schema_calc = properties_from_schema & set(ASE_PROPERTY_METHODS)
        schema_info = properties_from_schema - _ASELMDB_FIXED_KEYS
        declared_calc = declared & set(ASE_PROPERTY_METHODS)

        calculator_backed = properties_from_calculator | schema_calc | declared_calc
        # Priority: calculator accessors for ASE_PROPERTY_METHODS; info only for
        # custom / non-fixed keys. Healthy DBs keep these disjoint.
        ignored_fixed_in_row_data = set()
        if row_data:
            ignored_fixed_in_row_data = (
                set(row_data) & _ASELMDB_FIXED_KEYS & calculator_backed
            )
        self.properties_from_info = raw_info_keys | schema_info
        self.unique_properties_from_calculator = calculator_backed - self.properties_from_info
        overlapped_properties = calculator_backed & self.properties_from_info

        if (schema_calc or declared_calc) and (
            (schema_calc | declared_calc) - properties_from_calculator
        ):
            missing_on_first = (schema_calc | declared_calc) - properties_from_calculator
            logger.info(
                "Calculator fields not on the first ASE row but registered via "
                f"schema/declared maps: {sorted(missing_on_first)}"
            )

        # Q / S always available via atoms.info defaults (charge=0, spin multiplicity=1 → S=0)
        self._all_properties = (
            self.unique_properties_from_calculator
            | self.properties_from_info
            | _ASELMDB_GEOMETRY_KEYS
        )
        if ignored_fixed_in_row_data:
            logger.warning(
                f"Property {sorted(ignored_fixed_in_row_data)} present in ASE row "
                "data/info is ignored; using calculator accessors"
            )
        if overlapped_properties:
            logger.warning(
                f"Property {sorted(overlapped_properties)} found in both calculator "
                "and info; using calculator accessors"
            )
        if properties_from_schema:
            logger.info(f"Properties from ASE DB schema: {sorted(properties_from_schema)}")
        if declared:
            logger.info(f"Properties declared by Datahub: {sorted(declared)}")
        logger.info(f"Properties from calculator: {self.unique_properties_from_calculator}")
        logger.info(f"Properties from info: {self.properties_from_info}")

    def __len__(self) -> int:
        return sum(self.id_lens)

    def __getitem__(self, k) -> np.ndarray:
        if k == "Ra":
            get_property_method = lambda atoms: atoms.get_positions()
        elif k == "Za":
            get_property_method = lambda atoms: atoms.get_atomic_numbers()
        elif k == "N":
            get_property_method = lambda atoms: len(atoms)
        elif k == "Q":
            get_property_method = lambda atoms: atoms.info.get("charge", 0)
        elif k == "S":
            get_property_method = lambda atoms: atoms.info.get("spin", 1) - 1
        elif k in ASE_PROPERTY_METHODS and k in self.unique_properties_from_calculator:
            if k in {"E", "Fa"}:
                get_property_method = lambda atoms, p=k: ASE_PROPERTY_METHODS[p](atoms) * self.energy_unit_conversion_factor
            else:
                get_property_method = lambda atoms, p=k: ASE_PROPERTY_METHODS[p](atoms)
        else:
            get_property_method = lambda atoms, key=k: atoms.info.get(key, None)
        return ASELMDBSingleProperty(self, get_property_method=get_property_method)
    
    def __contains__(self, k) -> bool:
        return k in self._all_properties

    def items(self):
        return {k: self[k] for k in self._all_properties}

    def keys(self):
        return self._all_properties

    def values(self):
        return [self[k] for k in self._all_properties]


def _collect_types(types: Optional[Union[List, Dict]]) -> Dict:
    if types is None:
        return dict()
    elif isinstance(types, list):
        return {single_type: single_type for single_type in types}
    else:
        return {k: v if v is not None else k for k, v in types.items()}


def array_padding(data, max_N, pad_value=0):
    for i in range(len(data)):
        pad_shape = [(0, max_N - len(data[i]))] + [(0,0)] * (len(data[i].shape) - 1)
        data[i] = np.pad(data[i], pad_shape, constant_values=pad_value)
    return np.array(data)


class FieldDataset(Dataset):
    def __init__(self, data: Dict[str, Iterable]) -> None:
        self.data = data
        self.compressed_keys = set()
        self._h5_file = None
        for k, v in self.data.items():
            if len(v) == 1:
                self.compressed_keys.add(k)

    def __getstate__(self):
        """Pickle numpy arrays; reopen HDF5 datasets by filename in workers."""
        arrays = {}
        h5_filename = None
        h5_names = {}
        for k, v in self.data.items():
            if isinstance(v, h5py.Dataset):
                filename = v.file.filename
                if isinstance(filename, bytes):
                    filename = filename.decode()
                h5_filename = filename
                h5_names[k] = v.name
            else:
                arrays[k] = v
        return {
            "compressed_keys": set(self.compressed_keys),
            "arrays": arrays,
            "h5_filename": h5_filename,
            "h5_names": h5_names,
        }

    def __setstate__(self, state):
        self.compressed_keys = state["compressed_keys"]
        self.data = dict(state.get("arrays") or {})
        self._h5_file = None
        filename = state.get("h5_filename")
        names = state.get("h5_names") or {}
        if filename and names:
            self._h5_file = h5py.File(filename, "r")
            for k, name in names.items():
                self.data[k] = self._h5_file[name]

    def __del__(self):
        handle = getattr(self, "_h5_file", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
            self._h5_file = None

    def __getitem__(self, k) -> Iterable:
        return self.data[k]

    def __setitem__(self, k, v) -> None:
        self.data[k] = v
        if len(v) == 1:
            self.compressed_keys.add(k)

    def __contains__(self, k) -> bool:
        return k in self.data
    
    def __len__(self) -> int:
        for v in self.data.values():
            if len(v) != 1:
                return len(v)
        else:
            return 1

    def items(self):
        return self.data.items()
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def loc(self, idx) -> Dict[str, Iterable]:
        return {k: v[0 if k in self.compressed_keys else idx] for k, v in self.data.items()}

    def load_subset(self, indices: Iterable[int]) -> "FieldDataset":
        data = dict()
        for k, v in self.data.items():
            if k in self.compressed_keys:
                data[k] = np.array(v)
            else:
                data[k] = np.array([v[idx] for idx in indices])
        return FieldDataset(data)


class SingleDataHub:
    def __init__(self,  
        dump_dir=".",
        data_format: Optional[str]=None,
        data_path: str="", 
        preload: bool=True,
        features: Dict[str, str]=dict(),
        targets: Dict[str, str]=dict(),
        preprocessings: Optional[Dict[str, Union[str, bool]]]=None,
        global_transforms: Optional[Dict[str, Union[str, bool]]]=None,
        neighbor_list: Optional[str]=None,
        hash_length: int=16,
        compressed: bool=True,
        max_memory: int=10,
        connect_args: Dict[str, Any]=dict(),
        select_args: Dict[str, Any]=dict(),
        **params
    ):
        self.data_path = os.path.abspath(data_path)
        self.data_format = data_format
        self.preload = preload
        self.feature_types = _collect_types(features)
        self.target_types = _collect_types(targets)
        self.data_types = self.feature_types | self.target_types
        self._populate_uniform_qs_init = wants_uniform_qs_init(
            global_transforms
        ) or wants_uniform_qs_init(preprocessings)
        self._populate_xtb_qs_prior = wants_xtb_qs_prior(
            global_transforms
        ) or wants_xtb_qs_prior(preprocessings)
        self._populate_pyscf_nao_qs_prior = wants_pyscf_nao_qs_prior(
            global_transforms
        ) or wants_pyscf_nao_qs_prior(preprocessings)
        if self._populate_uniform_qs_init:
            self.feature_types = dict(self.feature_types)
            for _k in UniformSplitQSTransform.POPULATED_KEYS:
                self.feature_types.setdefault(_k, _k)
            self.data_types = self.feature_types | self.target_types
        if self._populate_xtb_qs_prior:
            self.feature_types = dict(self.feature_types)
            for _k in XTBQSPriorTransform.POPULATED_KEYS:
                self.feature_types.setdefault(_k, _k)
            self.data_types = self.feature_types | self.target_types
        if self._populate_pyscf_nao_qs_prior:
            self.feature_types = dict(self.feature_types)
            for _k in PySCFNAOQSPriorTransform.POPULATED_KEYS:
                self.feature_types.setdefault(_k, _k)
            self.data_types = self.feature_types | self.target_types
        self.neighbor_list_type = neighbor_list
        self.compressed = compressed
        self.max_memory = max_memory
        self.connect_args = connect_args or {}
        self.select_args = select_args or {}
        # Empty connect/select args and unset data_format are omitted so existing
        # processed_dataset_<hash>/ caches stay valid.
        datahub_str = data_path + str(neighbor_list) + \
            str(sorted(preprocessings.items()) if preprocessings is not None else '') + \
            str(sorted(global_transforms.items()) if global_transforms is not None else '')
        if data_format:
            datahub_str += str(data_format)
        if self.connect_args:
            datahub_str += str(sorted(self.connect_args.items()))
        if self.select_args:
            datahub_str += str(sorted(self.select_args.items()))
        self.hash = md5(datahub_str.encode("utf-8")).hexdigest()[:hash_length]
        self.dump_dir = dump_dir
        self.preload_path = os.path.join(dump_dir, f"processed_dataset_{self.hash}")
        logger.info(f"Preload path {self.preload_path} is created")
        _pre = preprocessings or {}
        _glb = global_transforms or {}
        _any_uniform = wants_uniform_qs_init(_pre) or wants_uniform_qs_init(_glb)
        _any_xtb = wants_xtb_qs_prior(_pre) or wants_xtb_qs_prior(_glb)
        _any_pyscf = wants_pyscf_nao_qs_prior(_pre) or wants_pyscf_nao_qs_prior(_glb)
        _prior_slots = int(_any_uniform) + int(_any_xtb) + int(_any_pyscf)
        if _prior_slots > 1:
            raise ValueError(
                "DataHub: do not combine uniform_qs_init, xtb_qs_prior, and pyscf_nao_qs_prior "
                "across preprocessings and/or global_transforms; all populate Q_init_a / S_init_a "
                "and a second HDF5 transform pass would overwrite the first."
            )
        if wants_qs_delta(_pre) and not (
            wants_uniform_qs_init(_pre) or wants_xtb_qs_prior(_pre) or wants_pyscf_nao_qs_prior(_pre)
        ):
            raise ValueError(
                "DataHub preprocessings: qs_delta is enabled but no Q/S prior transform "
                "(uniform_qs_init, xtb_qs_prior, or pyscf_nao_qs_prior) in preprocessings. "
                "Priors must be created in the same preprocessing pass before deltas "
                "(preprocessing runs before global_transforms)."
            )
        if wants_qs_delta(_glb) and not (
            wants_uniform_qs_init(_glb) or wants_xtb_qs_prior(_glb) or wants_pyscf_nao_qs_prior(_glb)
        ):
            raise ValueError(
                "DataHub global_transforms: qs_delta requires a Q/S prior transform "
                "(uniform_qs_init, xtb_qs_prior, or pyscf_nao_qs_prior) in the same "
                "global_transforms block so Q_init_a / S_init_a exist before delta targets."
            )
        self.preprocessing = Transform(preprocessings, self.preload_path)
        self.global_transform = Transform(global_transforms, self.preload_path)
        self.preprocessings = preprocessings
        self.global_transforms = global_transforms
        # All ranks always join the exclusive section. Rank 0 no-ops on a
        # cache hit; peers never skip the collective just because their local
        # stat() already sees the directory.
        from ..tasks.distributed import detect_launch_env, is_global_zero, run_rank0_exclusive

        launch = detect_launch_env()
        sync_dir = os.path.abspath(dump_dir)

        def _ensure_cache():
            if self.preload and self.preload_data():
                return
            self.get_handle("w")
            self._init_data()
            self._init_neighbor_list()
            self.preprocessing.transform(self.data)
            self.global_transform.transform(self.data)
            self._save_config()
            self.reset_handle()

        run_rank0_exclusive(
            _ensure_cache,
            env=launch,
            sync_dir=sync_dir,
            name=f"datahub_{self.hash}",
        )
        if not is_global_zero(launch):
            if not self.preload_data():
                raise RuntimeError(
                    f"Rank {launch.global_rank} failed to preload dataset from "
                    f"{self.preload_path} after rank 0 finished writing"
                )
        # HDF5 already holds transformed arrays; reload only fitted inverse
        # state (e.g. total_energy_normalization statistics.data). Stateless
        # transforms were fully specified from YAML at construction.
        self.preprocessing.reload_fitted_state()
        self.global_transform.reload_fitted_state()

    def _preload_data(self, hdf5_path):
        loaded_file = h5py.File(hdf5_path, mode="r")
        loaded_data = loaded_file["data"]
        self.data["N"] = loaded_data["N"]
        for k in self.data_types:
            if k == "N":
                continue
            elif is_atomic(k):
                self._load_atomic_data(k, loaded_data)
            else:
                self._load_molecular_data(k, loaded_data)
        loaded_file.close()
    
    def preload_data(self): 
        hdf5_path = os.path.join(self.preload_path, "pre_transformed.hdf5")
        config_path = os.path.join(self.preload_path, "datahub.yaml")
        if (
            os.path.isdir(self.preload_path) and
            os.path.isfile(hdf5_path) and
            os.path.isfile(config_path)
        ):
            handler = YamlHandler(config_path)
            datahub_config = handler.read_yaml()
            preload_data_types = _collect_types(datahub_config.feature) | _collect_types(datahub_config.target)
            if preload_data_types.keys() <= self.data_types.keys():
                # all kinds of features and targets are contained in the processed dataset
                self.get_handle()
                logger.info(f"Data matched and preloaded from {self.preload_path}")
                return True
        return False

    def _expand(self, k: str, values: Iterable) -> np.ndarray:
        if isinstance(values, int) or isinstance(values, float):
            if is_int(k) and self.compressed:
                return np.array([values])
            else:
                logger.info(f"Values of {k} (data type {self.data_types[k]}) are single and repeated")
                return np.full(self.n_datapoint, values)
        else:
            arr = np.asarray(values)
            if is_int(k) and self.compressed:
                return arr
            logger.info(f"Values of {k} (data type {self.data_types[k]}) are single and repeated")
            # Length-1 sequence holding one datapoint payload (scalar or vector)
            if arr.shape[0] == 1:
                arr = arr[0]
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return np.full(self.n_datapoint, arr.item())
            return np.repeat(arr[None, ...], self.n_datapoint, axis=0)
    
    def _compress(self, k: str, values: Iterable) -> np.ndarray:
        # only works for equal length data
        if not (isinstance(values, list) or isinstance(values, np.ndarray)):
            values = list(tqdm(values, total=len(values), desc=f"Enumerating {k} (data type {self.data_types[k]})"))
        value_array = np.array(values)
        if is_int(k) and self.compressed and (value_array == value_array[0]).all():
            logger.info(f"Values of {k} (data type {self.data_types[k]}) are all the same and compressed into a single value")
            return value_array[:1]
        else:
            return value_array

    def _validate_aselmdb_field_maps(self) -> None:
        """Reject pickle-style aliases for fixed ASE LMDB standard names."""
        bad = {
            k: src for k, src in self.data_types.items()
            if k in _ASELMDB_FIXED_KEYS and src != k
        }
        if not bad:
            return
        examples = ", ".join(f"{k}: {src}" for k, src in sorted(bad.items()))
        raise ValueError(
            "data_format=aselmdb exposes standard Enerzyme names only "
            f"(Ra/Za/N/Q/S and calculator fields {sorted(ASE_PROPERTY_METHODS)}). "
            "Use identity maps (e.g. E: null or E: E), not pickle-style aliases "
            f"such as E: energy / Ra: coord. Invalid maps: {{{examples}}}. "
            "Custom row-data fields may still use non-identity maps."
        )

    def _missing_source_error(self, k: str, raw_data: Dict) -> KeyError:
        src = self.data_types[k]
        available = sorted(raw_data.keys()) if hasattr(raw_data, "keys") else []
        hint = ""
        if self.data_format == "aselmdb" and k in _ASELMDB_FIXED_KEYS and src != k:
            hint = (
                f" For data_format=aselmdb use an identity map "
                f"(e.g. {k}: null or {k}: {k}), not a pickle-style alias."
            )
        return KeyError(
            f"Requested field '{k}' (source '{src}') not found in raw data "
            f"(available: {available}).{hint}"
        )

    def _load_molecular_data(self, k: str, raw_data: Dict) -> None:
        if self.data_types[k] in raw_data.keys():
            values = raw_data[self.data_types[k]]
            if isinstance(values, int) or isinstance(values, float) or len(values) == 1:
                self.data.create_dataset(k, data=self._expand(k, values))
            elif len(values) == self.n_datapoint:
                self.data.create_dataset(k, data=self._compress(k, values))
            else:
                raise IndexError(f"Length of '{k}' should be n_datapoint or 1")
        elif (k + "a") in self.data_types and self.data_types[k + "a"] in raw_data.keys():
            self._load_atomic_data(k + "a", raw_data)
            # reduce atomic property into molecular property, mainly for Qa into Q
            logger.info(f"Molecular property {k} are reduced from atomic property {k + 'a'} ({self.data_types[k + 'a']})")
            if is_rounded(k):
                values = [round(sum(self.data[k + "a"][i][:self.data["N"][i % len(self.data["N"])]])) for i in tqdm(range(self.n_datapoint))]
            else:
                values = [sum(self.data[k + "a"][i][:self.data["N"][i % len(self.data["N"])]]) for i in tqdm(range(self.n_datapoint))]
            # don't compress summation of atomic property
            self.data.create_dataset(k, data=np.array(values))
        else:
            raise self._missing_source_error(k, raw_data)

    def _load_atomic_data(self, k: str, raw_data: Dict) -> None:
        if k in self.data:
            return
        src = self.data_types[k]
        if src not in raw_data.keys():
            raise self._missing_source_error(k, raw_data)
        values = raw_data[src]
        sample = values[0]
        if sample is None:
            raise ValueError(
                f"Atomic field '{k}' (source '{src}') returned None for the first "
                f"datapoint; check the feature/target map and dataset contents."
            )
        v0 = np.array(sample)
        if len(values) == self.n_datapoint:
            # for a datapoint, the shape of this property is (N, a, b, ...)
            # for the whole dataset, the shape of this property is (n_datapoint, max_N, a, b, ...)
            self.data.create_dataset(k, shape=(self.n_datapoint, self.max_N, *v0.shape[1:]), dtype=v0.dtype)
            logger.info(f"Storing atomic data {k} ({src})")
            for i, v in tqdm(enumerate(values), total=self.n_datapoint):
                if v is None:
                    raise ValueError(
                        f"Atomic field '{k}' (source '{src}') returned None at index {i}."
                    )
                self.data[k][i,:len(v)] = v

        elif len(values) == 1:
            self.data.create_dataset(k, data=self._expand(k, values))
        else:
            raise IndexError(f"Length of {k} ({src}) should be n_datapoint")

    def _init_data(self) -> None:
        suffix = self.data_path.split(".")[-1]
        # aselmdb accepts a file, directory of DB files, or glob; ASELMDBDataset validates the path
        if self.data_format == "aselmdb" or suffix == "aselmdb":
            self.data_format = "aselmdb"
            self._validate_aselmdb_field_maps()
            aselmdb_transforms = {}
            if self.preprocessings:
                aselmdb_transforms.update(self.preprocessings)
            if self.global_transforms:
                aselmdb_transforms.update(self.global_transforms)
            raw_data = ASELMDBDataset(
                self.data_path,
                connect_args=self.connect_args,
                select_args=self.select_args,
                transforms=aselmdb_transforms or None,
                declared_properties=self.data_types.keys(),
            )
            # Unit conversion is already applied in ASELMDBDataset; drop the transform so it is not applied twice.
            if aselmdb_transforms.get("energy_unit_conversion"):
                self.preprocessing.scales = [
                    s for s in self.preprocessing.scales
                    if not isinstance(s, EnergyUnitConversionTransform)
                ]
                self.global_transform.scales = [
                    s for s in self.global_transform.scales
                    if not isinstance(s, EnergyUnitConversionTransform)
                ]
            # ASELMDB Fa is already physical forces (get_forces); do not flip as if Fa were ∇E.
            if aselmdb_transforms.get("negative_gradient"):
                self.preprocessing.scales = [
                    s for s in self.preprocessing.scales
                    if not isinstance(s, NegativeGradientTransform)
                ]
                self.global_transform.scales = [
                    s for s in self.global_transform.scales
                    if not isinstance(s, NegativeGradientTransform)
                ]
        elif not os.path.isfile(self.data_path):
            raise ValueError(f"Data path {self.data_path} doesn't exist.")
        elif self.data_format == "hdf5" or suffix == "hdf5":
            self.data_format = "hdf5"
            raw_data = h5py.File(self.data_path, mode="r")["data"]
        elif self.data_format == "pickle" or suffix == "pkl" or suffix == "pickle":
            self.data_format = "pickle"
            raw_data = load_from_pickle(self.data_path)
        elif self.data_format == "npz" or suffix == "npz":
            self.data_format = "npz"
            raw_data = np.load(self.data_path, allow_pickle=True)
        elif self.data_format == "sdf" or suffix == "sdf":
            self.data_format = "sdf"
            raw_data = load_from_sdf(self.data_path)
        else:
            raise ValueError(f"Data format of {self.data_path} is unknown")

        if "Ra" not in self.data_types:
            # atomic position must be provided
            raise KeyError(f"Dataset must contain 'Ra' key (Atomic positions)")
        # number of datapoints is defined as number of different configurations
        n_datapoint = len(raw_data[self.data_types["Ra"]])
        self.n_datapoint = n_datapoint

        if "Za" not in self.data_types:
            # atomic number/type must be provided
            raise KeyError(f"Dataset must contain 'Za' key (Atomic numbers)")
        
        n_Za = len(raw_data[self.data_types["Za"]])
        if self.data_format == "pickle":
            Zas = parse_Za(raw_data[self.data_types["Za"]])
        else:
            Zas = raw_data[self.data_types["Za"]]
        if n_Za == 1:
            # list-of-dicts pickle yields Za as [Za_array]; unwrap to the shared topology
            Za_ref = Zas[0] if isinstance(Zas, list) else Zas
            Za_ref = np.asarray(Za_ref)
            if self.data_types["N"] not in raw_data.keys():
                # atom count determined by length of atomic numbers
                self.data.create_dataset("N", data=self._expand("N", int(len(Za_ref))))
            else:
                self._load_molecular_data("N", raw_data)
            self.data.create_dataset("Za", data=self._expand("Za", Za_ref))
            self.max_N = int(max(self.data["N"]))
        elif n_Za == n_datapoint:
            if self.data_types["N"] not in raw_data.keys():
                self.data.create_dataset("N", data=self._compress("N", [len(Za) for Za in Zas]))
            else:
                self._load_molecular_data("N", raw_data)
            self.max_N = max(self.data["N"])
            Za_compressed_flag = True
            Za0 = np.array(Zas[0])
            N0 = len(Za0)
            for Za in Zas:
                if len(Za) != N0 or (Za != Za0).any():
                    Za_compressed_flag = False
                    break
            if self.compressed and Za_compressed_flag:
                self.data.create_dataset("Za", data=[Za0])
            else:
                self.data.create_dataset("Za", shape=(n_datapoint, self.max_N), dtype=int)
                logger.info(f'Storing Za ({self.data_types["Za"]})')
                for i, Za in tqdm(enumerate(Zas), total=self.n_datapoint):
                    self.data["Za"][i,:len(Za)] = Za
        else:
            raise IndexError(f"Length of 'Za' should be n_datapoint or 1")
        
        for k in self.data_types:
            if k in ["Za", "N"]:
                continue
            elif (
                is_atomic(k)
                and k in UniformSplitQSTransform.POPULATED_KEYS
                and (
                    self._populate_uniform_qs_init
                    or self._populate_xtb_qs_prior
                    or self._populate_pyscf_nao_qs_prior
                )
            ):
                continue
            elif is_atomic(k):
                self._load_atomic_data(k, raw_data)
            else:
                self._load_molecular_data(k, raw_data)
        
        if self.data_format in ["hdf5", "npz"]:
            raw_data.close()

    def _init_neighbor_list(self) -> None:
        if self.neighbor_list_type == "full":
            from .neighbor_list import full_neighbor_list
            logger.info("producing neighbor list")
            if self.compressed and len(self.data["N"]) == 1:
                idx_i, idx_j = full_neighbor_list(self.data["N"][0])
                self.data.create_dataset("idx_i", data=[idx_i])
                self.data.create_dataset("idx_j", data=[idx_j])
                self.data.create_dataset("N_pair", data=[len(idx_i)])
            else:
                max_N_pairs = self.max_N * (self.max_N - 1)
                self.data.create_dataset("idx_i", shape=(self.n_datapoint, max_N_pairs), dtype=int)
                self.data.create_dataset("idx_j", shape=(self.n_datapoint, max_N_pairs), dtype=int)
                self.data.create_dataset("N_pair", shape=self.n_datapoint, dtype=int)
                for i in tqdm(range(self.n_datapoint)):
                    idx_i, idx_j = full_neighbor_list(self.data["N"][i])
                    self.data["N_pair"][i] = len(idx_i)
                    self.data["idx_i"][i] = array_padding([idx_i], max_N_pairs, pad_value=-1)
                    self.data["idx_j"][i] = array_padding([idx_j], max_N_pairs, pad_value=-1)

    def get_handle(self, mode: Literal["r", "w"]="r") -> None:
        if mode == "w" and os.path.exists(self.preload_path):
            logger.warning(f"Preload path {self.preload_path} exists and will be overwritten")
        else:
            os.makedirs(self.preload_path, exist_ok=True)
        self.file = h5py.File(os.path.join(self.preload_path, "pre_transformed.hdf5"), mode=mode, rdcc_nbytes=1024 ** 3 * self.max_memory)
        if mode == "r":
            self.data = self.file["data"]
        else:
            self.file.clear()
            self.data = self.file.create_group("data")

    def reset_handle(self):
        self.file.close()
        self.get_handle()

    def _save_config(self):
        handler = YamlHandler(os.path.join(self.preload_path, "datahub.yaml"))
        datahub_config = Dict({
            "feature": self.feature_types,
            "target": self.target_types,
            "preprocessings": self.preprocessings,
            "global_transforms": self.global_transforms,
            "neighbor_list": self.neighbor_list_type
        })
        handler.write_yaml(datahub_config)
        logger.info(f"Save preloaded dataset at {self.preload_path}")

    @property
    def features(self) -> FieldDataset:
        return FieldDataset({k: v for k, v in self.data.items() if k in self.feature_types.keys() | {"idx_i", "idx_j", "N_pair"}})
    
    @property
    def targets(self) -> FieldDataset:
        return FieldDataset({k: v for k, v in self.data.items() if k in self.target_types})


def _coerce_dataset_params(
    dataset_params: Dict[str, Any],
    global_transforms: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map YAML per-dataset ``transforms`` onto ``SingleDataHub.preprocessings``.

    Historical multi-dataset configs use ``transforms:`` under each dataset entry,
    but :class:`SingleDataHub` expects ``preprocessings``. Explicit ``preprocessings``
    wins on key conflicts with remapped ``transforms``.

    Keys already present in ``global_transforms`` are omitted from the remapped
    preprocessings so overlapping AL configs (same dict under both) are not
    applied twice. When values differ, ``global_transforms`` wins and a warning
    is logged.
    """
    params = dict(dataset_params)
    transforms = params.pop("transforms", None)
    if transforms:
        existing = params.get("preprocessings")
        if existing:
            merged = dict(transforms)
            merged.update(existing)
            params["preprocessings"] = merged
        else:
            params["preprocessings"] = dict(transforms)
    preprocessings = params.get("preprocessings")
    if preprocessings and global_transforms:
        kept = {}
        for key, value in preprocessings.items():
            if key in global_transforms:
                if global_transforms[key] != value:
                    logger.warning(
                        "Dropping per-dataset preprocessing %r=%r because "
                        "global_transforms already defines %r=%r; global wins "
                        "to avoid double application.",
                        key,
                        value,
                        key,
                        global_transforms[key],
                    )
                continue
            kept[key] = value
        params["preprocessings"] = kept or None
    return params


class DataHub:
    def __init__(self,  
        dump_dir=".",
        datasets: Optional[Union[List, Dict]]=None,
        fields: Optional[Dict[str, str]]=None,
        **params
    ):
        self.dump_dir = dump_dir
        if fields is not None:
            for k, v in fields.items():
                register_data_type(k, **v)
        if datasets is None:
            if "global_transforms" not in params:
                params["global_transforms"] = params.get("transforms", None)
            self.datahubs = {"default": SingleDataHub(dump_dir=dump_dir, **params)}
        elif isinstance(datasets, list):
            global_transforms = params.get("global_transforms", None)
            self.datahubs = {
                str(i): SingleDataHub(
                    dump_dir=dump_dir,
                    global_transforms=global_transforms,
                    **_coerce_dataset_params(dataset_params, global_transforms),
                )
                for i, dataset_params in enumerate(datasets)
            }
        elif isinstance(datasets, dict):
            global_transforms = params.get("global_transforms", None)
            self.datahubs = {
                name: SingleDataHub(
                    dump_dir=dump_dir,
                    global_transforms=global_transforms,
                    **_coerce_dataset_params(dataset_params, global_transforms),
                )
                for name, dataset_params in datasets.items()
            }
        else:
            raise ValueError(f"Unknown type of datasets: {type(datasets)}")
            
    @property
    def features(self) -> Dict[str, FieldDataset]:
        return {name: datahub.features for name, datahub in self.datahubs.items()}

    @property
    def targets(self) -> Dict[str, FieldDataset]:
        return {name: datahub.targets for name, datahub in self.datahubs.items()}
    
    @property
    def preload_path(self) -> Dict[str, str]:
        return {name: datahub.preload_path for name, datahub in self.datahubs.items()}
    
    @property
    def transform(self) -> Transform:
        return list(self.datahubs.values())[0].global_transform
