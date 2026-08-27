from typing import Any

import numpy as np

IS_INT = 1
IS_ROUNDED = 2
IS_ATOMIC = 4
REQUIRES_GRAD = 8
IS_IDX = 16
IS_TARGET = 32
IS_GRAD = 64
TENSOR_RANK_BIT = 8
TENSOR_RANK_BASE = 2 << TENSOR_RANK_BIT
DATA_TYPES = {
    "N": IS_INT | IS_IDX,
    "Za": IS_INT | IS_ATOMIC,
    "Ra": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | REQUIRES_GRAD,
    "Q": IS_ROUNDED | IS_TARGET,
    "Qa": IS_ATOMIC | IS_TARGET,
    "Q_init_a": IS_ATOMIC,
    "Q_flow_a": IS_ATOMIC,
    "S": IS_ROUNDED | IS_TARGET,
    "Sa": IS_ATOMIC | IS_TARGET,
    "S_init_a": IS_ATOMIC,
    "S_flow_a": IS_ATOMIC,
    "E": IS_TARGET,
    "Fa": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET | IS_GRAD,
    "dE": IS_TARGET,
    "dFa": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET | IS_GRAD,
    "dielectric": 0,
    "M2": TENSOR_RANK_BASE * 1 | IS_TARGET,
    "M2a": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "idx_i": IS_INT | IS_IDX,
    "idx_j": IS_INT | IS_IDX,
    "N_pair": IS_INT | IS_IDX,
    # E2Former-LSR / BRICS fragment labels (optional Datahub fields)
    "cluster_ids": IS_INT | IS_ATOMIC,
    "cluster_centers": IS_ATOMIC | (TENSOR_RANK_BASE * 1),
}

DIELECTRIC_FIELD = "dielectric"
MIN_DIELECTRIC = 1.0

def is_int(k):
    return bool(DATA_TYPES.get(k, 0) & IS_INT)

def is_rounded(k):
    return bool(DATA_TYPES.get(k, 0) & IS_ROUNDED)

def is_atomic(k):
    return bool(DATA_TYPES.get(k, 0) & IS_ATOMIC)

def requires_grad(k):
    return bool(DATA_TYPES.get(k, 0) & REQUIRES_GRAD)

def is_idx(k):
    return bool(DATA_TYPES.get(k, 0) & IS_IDX)

def is_target(k):
    return bool(DATA_TYPES.get(k, 0) & IS_TARGET)

def is_target_uq(k):
    if k.endswith("_var") or k.endswith("_std"):
        target = k[:-4]
        return is_target(target)
    return False

def is_grad(k):
    if bool(DATA_TYPES.get(k, 0) & IS_GRAD) or k.endswith("_grad"):
        return True
    return False

def get_tensor_rank(k):
    return bool(DATA_TYPES.get(k, 0) >> TENSOR_RANK_BIT)

TYPE_ATTRS = {
    "is_atomic": IS_ATOMIC,
}

def register_data_type(k, **type_info):
    DATA_TYPES[k] = 0
    for type_attr, v in type_info.items():
        if v is True:
            DATA_TYPES[k] |= TYPE_ATTRS[type_attr]
        else:
            DATA_TYPES[k] &= ~TYPE_ATTRS[type_attr]


def validate_dielectric_array(value: Any, *, source: str = DIELECTRIC_FIELD) -> np.ndarray:
    """Return a 1-D float64 array of ε values, rejecting missing / non-finite / ``ε < 1``."""
    if value is None:
        raise ValueError(
            f"{source} is missing; required graph-level dielectric ε >= {MIN_DIELECTRIC:g}"
        )
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(
            f"{source} is empty; required graph-level dielectric ε >= {MIN_DIELECTRIC:g}"
        )
    flat = np.reshape(arr, -1)
    if not np.isfinite(flat).all():
        raise ValueError(f"{source} contains NaN or Inf; dielectric ε must be finite and >= {MIN_DIELECTRIC:g}")
    if np.any(flat < MIN_DIELECTRIC):
        raise ValueError(
            f"{source} must be >= {MIN_DIELECTRIC:g}, got min {float(np.min(flat)):g}"
        )
    return flat


def validate_dielectric_value(value: Any, *, source: str = DIELECTRIC_FIELD) -> float:
    """Validate a single graph-level dielectric constant."""
    flat = validate_dielectric_array(value, source=source)
    if flat.size != 1:
        raise ValueError(f"{source} must be a scalar dielectric ε, got shape {flat.shape}")
    return float(flat[0])


def dielectric_from_atoms(atoms: Any) -> float:
    """Read ``atoms.info['dielectric']`` and reject missing, NaN, or ``ε < 1``."""
    info = getattr(atoms, "info", None)
    if info is None or DIELECTRIC_FIELD not in info:
        raise ValueError(
            "atoms.info['dielectric'] is missing; required graph-level dielectric ε >= "
            f"{MIN_DIELECTRIC:g}"
        )
    return validate_dielectric_value(
        info[DIELECTRIC_FIELD], source="atoms.info['dielectric']"
    )


def validate_dielectric_tensor(dielectric, *, source: str = DIELECTRIC_FIELD):
    """Validate a torch dielectric tensor and return a 1-D float tensor."""
    import torch
    from torch import Tensor

    if dielectric is None:
        raise ValueError(
            f"{source} is missing; required graph-level dielectric ε >= {MIN_DIELECTRIC:g}"
        )
    if not torch.is_tensor(dielectric):
        dielectric = torch.as_tensor(dielectric)
    if dielectric.numel() == 0:
        raise ValueError(
            f"{source} is empty; required graph-level dielectric ε >= {MIN_DIELECTRIC:g}"
        )
    flat: Tensor = dielectric.reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError(
            f"{source} contains NaN or Inf; dielectric ε must be finite and >= {MIN_DIELECTRIC:g}"
        )
    if (flat < MIN_DIELECTRIC).any():
        raise ValueError(
            f"{source} must be >= {MIN_DIELECTRIC:g}, got min {float(flat.min().item()):g}"
        )
    return flat


__all__ = [
    "is_int",
    "is_rounded",
    "is_atomic",
    "requires_grad",
    "is_idx",
    "get_tensor_rank",
    "is_target",
    "is_target_uq",
    "register_data_type",
    "is_grad",
    "DIELECTRIC_FIELD",
    "MIN_DIELECTRIC",
    "validate_dielectric_array",
    "validate_dielectric_value",
    "dielectric_from_atoms",
    "validate_dielectric_tensor",
]