IS_INT = 1
IS_ROUNDED = 2
IS_ATOMIC = 4
REQUIRES_GRAD = 8
IS_IDX = 16
IS_TARGET = 32
TENSOR_RANK_BIT = 8
TENSOR_RANK_BASE = 2 << TENSOR_RANK_BIT
DATA_TYPES = {
    "N": IS_INT | IS_IDX,
    "Za": IS_INT | IS_ATOMIC,
    "Ra": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | REQUIRES_GRAD,
    "Q": IS_ROUNDED | IS_TARGET,
    "Qa": IS_ATOMIC | IS_TARGET,
    "S": IS_ROUNDED | IS_TARGET,
    "E": IS_TARGET,
    "Fa": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "M2": TENSOR_RANK_BASE * 1 | IS_TARGET,
    "M2a": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "idx_i": IS_INT | IS_IDX,
    "idx_j": IS_INT | IS_IDX,
    "N_pair": IS_INT | IS_IDX,
}

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

def get_tensor_rank(k):
    return bool(DATA_TYPES.get(k, 0) >> TENSOR_RANK_BIT)

__all__ = ["is_int", "is_rounded", "is_atomic", "requires_grad", "is_idx", "get_tensor_rank", "is_target", "is_target_uq"]