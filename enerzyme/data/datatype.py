IS_INT = 1
IS_ROUNDED = 2
IS_ATOMIC = 4
REQUIRES_GRAD = 8
IS_PAIR_IDX = 16
IS_TARGET = 32
TENSOR_RANK_BIT = 8
TENSOR_RANK_BASE = 2 << TENSOR_RANK_BIT
DATA_TYPES = {
    "N": IS_INT | IS_PAIR_IDX,
    "Za": IS_INT | IS_ATOMIC,
    "Ra": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | REQUIRES_GRAD,
    "Q": IS_ROUNDED | IS_TARGET,
    "Qa": IS_ATOMIC | IS_TARGET,
    "S": IS_ROUNDED | IS_TARGET,
    "E": IS_TARGET,
    "Fa": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "M2": TENSOR_RANK_BASE * 1 | IS_TARGET,
    "M2a": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "idx_i": IS_INT | IS_PAIR_IDX,
    "idx_j": IS_INT | IS_PAIR_IDX,
    "N_pair": IS_INT | IS_PAIR_IDX,
    "nh_loss": 0
}

def is_int(k):
    return bool(DATA_TYPES[k] & IS_INT)

def is_rounded(k):
    return bool(DATA_TYPES[k] & IS_ROUNDED)

def is_atomic(k):
    return bool(DATA_TYPES[k] & IS_ATOMIC)

def requires_grad(k):
    return bool(DATA_TYPES[k] & REQUIRES_GRAD)

def is_pair_idx(k):
    return bool(DATA_TYPES[k] & IS_PAIR_IDX)

def is_target(k):
    return bool(DATA_TYPES[k] & IS_TARGET)

def get_tensor_rank(k):
    return bool(DATA_TYPES[k] >> TENSOR_RANK_BIT)

__all__ = ["is_int", "is_rounded", "is_atomic", "requires_grad", "is_pair_idx", "get_tensor_rank", "is_target"]