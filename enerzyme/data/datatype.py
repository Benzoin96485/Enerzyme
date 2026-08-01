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
    "M2": TENSOR_RANK_BASE * 1 | IS_TARGET,
    "M2a": IS_ATOMIC | (TENSOR_RANK_BASE * 1) | IS_TARGET,
    "idx_i": IS_INT | IS_IDX,
    "idx_j": IS_INT | IS_IDX,
    "N_pair": IS_INT | IS_IDX
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
        

__all__ = ["is_int", "is_rounded", "is_atomic", "requires_grad", "is_idx", "get_tensor_rank", "is_target", "is_target_uq", "register_data_type", "is_grad"]