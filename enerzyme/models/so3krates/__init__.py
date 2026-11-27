# Copyright (c) So3krates authors (Frank et al., NeurIPS 2022).
# Ported from So3krates-torch / mlff (MIT License).

"""So3krates package — Core exports are lazy to avoid import cycles."""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [

    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "SO3LR_DEFAULT_BUILD_PARAMS",
    "SO3LR_DEFAULT_LAYER_PARAMS",
]


def __getattr__(name: str):
    if name == "SO3LR_DEFAULT_BUILD_PARAMS":
        from .so3lr import DEFAULT_BUILD_PARAMS as SO3LR_DEFAULT_BUILD_PARAMS

        return SO3LR_DEFAULT_BUILD_PARAMS
    if name == "SO3LR_DEFAULT_LAYER_PARAMS":
        from .so3lr import DEFAULT_LAYER_PARAMS as SO3LR_DEFAULT_LAYER_PARAMS

        return SO3LR_DEFAULT_LAYER_PARAMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
