"""UMA wrappers and the native eSEN Core.

``UMAWrapperQS`` and ``UMAFlowWrapperQS`` (architectures ``uma_qs`` and
``uma_flow_qs``) load fairchem the first time they are accessed. They wrap
a UMA / eSCN-MD checkpoint. They are not the 2023 paper eSCN and they are
not the native eSEN network.

``eSENCore`` (architecture ``esen``) is Fu et al., 2025
(arXiv:2502.12147). It is implemented in :mod:`enerzyme.models.esen.native`
on the shared SO(2) / SO(3) modules and does not import fairchem.
"""

from .native import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_30M_CORE_PARAMS,
    PUBLISHED_MP_3M_CORE_PARAMS,
    PUBLISHED_SPICE_6M_CORE_PARAMS,
    eSENCore,
)

_UMA_EXPORTS = {
    "UMAWrapperQS": (".core", "UMAWrapperQS"),
    "UMAFlowWrapperQS": (".core", "UMAFlowWrapperQS"),
    "ESCNMDBackboneFlow": (".flow_umabackbone", "ESCNMDBackboneFlow"),
    "ESCNMDMoeBackboneFlow": (".flow_umabackbone", "ESCNMDMoeBackboneFlow"),
    "build_flow_backbone": (".flow_umabackbone", "build_flow_backbone"),
}


def __getattr__(name: str):
    if name not in _UMA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = _UMA_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_30M_CORE_PARAMS",
    "PUBLISHED_MP_3M_CORE_PARAMS",
    "PUBLISHED_SPICE_6M_CORE_PARAMS",
    "eSENCore",
    "UMAWrapperQS",
    "UMAFlowWrapperQS",
    "ESCNMDBackboneFlow",
    "ESCNMDMoeBackboneFlow",
    "build_flow_backbone",
]
