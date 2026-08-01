"""SO3LR + Euclidean Fast Attention defaults (architecture: so3lr_efa).

Same universal pairwise FF stack as ``so3lr``, with EFA enabled on the shared
``So3kratesCore``.
"""

from __future__ import annotations

import math
from copy import deepcopy

from .so3lr import DEFAULT_BUILD_PARAMS as _SO3LR_BUILD
from .so3lr import DEFAULT_LAYER_PARAMS as _SO3LR_LAYERS

_SO3LR_AVG_NUM_NEIGHBORS = 13.168995780096482

DEFAULT_BUILD_PARAMS = deepcopy(_SO3LR_BUILD)

DEFAULT_LAYER_PARAMS = deepcopy(_SO3LR_LAYERS)

# Enable EFA on Core params inside the copied layer stack.
for _layer in DEFAULT_LAYER_PARAMS:
    if _layer.get("name") == "Core":
        _params = _layer.setdefault("params", {})
        _params.update(
            {
                "era_use_in_iterations": [0, 1],
                "era_lebedev_num": 146,
                "era_max_frequency": float(3 * math.pi),
                "era_max_length": 12.0,
                "era_qk_num_features": 32,
                "era_v_num_features": 16,
                "avg_num_neighbors": _SO3LR_AVG_NUM_NEIGHBORS,
            }
        )
        break
