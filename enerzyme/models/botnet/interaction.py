"""BOTNet interaction choices.

The convolutions are the design-space blocks in
:mod:`enerzyme.models.mace.interaction`. They were ported from
`gncs/botnet <https://github.com/gncs/botnet>`_ and are the layers MACE kept
when it added the symmetric product. BOTNet stops before that product: each
block returns the updated node state, including its species tensor-product
skip.

``idx_i`` is the ASE sender and ``idx_j`` is the ASE receiver, matching those
blocks and ``vij = R_j - R_i``. Messages are summed onto ``idx_j``. NequIP and
PaiNN gather the neighbor onto ``idx_i``; these blocks do not.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from torch.nn import Module

from ..mace.interaction import (
    AgnosticNonlinearInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    ResidualElementDependentInteractionBlock,
)


INTERACTION_CLASSES: Dict[str, Type[Module]] = {
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
}

# First layer, then every later layer. Names match the gncs/botnet training
# commands: agnostic scale-shifted BOTNet, element-dependent radial weights
# after the first layer, and a residual element-dependent block on every layer.
VARIANTS: Dict[str, Tuple[str, str]] = {
    "agnostic": (
        "AgnosticNonlinearInteractionBlock",
        "AgnosticResidualNonlinearInteractionBlock",
    ),
    "element": (
        "AgnosticNonlinearInteractionBlock",
        "ResidualElementDependentInteractionBlock",
    ),
    "residual_element": (
        "ResidualElementDependentInteractionBlock",
        "ResidualElementDependentInteractionBlock",
    ),
}
