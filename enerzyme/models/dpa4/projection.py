"""S² grid projection for DPA4 FFN — thin wrapper over shared ``so3.lebedev``.

Prefer importing ``S2LebedevProjector`` from ``enerzyme.models.so3`` directly
for new code. This module keeps the historical ``S2GridProjector`` name used
by DPA4 FFN.
"""

from __future__ import annotations

from typing import Optional

from ..so3 import S2LebedevProjector

# Re-export shared helpers for local DPA4 imports / tests.
from ..so3.lebedev import (  # noqa: F401
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
    resolve_lebedev_precision,
)


class S2GridProjector(S2LebedevProjector):
    """Alias of :class:`~enerzyme.models.so3.lebedev.S2LebedevProjector`."""

    def __init__(self, lmax: int, mmax: Optional[int] = None) -> None:
        # ``mmax`` is accepted for API compatibility with earlier DPA4 drafts;
        # Lebedev S² projection always uses the full packed ``lmax`` basis.
        del mmax
        super().__init__(lmax=lmax)
