"""Species vocabulary of the published Nequix MPtrj checkpoints.

``configs/nequix-mp-1.yml`` lists atomic numbers 1..94 except 84..88.
"""

from __future__ import annotations

from typing import Tuple


# 94 elements in 1..94, minus Po, At, Rn, Fr, Ra.
MPTRJ_ATOMIC_NUMBERS: Tuple[int, ...] = tuple(
    atomic_number for atomic_number in range(1, 95) if atomic_number not in range(84, 89)
)
