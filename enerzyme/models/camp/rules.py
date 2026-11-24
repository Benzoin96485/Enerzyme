"""Construction rules for CAMP atomic moments and hyper moments.

Atomic-moment paths are Eq. (5) with the supplement's two constraints:
``v1 <= v2`` and the feature indices are fully contracted, so the output rank
is ``v2 - v1``. Hyper-moment paths are Eq. (7): every factor except one copy
of the highest rank is fully contracted into that tensor, and scalar moments
are not multiplied in. Rank 0 additionally keeps a copy of ``M_0``.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def atomic_paths(max_dyad: int, max_feature: int, out_rank: int) -> tuple:
    """Pairs ``(v1, v2)`` whose contraction has rank ``out_rank``."""
    if out_rank < 0 or out_rank > max_dyad:
        return tuple()
    paths = []
    for feature_rank in range(max_dyad - out_rank + 1):
        dyad_rank = feature_rank + out_rank
        if feature_rank <= max_feature and dyad_rank <= max_dyad:
            paths.append((feature_rank, dyad_rank))
    return tuple(paths)


@lru_cache(maxsize=None)
def hyper_moment_rules(max_v: int, out_rank: int) -> tuple:
    """Non-decreasing rank tuples that contract to ``out_rank``.

    A tuple ``(v1, ..., vn)`` uses one copy of each listed atomic moment.
    The highest rank is the tensor that keeps the free indices.
    """
    if max_v < 0 or out_rank < 0 or out_rank > max_v:
        return tuple()
    budget = 2 * max_v
    found = []

    def walk(start: int, chosen: list, total: int) -> None:
        if chosen:
            peak = chosen[-1]
            if peak == total - peak + out_rank:
                found.append(tuple(chosen))
        for rank in range(start, max_v + 1):
            if total + rank > budget:
                break
            chosen.append(rank)
            walk(rank, chosen, total + rank)
            chosen.pop()

    walk(1, [], 0)
    if out_rank == 0:
        return ((0,), *found)
    return tuple(found)
