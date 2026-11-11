"""Irrep schedule for Allegro's tensor track.

The forward pass keeps every allowed irrep that a Clebsch–Gordan product with
the spherical-harmonic environment can build. A backward pass then drops
irreps that cannot reach the final even scalars. The last layer emits ``0e``
only, and every retained output starts with ``0e`` so the scalar channels are
a prefix of the flat e3nn layout.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from e3nn import o3

from ..nequip.interaction import tp_path_exists


def edge_sh_irreps(l_max: int) -> o3.Irreps:
    """Real spherical harmonics through ``l_max`` with parity ``(-1)^l``."""
    if int(l_max) < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    return o3.Irreps.spherical_harmonics(int(l_max), p=-1)


def allowed_tensor_irreps(l_max: int, parity: bool) -> o3.Irreps:
    """Irreps the tensor track may carry.

    ``parity=True`` keeps both mirror parities at every ``l`` (the Allegro
    default). ``parity=False`` keeps the spherical-harmonic parity only.
    """
    edge = edge_sh_irreps(l_max)
    if not parity:
        return edge
    irreps = [
        (1, (ell, mirror))
        for ell in range(int(l_max) + 1)
        for mirror in (1, -1)
    ]
    return o3.Irreps(irreps)


def allegro_layer_irreps(
    l_max: int,
    parity: bool,
    num_layers: int,
) -> List[Tuple[o3.Irreps, o3.Irreps]]:
    """``(input, output)`` irreps of multiplicity 1 for each tensor product.

    The environment factor is always the spherical-harmonic irreps. Returned
    irreps are not simplified, so ``0e`` stays the first output block.
    """
    if int(num_layers) < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    environment = edge_sh_irreps(l_max)
    allowed = allowed_tensor_irreps(l_max, parity)
    scalar = o3.Irreps([(1, (0, 1))])
    schedule: List[o3.Irreps] = [environment]
    current = environment
    for layer in range(int(num_layers)):
        target = scalar if layer == int(num_layers) - 1 else allowed
        produced = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in target
                if tp_path_exists(current, environment, ir)
            ]
        )
        if len(produced) == 0:
            raise ValueError(
                f"Allegro layer {layer} has no tensor-product paths from "
                f"{current} x {environment}"
            )
        if produced[0].ir != o3.Irrep("0e"):
            raise ValueError(
                f"Allegro layer {layer} must expose 0e first, got {produced}"
            )
        schedule.append(produced)
        current = produced

    pruned = _prune_unreachable(schedule, environment)
    return list(zip(pruned[:-1], pruned[1:]))


def _prune_unreachable(
    schedule: Sequence[o3.Irreps],
    environment: o3.Irreps,
) -> List[o3.Irreps]:
    """Drop irreps that have no path into the final scalars."""
    kept: List[o3.Irreps] = [schedule[-1]]
    downstream = schedule[-1]
    for upstream in reversed(list(schedule[:-1])):
        useful = []
        for mul, ir in upstream:
            if _reaches(ir, environment, downstream):
                useful.append((mul, ir))
        if not useful:
            raise ValueError(
                f"Allegro pruning removed every irrep from {upstream} "
                f"on the way to {downstream}"
            )
        downstream = o3.Irreps(useful)
        if downstream[0].ir != o3.Irrep("0e"):
            raise ValueError(
                f"pruned tensor irreps must still start with 0e, got {downstream}"
            )
        kept.append(downstream)
    kept.reverse()
    return kept


def _reaches(ir: o3.Irrep, environment: o3.Irreps, downstream: o3.Irreps) -> bool:
    for _, env_ir in environment:
        for coupled in ir * env_ir:
            if coupled in downstream:
                return True
    return False
