"""Irreps schedules for EquFlashV2.

The published materials models list every layer explicitly (scalar input,
full natural-parity hidden irreps, scalar output). ``equflash_schedule``
builds that same pattern and drops hidden irreps that cannot reach the next
layer by a tensor product with the edge spherical harmonics.
"""

from __future__ import annotations

from typing import List, Sequence, Union

from e3nn import o3


def spherical_harmonics_irreps(lmax: int, parity: bool) -> o3.Irreps:
    """Edge filter irreps. ``parity`` keeps the natural ``(-1)^l`` signs."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    return o3.Irreps.spherical_harmonics(lmax, -1 if parity else 1)


def scale_irreps(irreps: Union[str, o3.Irreps], scale: float) -> o3.Irreps:
    """Multiply every multiplicity by ``scale`` (EquFlash ``irreps_scale``)."""
    factor = float(scale)
    if factor <= 0:
        raise ValueError(f"irreps scale must be positive, got {scale}")
    scaled = []
    for mul, ir in o3.Irreps(irreps):
        channels = int(round(mul * factor))
        if channels < 1:
            raise ValueError(f"scaled multiplicity of {ir} is {channels}")
        scaled.append((channels, ir))
    return o3.Irreps(scaled).simplify()


def _reaches(ir: o3.Irrep, edge: o3.Irreps, target: o3.Irreps) -> bool:
    wanted = {out for _, out in target}
    for _, ir_edge in edge:
        for produced in ir * ir_edge:
            if produced in wanted:
                return True
    return False


def equflash_schedule(
    channels: int,
    lmax: int,
    num_layers: int,
    parity: bool = True,
) -> List[o3.Irreps]:
    """Scalar, then ``num_layers - 1`` equivariant layers, then scalar again.

    Length is ``num_layers + 1``. Hidden irreps that have no Clebsch-Gordan
    path into the following layer are removed, walking backward from the
    scalar output.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    edge = spherical_harmonics_irreps(lmax, parity)
    scalar = o3.Irreps(f"{int(channels)}x0e")
    schedule: List[o3.Irreps] = [scalar]
    current = scalar
    for _ in range(num_layers):
        produced = set()
        for _, ir in current:
            for _, ir_edge in edge:
                for ir_out in ir * ir_edge:
                    if ir_out.l <= lmax and (parity or ir_out.p == 1):
                        produced.add(ir_out)
        ordered = sorted(produced, key=lambda ir: (ir.l, -ir.p))
        if not ordered:
            raise ValueError("tensor product with edge spherical harmonics is empty")
        current = o3.Irreps([(int(channels), ir) for ir in ordered]).simplify()
        schedule.append(current)
    schedule[-1] = scalar
    target = schedule[-1]
    for index in range(num_layers - 1, 0, -1):
        kept = [
            (mul, ir)
            for mul, ir in schedule[index]
            if _reaches(ir, edge, target)
        ]
        if not kept:
            raise ValueError(f"layer {index} has no irreps that reach the next layer")
        schedule[index] = o3.Irreps(kept).simplify()
        target = schedule[index]
    return schedule


def parse_schedule(
    irreps_manual: Sequence[Union[str, o3.Irreps]],
) -> List[o3.Irreps]:
    schedule = [o3.Irreps(item).simplify() for item in irreps_manual]
    if len(schedule) < 2:
        raise ValueError("irreps_manual needs at least an input and an output")
    return schedule
