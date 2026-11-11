"""Irrep schedules shared by EquFlash and EquFlashV2.

``spherical_harmonics_irreps`` is the edge filter for both Cores.
``equflash_irreps`` is the EquFlash list (scalar, natural-parity hidden
layers, scalar), including the ``closure`` fallback from ``generate_irreps``.
``equflash_schedule`` is the EquFlashV2 list: open the tensor product with
that filter, force a scalar output, then drop hidden irreps that cannot reach
the next layer. The scalar embedding itself is left in place.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union

from e3nn import o3


def spherical_harmonics_irreps(lmax: int, parity: bool) -> o3.Irreps:
    """Edge irreps. ``parity`` keeps natural ``(-1)^l``; otherwise every l is even."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    return o3.Irreps.spherical_harmonics(lmax, -1 if parity else 1)


def scale_irreps(irreps: Union[str, o3.Irreps], scale: float) -> o3.Irreps:
    """Multiply every multiplicity by ``scale`` (EquFlashV2 ``irreps_scale``)."""
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


def _prune_backward(
    schedule: List[o3.Irreps],
    edge: o3.Irreps,
    *,
    lowest_index: int,
    strict: bool,
) -> None:
    """Drop irreps with no Clebsch-Gordan path into the following layer.

    ``lowest_index`` is the first schedule entry that may be pruned. EquFlash
    closure includes the input (0). EquFlashV2 keeps the scalar embedding (1).
    """
    target = schedule[-1]
    for index in range(len(schedule) - 2, lowest_index - 1, -1):
        kept = [
            (mul, ir)
            for mul, ir in schedule[index]
            if _reaches(ir, edge, target)
        ]
        if strict and not kept:
            raise ValueError(f"layer {index} has no irreps that reach the next layer")
        target = o3.Irreps(kept).simplify()
        schedule[index] = target


def _natural_hidden(channel: int, lmax: int, parity: bool) -> o3.Irreps:
    return o3.Irreps(
        [(channel, ir) for _, ir in spherical_harmonics_irreps(lmax, parity)]
    )


def _closure_schedule(
    channel: int, lmax: int, num_layers: int, parity: bool
) -> List[o3.Irreps]:
    edge = spherical_harmonics_irreps(lmax, parity)
    schedule: List[o3.Irreps] = [o3.Irreps(f"{channel}x0e")]
    for _ in range(num_layers):
        # Irrep types do not depend on multiplicity. A unit-multiplicity
        # product avoids allocating the channel-sized FullTensorProduct.
        unit = o3.Irreps([(1, ir) for _, ir in schedule[-1]]).simplify()
        product = o3.FullTensorProduct(unit, edge)
        opened = [
            (channel, ir)
            for _, ir in product.irreps_out.simplify()
            if ir.l <= lmax
        ]
        schedule.append(o3.Irreps(opened))

    schedule[-1] = schedule[0]
    _prune_backward(schedule, edge, lowest_index=0, strict=False)
    return schedule


def equflash_irreps(
    channel: int,
    lmax: int,
    num_layers: int,
    parity: bool = True,
    irreps_manual: Optional[Sequence[str]] = None,
    schedule: Literal["natural", "closure"] = "natural",
) -> List[o3.Irreps]:
    """Return ``num_layers + 1`` irreps, from the scalar embedding to the output.

    ``irreps_manual`` replaces the generated schedule. Its length is the
    number of convolution layers plus the input irreps. ``schedule="natural"``
    is the released EquFlash list. ``schedule="closure"`` is ``generate_irreps``.
    """
    if irreps_manual is not None:
        parsed = [o3.Irreps(text) for text in irreps_manual]
        if len(parsed) < 2:
            raise ValueError("irreps_manual needs the input irreps and at least one layer")
        return parsed

    if channel < 1:
        raise ValueError(f"channel must be >= 1, got {channel}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if schedule == "closure":
        return _closure_schedule(channel, lmax, num_layers, parity)
    if schedule != "natural":
        raise ValueError(
            f"Unknown EquFlash irrep schedule {schedule!r}; expected 'natural' or 'closure'"
        )

    scalar = o3.Irreps(f"{channel}x0e")
    if num_layers == 1:
        return [scalar, scalar]
    hidden = _natural_hidden(channel, lmax, parity)
    return [scalar, *([hidden] * (num_layers - 1)), scalar]


def equflash_schedule(
    channels: int,
    lmax: int,
    num_layers: int,
    parity: bool = True,
) -> List[o3.Irreps]:
    """Scalar, then ``num_layers - 1`` grown layers, then scalar again.

    Length is ``num_layers + 1``. Hidden irreps that have no Clebsch-Gordan
    path into the following layer are removed, walking backward from the
    scalar output. The input embedding is not pruned.
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
    _prune_backward(schedule, edge, lowest_index=1, strict=True)
    return schedule


def parse_schedule(
    irreps_manual: Sequence[Union[str, o3.Irreps]],
) -> List[o3.Irreps]:
    schedule = [o3.Irreps(item).simplify() for item in irreps_manual]
    if len(schedule) < 2:
        raise ValueError("irreps_manual needs at least an input and an output")
    return schedule
