"""Irrep schedule for EquFlash interaction layers.

The released EquFlash config (``irreps_manual`` in SamsungDS/GGNN) keeps every
hidden layer on natural parity, ``channel x (0e + 1o + … + lmax)``, with
scalar features on the input and the output. ``schedule="closure"`` is the
package's ``generate_irreps`` fallback: open every tensor product up to
``lmax``, then drop irreps that cannot reach the scalar output.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from e3nn import o3


def spherical_harmonics_irreps(lmax: int, parity: bool) -> o3.Irreps:
    """Edge irreps: ordinary Y_lm, or all-even parity when ``parity`` is false."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    return o3.Irreps(
        [
            (1, (l, 1 if (l % 2 == 0 or not parity) else -1))
            for l in range(lmax + 1)
        ]
    )


def _natural_hidden(channel: int, lmax: int, parity: bool) -> o3.Irreps:
    return o3.Irreps(
        [
            (channel, (l, 1 if (l % 2 == 0 or not parity) else -1))
            for l in range(lmax + 1)
        ]
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
    target = schedule[-1]
    for step in range(num_layers - 1):
        incoming = schedule[-step - 2]
        reachable = {ir for _, ir in target}
        kept = []
        for mul, ir_in in incoming:
            produces_target = False
            for _, ir_edge in edge:
                for ir_out in ir_in * ir_edge:
                    if ir_out in reachable:
                        produces_target = True
                        break
                if produces_target:
                    break
            if produces_target:
                kept.append((mul, ir_in))
        target = o3.Irreps(kept)
        schedule[-step - 2] = target
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
