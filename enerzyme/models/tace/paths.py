"""CG path generation for TACE tensor products.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from e3nn import o3


def satisfy(l1: int, l2: int, restriction: Optional[str] = None) -> bool:
    if restriction is None:
        return True
    if restriction == "<":
        return l1 < l2
    if restriction == "<=":
        return l1 <= l2
    if restriction == ">":
        return l1 > l2
    if restriction == ">=":
        return l1 >= l2
    if restriction == "==":
        return l1 == l2
    if restriction == "!=":
        return l1 != l2
    raise ValueError(f"Unknown restriction: {restriction}")


def generate_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    *,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
    e3nn_mode: str = "uvu",
    trainable: bool = False,
    identical_inputs: bool = False,
) -> Tuple[List[Tuple[int, int, int, str, bool]], o3.Irreps]:
    if identical_inputs and irreps_in1 != irreps_in2:
        raise ValueError("identical_inputs requires matching input irreps")

    e3nn_paths: List[Tuple[int, int, int, str, bool]] = []
    e3nn_out_irreps: List[Tuple[int, o3.Irrep]] = []

    for _, ir_out in irreps_out:
        for i, (mul, ir1) in enumerate(irreps_in1):
            for j, (_, ir2) in enumerate(irreps_in2):
                l1, l2, l3 = ir1.l, ir2.l, ir_out.l
                if (
                    ir_out in ir1 * ir2
                    and satisfy(l1, l2, l1l2)
                    and satisfy(l2, l3, l2l3)
                    and satisfy(l3, l1, l3l1)
                ):
                    if identical_inputs and i == j and (l1 + l2 - l3) % 2 == 1:
                        continue
                    k = len(e3nn_out_irreps)
                    e3nn_out_irreps.append((mul, (ir_out.l, ir_out.p)))
                    e3nn_paths.append(
                        (i, j, k, e3nn_mode, e3nn_mode == "uvu" or trainable)
                    )
    return e3nn_paths, o3.Irreps(e3nn_out_irreps)


def to_possible_tp_irreps(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    parity: bool,
    lmax: Optional[int] = None,
) -> o3.Irreps:
    lmax = irreps_in2.lmax if lmax is None else lmax
    irrep_set = {
        ir3
        for _, ir1 in irreps_in1
        for _, ir2 in irreps_in2
        for ir3 in ir1 * ir2
        if ir3.l <= lmax and (parity or ir3.p == (-1) ** ir3.l)
    }
    return o3.Irreps(sorted(irrep_set, key=lambda ir: (ir.l, ir.p))).regroup()
