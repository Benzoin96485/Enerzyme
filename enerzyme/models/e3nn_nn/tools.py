from typing import List, Optional, Tuple, Union
from collections import namedtuple
import torch
from torch import Tensor
import e3nn.o3 as o3
from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps

_INPUT = namedtuple("_INPUT", "tensor, start, stop")
_TP = namedtuple("_TP", "op, args")


def scalar_0e_dim(irreps: Union[str, Irreps, None]) -> Optional[int]:
    """Number of even scalar (0e) channels, or None if ``irreps`` is unset."""
    if irreps is None:
        return None
    ir = Irreps(irreps)
    return sum(mul for mul, irr in ir if irr.l == 0 and irr.p == 1)


def extract_scalar_0e(
    atom_feature: Tensor, irreps: Union[str, Irreps, None]
) -> Tensor:
    """Return even-scalar (0e) channels from a flat irreps feature.

    If ``irreps`` is None, ``atom_feature`` is treated as already-scalar and
    returned unchanged.
    """
    if irreps is None:
        return atom_feature
    ir = Irreps(irreps)
    if atom_feature.shape[-1] != ir.dim:
        raise ValueError(
            f"atom_feature last dim {atom_feature.shape[-1]} != irreps.dim {ir.dim}"
        )
    pieces = []
    offset = 0
    for mul, irr in ir:
        width = mul * irr.dim
        if irr.l == 0 and irr.p == 1:
            pieces.append(atom_feature[..., offset : offset + width])
        offset += width
    if not pieces:
        raise ValueError(f"No scalar (0e) channels in irreps {ir}")
    return torch.cat(pieces, dim=-1)


def linear_out_irreps(irreps: Irreps, target_irreps: Irreps) -> Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f"{ir_in} not in {target_irreps}")

    return Irreps(irreps_mid)


def tp_out_irreps_with_instructions(
    irreps1: Irreps, irreps2: Irreps, target_irreps: Irreps
) -> Tuple[Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


def satisfy(l1: int, l2: int, restriction: Optional[str] = None) -> bool:
    """Angular-momentum restriction used by ACE / TACE path filters."""
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
    irreps_out: Irreps,
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    *,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
    e3nn_mode: str = "uvu",
    trainable: bool = False,
    identical_inputs: bool = False,
) -> Tuple[List[Tuple[int, int, int, str, bool]], Irreps]:
    """Build e3nn TensorProduct instructions with optional l-restrictions.

    Adapted from xvzemin/tace (MIT).
    """
    irreps_out = Irreps(irreps_out)
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)
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
    return e3nn_paths, Irreps(e3nn_out_irreps)


def to_possible_tp_irreps(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    parity: bool,
    lmax: Optional[int] = None,
) -> Irreps:
    """Irreps closed under TP(in1, in2), truncated to ``lmax`` / natural parity."""
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)
    lmax = irreps_in2.lmax if lmax is None else lmax
    irrep_set = {
        ir3
        for _, ir1 in irreps_in1
        for _, ir2 in irreps_in2
        for ir3 in ir1 * ir2
        if ir3.l <= lmax and (parity or ir3.p == (-1) ** ir3.l)
    }
    return Irreps(sorted(irrep_set, key=lambda ir: (ir.l, ir.p))).regroup()


@compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: Irreps) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)
    

def _wigner_nj(
    irrepss: List[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = Irreps(irreps_out)
    irrepss = [Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out