################################################################################
# Cartesian version of precomputed product basis
# It only has practical value when the correlation is equal to 2
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks 
# for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
################################################################################

from typing import Dict, Optional, Union, List

import opt_einsum_fx
import torch
import torch.fx
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode


from ._irreps import Irreps
from ._zemin import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
        element_aware: Optional[Dict[int, bool]] = None,
        coupled_channel: Optional[Dict[int, bool]] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            l = irrep_out.ir.l
            this_element = element_aware[l] if element_aware is not None else True
            this_coupled = coupled_channel[l] if coupled_channel is not None else False
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                    element_aware=this_element,
                    coupled_channel=this_coupled,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        return [contraction(x, y) for contraction in self.contractions]


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irrep_out: Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        element_aware: bool = True,
        coupled_channel: bool = False,
    ) -> None:
        super().__init__()

        assert not coupled_channel, "Precompute coupled product basis is not allowed now" # TODO

        num_elements = num_elements if element_aware else 1
        self.element_aware = element_aware
        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()

        path_weight = []
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            path_weight.append(not torch.equal(U_matrix, torch.zeros_like(U_matrix)))
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1] # num_path
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2] 
            
            if coupled_channel:
                weight_shape = (num_elements, num_params, self.num_features, self.num_features)
                alpha = num_params * self.num_features # TODO, in forward better ?
                w_str = "ekcC"
                out_str = "bC" # TODO maybe nor correct
            else:
                weight_shape = (num_elements, num_params, self.num_features)
                alpha = num_params
                w_str = "ekc"
                out_str = "bc"

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + [f"ik,{w_str},bci,be -> {out_str}"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn(weight_shape),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                w = torch.nn.Parameter(
                    torch.randn(weight_shape)
                    / alpha
                )
                self.weights_max = w
            else:
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + [f"k,{w_str},be->{out_str}"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn(weight_shape),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn(weight_shape)
                    / alpha
                )
                self.weights.append(w)

        for idx, keep in enumerate(path_weight):
            zero_flag = not keep
            if idx < correlation - 1:
                if zero_flag:
                    self.weights[idx] = EmptyParam(self.weights[idx])
                self.register_buffer(
                    f"weights_{idx}_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )
            else:
                if zero_flag:
                    self.weights_max = EmptyParam(self.weights_max)
                self.register_buffer(
                    "weights_max_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        
        if not self.element_aware:
            y = torch.ones(
                (y.size(0), 1),
                dtype=y.dtype,
                device=y.device,
            )

        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


class EmptyParam(torch.nn.Parameter):
    def __new__(cls, data):  # pylint: disable=signature-differs
        zero = torch.zeros_like(data)
        return super().__new__(cls, zero, requires_grad=False)

    def requires_grad_(
        self, mode: bool = True
    ):  # pylint: disable=arguments-differ, arguments-renamed
        del mode
        return self