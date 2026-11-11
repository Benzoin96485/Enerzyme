import math
from typing import Dict, List, Set
import torch
from torch import Tensor
from . import BaseFFLayer
from ..functional import segment_sum_coo


class EnergyReduceLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"Ea", "batch_seg", "Za"}, output_fields={"E", "Ea"})

    def get_relevant_input_fields(self, net_input_fields: Set[str]) -> Set[str]:
        relevant_input_fields = set()
        for k in net_input_fields:
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                relevant_input_fields.add(k)
        return relevant_input_fields

    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        Ea = relevant_input["Ea"]
        for k, v in relevant_input.items():
            if k[0] == "E" and k[-1] == "a" and len(k) > 2:
                Ea = Ea + v
        if relevant_input["batch_seg"] is not None:
            batch_seg = relevant_input["batch_seg"]
        else:
            batch_seg = torch.zeros_like(relevant_input["Za"])
        return {"E": segment_sum_coo(Ea, batch_seg), "Ea": Ea}


class EdgeEnergyReduceLayer(BaseFFLayer):
    """Sum directed pair energies onto the center atom.

    ``idx_i_sr`` is the center. The result is divided by
    ``sqrt(avg_num_neighbors)``. ``pair_symmetry`` also divides by ``sqrt(2)``,
    the Allegro normalization for a graph that stores both ``ij`` and ``ji``:
    each endpoint contributes, and the factor keeps ``dE/dr`` from double-counting
    that pair of directions. Isolated atoms stay at zero so their reference
    energy can sit in ``AtomicAffine``.
    """

    def __init__(
        self,
        avg_num_neighbors: float = 1.0,
        pair_symmetry: bool = True,
    ) -> None:
        super().__init__(
            input_fields={"Eij", "idx_i_sr", "Za"},
            output_fields={"Ea"},
        )
        if float(avg_num_neighbors) <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
            )
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.pair_symmetry = bool(pair_symmetry)

    def get_output(self, Eij: Tensor, idx_i_sr: Tensor, Za: Tensor) -> Dict[str, Tensor]:
        n_atoms = Za.shape[0]
        if Eij.shape[0] != idx_i_sr.shape[0]:
            raise ValueError(
                f"Eij has {Eij.shape[0]} edges but idx_i_sr has {idx_i_sr.shape[0]}"
            )
        out = Eij.new_zeros((n_atoms,) + tuple(Eij.shape[1:]))
        if idx_i_sr.numel():
            centers = idx_i_sr.long()
            if int(centers.max()) >= n_atoms or int(centers.min()) < 0:
                raise ValueError(
                    f"idx_i_sr must lie in 0..{n_atoms - 1}, "
                    f"got {int(centers.min())}..{int(centers.max())}"
                )
            out.index_add_(0, centers, Eij)
        scale = 1.0 / math.sqrt(self.avg_num_neighbors)
        if self.pair_symmetry:
            scale /= math.sqrt(2.0)
        return {"Ea": out * scale}


class ShallowEnsembleReduceLayer(BaseFFLayer):
    def __init__(self, 
        reduce_mean: List[str]=[], 
        var: List[str]=[], 
        std: List[str]=[], 
        relative_energy: bool=False, 
        train_only: bool=False, 
        eval_only: bool=False,
        test_only: bool=False,
        test_exclude: bool=False
    ) -> None:
        super().__init__(
            input_fields=set(reduce_mean) | set(var) | set(std), 
            output_fields=set(reduce_mean) | set(
                [name + "_var" for name in var]
            ) | set(
                [name + "_std" for name in std]
            ),
            train_only=train_only,
            eval_only=eval_only,
            test_only=test_only,
            test_exclude=test_exclude
        )
        self.var = var
        self.std = std
        self.reduce_mean = reduce_mean
        self.relative_energy = relative_energy

    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = dict()
        for name in self.var:
            if self.relative_energy and name.startswith("E"):
                output[name + "_var"] = (relevant_input[name] - relevant_input[name].mean(dim=0)).var(dim=-1, unbiased=True)
            else:
                output[name + "_var"] = relevant_input[name].var(dim=-1, unbiased=True)
        for name in self.std:
            if self.relative_energy and name.startswith("E"):
                output[name + "_std"] = (relevant_input[name] - relevant_input[name].mean(dim=0)).std(dim=-1, unbiased=True)
            else:
                output[name + "_std"] = relevant_input[name].std(dim=-1, unbiased=True)
        for name in self.reduce_mean:
            output[name] = relevant_input[name].mean(dim=-1)
        return output
