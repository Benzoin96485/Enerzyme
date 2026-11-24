from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Union, Callable
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from e3nn import o3
from ..mp.nn.radial_mlp import e3nn_radial_mlp
from e3nn.nn import Activation
from e3nn.o3 import Irreps, TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from ..mp.equivariant.body_order import SymmetricContraction
from ..mp.block import BlockState
from ..mp.graph import GraphContext
from ..layers.atom_embedding import one_hot_species
from ..mp.ops import scatter_sum
from ..mp.equivariant.e3nn import tp_out_irreps_with_instructions, reshape_irreps, linear_out_irreps


@compile_mode("script")
class EquivariantProductBasisBlock(nn.Module):
    def __init__(
        self,
        node_feats_irreps: Irreps,
        target_irreps: Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: Tensor,
        sc: Optional[Tensor],
        node_attrs: Tensor,
    ) -> Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class TensorProductWeightsBlock(Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


@compile_mode("script")
class InteractionBlock(ABC, Module):
    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps # one hot atom type
        self.node_feats_irreps = node_feats_irreps # atom embedding
        self.edge_attrs_irreps = edge_attrs_irreps # spherical harmonics
        self.edge_feats_irreps = edge_feats_irreps # radial basis functions
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        ...

    @abstractmethod
    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        ...


@compile_mode("script")
class ResidualElementDependentInteractionBlock(InteractionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tensor:
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return message + sc  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = e3nn_radial_mlp(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            F.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tensor:
        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = self.linear_up(node_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return message  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = e3nn_radial_mlp(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            F.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tensor:
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = message + sc
        return message  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = e3nn_radial_mlp(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            F.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = e3nn_radial_mlp(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            F.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats) # Linear map of node features
        tp_weights = self.conv_tp_weights(edge_feats) # Lift rbf to tensor product weights
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps] tensor product of edge irreps and node features with rbf weights
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps] reduce the messages from all neighbors
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]



@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        self.node_feats_down_irreps = Irreps("64x0e")
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        self.linear_down = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = e3nn_radial_mlp(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            F.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(self.irreps_out)

        # Skip connection.
        self.skip_linear = o3.Linear(self.node_feats_irreps, self.hidden_irreps)

    def forward(
        self,
        node_attrs: Tensor,
        node_feats: Tensor,
        edge_attrs: Tensor,
        edge_feats: Tensor,
        sender: Tensor,
        receiver: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class LinearReadoutBlock(nn.Module):
    def __init__(self, irreps_in: Irreps, shallow_ensemble_size: int=1):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=Irreps(f"{shallow_ensemble_size * 2}x0e"))

    def forward(self, x: Tensor) -> Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 2 * shallow_ensemble_size]


@compile_mode("script")
class NonLinearReadoutBlock(nn.Module):
    def __init__(
        self, irreps_in: Irreps, MLP_irreps: Irreps, gate: Optional[Callable], shallow_ensemble_size: int=1
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=Irreps(f"{shallow_ensemble_size * 2}x0e")
        )

    def forward(self, x: Tensor) -> Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 2 * shallow_ensemble_size]


INTERACTION_CLASSES = {
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
}

def element_one_hot(ctx: GraphContext, num_elements: int, dtype: torch.dtype) -> Tensor:
    """One-hot species ``(N, num_elements)`` from ``Za``, cached per forward."""
    def encode():
        return one_hot_species(ctx.inputs["Za"], num_elements, dtype)

    return ctx.cached(("element_one_hot", int(num_elements), dtype), encode)


class MACEBlock(Module):
    """One MACE layer: equivariant convolution, then the symmetric-contraction product.

    Node features live in ``state.equi`` (e3nn flat irreps). Edge attributes are
    the component-normalized e3nn harmonics of ``R_receiver - R_sender``.
    """

    def __init__(self, interaction: InteractionBlock, product: EquivariantProductBasisBlock,
                 num_elements: int, max_ell: int) -> None:
        super().__init__()
        self.interaction = interaction
        self.product = product
        self.num_elements = int(num_elements)
        self.max_ell = int(max_ell)

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        node_feats = state.equi
        node_attrs = element_one_hot(ctx, self.num_elements, node_feats.dtype)
        edge_attrs = ctx.spherical_harmonics(self.max_ell, "component", True, layout="e3nn", edge_sign=-1.0)
        node_feats, sc = self.interaction(
            node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
            edge_feats=ctx.rbf, sender=ctx.sender, receiver=ctx.receiver,
        )
        return state._replace(equi=self.product(node_feats=node_feats, sc=sc, node_attrs=node_attrs))
