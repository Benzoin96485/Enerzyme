from typing import Dict, Optional, Literal
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .mlp import ResidualMLP
from ..functional import segment_sum


class BaseElectronEmbedding(ABC):
    def __init__(self, dim_embedding: int, num_residual: int, activation: str, attribute: Literal["charge", "spin"]="charge"):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_residual = num_residual
        self.activation = activation
        self.attribute = attribute

    @abstractmethod
    def get_embedding(self, atom_embedding, Q, batch_seg, eps, **kwargs) -> Tensor:
        ...

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output["electron_embedding"] = self.get_embedding(**net_input)
        return output


class ElectronicEmbedding(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with the
    electrons.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the interaction.
        num_residual_pre_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the interaction.
        num_residual_post (int):
            Number of residual blocks applied to interaction features.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_residual: int,
        activation: str = "swish",
        attribute: Literal["charge", "spin"]="charge"
    ) -> None:
        """ Initializes the ElectronicEmbedding class. """
        super().__init__(dim_embedding, num_residual, activation, attribute)
        self.linear_q = nn.Linear(dim_embedding, dim_embedding)
        self.sqrt_dim_embedding = dim_embedding ** 0.5
        if attribute == "charge":  # charges are duplicated to use separate weights for +/-
            self.linear_k = nn.Linear(2, dim_embedding, bias=False)
            self.linear_v = nn.Linear(2, dim_embedding, bias=False)
        else:
            self.linear_k = nn.Linear(1, dim_embedding, bias=False)
            self.linear_v = nn.Linear(1, dim_embedding, bias=False)
        self.resblock = ResidualMLP(
            dim_embedding,
            num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        nn.init.orthogonal_(self.linear_k.weight)
        nn.init.orthogonal_(self.linear_v.weight)
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

    def get_embedding(
        self,
        atom_embedding: Tensor,
        Q: Optional[Tensor]=None,
        S: Optional[Tensor]=None,
        batch_seg: Optional[Tensor]=None,
        mask: Optional[Tensor]=None,  # only for backwards compatibility
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = torch.zeros(atom_embedding.size(0), dtype=torch.int64, device=x.device)
        q = self.linear_q(atom_embedding)  # queries
        if self.attribute == "charge":
            e = F.relu(torch.stack([Q, -Q], dim=-1))
        else:
            e = torch.abs(S).unsqueeze(-1)  # +/- spin is the same => abs
        enorm = torch.maximum(e, torch.ones_like(e))
        k = self.linear_k(e / enorm)[batch_seg]  # keys
        v = self.linear_v(e)[batch_seg]  # values
        dot = torch.sum(k * q, dim=-1) / self.sqrt_dim_embedding  # scaled dot product
        a = nn.functional.softplus(dot)  # unnormalized attention weights
        anorm = segment_sum(a, batch_seg)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return self.resblock((a / (anorm + eps)).unsqueeze(-1) * v)