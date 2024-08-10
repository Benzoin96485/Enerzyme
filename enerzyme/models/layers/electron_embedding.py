from typing import Dict, Optional, Literal
from abc import abstractmethod
import torch
from torch import Tensor
from torch.nn import Linear, init
import torch.nn.functional as F
from .mlp import ResidualMLP as _ResidualMLP
from . import BaseFFLayer
from ..activation import ACTIVATION_KEY_TYPE
from ..functional import segment_sum


def ResidualMLP(
    dim_embedding: int, num_residual: int, 
    activation_fn: ACTIVATION_KEY_TYPE,
    use_bias: bool=True, zero_init: bool=True
) -> _ResidualMLP:
    return _ResidualMLP(
        dim_feature_in=dim_embedding,
        dim_feature_out=dim_embedding,
        num_residual=num_residual,
        activation_fn=activation_fn,
        activation_params = {
            "dim_feature": dim_embedding,
            "learnable": True
        },
        initial_weight1="orthogonal", initial_weight2="zero", initial_weight_out="zero" if zero_init else "orthogonal",
        use_bias_residual=use_bias, use_bias_out=use_bias
    )


class BaseElectronEmbedding(BaseFFLayer):
    def __init__(
        self, dim_embedding: int, num_residual: int, attribute: Literal["charge", "spin"]="charge"
    ) -> None:
        input_fields = {"atom_embedding", "batch_seg"}
        if attribute == "charge":
            input_fields.add("Q")
        elif attribute == "spin":
            input_fields.add("S")
        super().__init__(input_fields=input_fields, output_fields={"electron_embedding"})
        self.dim_embedding = dim_embedding
        self.num_residual = num_residual
        self.attribute = attribute
        self.reset_field_name(electron_embedding=f"{attribute}_embedding")

    @abstractmethod
    def get_electron_embedding(self, atom_embedding, Q, batch_seg) -> Dict[Literal["electron_embedding"], Tensor]:
        ...


class ElectronicEmbedding(BaseElectronEmbedding):
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
        activation_fn: ACTIVATION_KEY_TYPE="swish",
        attribute: Literal["charge", "spin"]="charge"
    ) -> None:
        """ Initializes the ElectronicEmbedding class. """
        
        super().__init__(dim_embedding, num_residual, attribute)
        self.linear_q = Linear(dim_embedding, dim_embedding)
        self.sqrt_dim_embedding = dim_embedding ** 0.5
        if attribute == "charge":  # charges are duplicated to use separate weights for +/-
            self.linear_k = Linear(2, dim_embedding, bias=False)
            self.linear_v = Linear(2, dim_embedding, bias=False)
        else:
            self.linear_k = Linear(1, dim_embedding, bias=False)
            self.linear_v = Linear(1, dim_embedding, bias=False)
        self.resblock = ResidualMLP(
            dim_embedding, num_residual, activation_fn,
            use_bias=False, zero_init=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        init.orthogonal_(self.linear_k.weight)
        init.orthogonal_(self.linear_v.weight)
        init.orthogonal_(self.linear_q.weight)
        init.zeros_(self.linear_q.bias)
        self.eps = 1e-8

    def get_electron_embedding(
        self,
        atom_embedding: Tensor,
        Q: Optional[Tensor]=None,
        S: Optional[Tensor]=None,
        batch_seg: Optional[Tensor]=None
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = torch.zeros(atom_embedding.size(0), dtype=torch.long, device=atom_embedding.device)
        q = self.linear_q(atom_embedding)  # queries
        if self.attribute == "charge":
            if Q is None:
                Q = torch.zeros(batch_seg[-1] + 1, dtype=atom_embedding.dtype, device=atom_embedding.device)
            e = F.relu(torch.stack([Q, -Q], dim=-1))
        else:
            if S is None:
                S = torch.zeros(batch_seg[-1] + 1, dtype=atom_embedding.dtype, device=atom_embedding.device)
            e = torch.abs(S).unsqueeze(-1)  # +/- spin is the same => abs
        enorm = torch.maximum(e, torch.ones_like(e))
        k = self.linear_k(e / enorm)[batch_seg]  # keys
        v = self.linear_v(e)[batch_seg]  # values
        dot = torch.sum(k * q, dim=-1) / self.sqrt_dim_embedding  # scaled dot product
        a = F.softplus(dot)  # unnormalized attention weights
        anorm = segment_sum(a, batch_seg)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return self.resblock((a / (anorm + self.eps)).unsqueeze(-1) * v)


class NonlinearElectronicEmbedding(BaseElectronEmbedding):
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
        self, dim_embedding: int, num_residual: int, activation_fn: str="swish", attribute: Literal["charge", "spin"]="charge"
    ) -> None:
        """ Initializes the NonlinearElectronicEmbedding class. """
        super(NonlinearElectronicEmbedding, self).__init__(dim_embedding, num_residual, attribute)
        self.linear_q = Linear(dim_embedding, dim_embedding, bias=False)
        self.featurize_k = Linear(1, dim_embedding)
        self.resblock_k = ResidualMLP(
            dim_embedding, num_residual, activation_fn=activation_fn, zero_init=True
        )
        self.featurize_v = Linear(1, dim_embedding, bias=False)
        self.resblock_v = ResidualMLP(
            dim_embedding,
            num_residual,
            activation_fn=activation_fn,
            zero_init=True,
            use_bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        init.orthogonal_(self.linear_q.weight)
        init.orthogonal_(self.featurize_k.weight)
        init.zeros_(self.featurize_k.bias)
        init.orthogonal_(self.featurize_v.weight)

    def get_electron_embedding(
        self,
        atom_embedding: Tensor,
        Q: Optional[Tensor]=None,
        S: Optional[Tensor]=None,
        batch_seg: Optional[Tensor]=None,
        mask: Optional[Tensor] = None,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = torch.zeros(atom_embedding.size(0), dtype=torch.long, device=atom_embedding.device)
        if self.attribute == "charge":
            E = Q
        else:
            E = S
        e = E.unsqueeze(-1)
        q = self.linear_q(atom_embedding)  # queries
        k = self.resblock_k(self.featurize_k(e))[batch_seg]  # keys
        v = self.resblock_v(self.featurize_v(e))[batch_seg]  # values
        # dot product
        dot = torch.sum(k * q, dim=-1)
        # determine maximum dot product (for numerics)
        num_batch = batch_seg[-1] + 1
        if num_batch > 1:
            if mask is None:
                mask = (
                    F.one_hot(batch_seg)
                    .to(dtype=atom_embedding.dtype, device=atom_embedding.device)
                    .transpose(-1, -2)
                )
            tmp = dot.view(1, -1).expand(num_batch, -1)
            tmp, _ = torch.max(mask * tmp, dim=-1)
            if tmp.device.type == "cpu":  # indexing is faster on CPUs
                maximum = tmp[batch_seg]
            else:  # gathering is faster on GPUs
                maximum = torch.gather(tmp, 0, batch_seg)
        else:
            maximum = torch.max(dot)
        # attention
        d = k.shape[-1]
        a = torch.exp((dot - maximum) / d ** 0.5)

        anorm = segment_sum(a, batch_seg)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return (a / (anorm + eps)).unsqueeze(-1) * v
