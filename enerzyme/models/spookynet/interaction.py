from typing import Optional, Tuple
import torch
from torch.nn import Module, Linear, init
from ..functional import segment_sum_coo
from ..activation import ACTIVATION_KEY_TYPE
from ..layers.electron_embedding import ResidualMLP
from ..layers.mlp import ResidualStack as _ResidualStack
from ..layers.attention import Attention


def ResidualStack(
    dim_embedding: int, num_residual: int, 
    activation_fn: ACTIVATION_KEY_TYPE,
    use_bias: bool=True, zero_init: bool=True
) -> _ResidualStack:
    return _ResidualStack(
        dim_feature=dim_embedding,
        num_residual=num_residual,
        activation_fn=activation_fn,
        activation_params = {
            "dim_feature": dim_embedding,
            "learnable": True
        },
        initial_weight1="orthogonal", initial_weight2="zero" if zero_init else "orthogonal",
        use_bias=use_bias
    )


class LocalInteraction(Module):
    """
    Block for updating atomic features through local interactions with
    neighboring atoms (message-passing).

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
        num_rbf: int,
        num_residual_x: int,
        num_residual_s: int,
        num_residual_p: int,
        num_residual_d: int,
        num_residual: int,
        activation_fn: ACTIVATION_KEY_TYPE="swish",
    ) -> None:
        """ Initializes the LocalInteraction class. """
        super().__init__()
        self.radial_s = Linear(num_rbf, dim_embedding, bias=False)
        self.radial_p = Linear(num_rbf, dim_embedding, bias=False)
        self.radial_d = Linear(num_rbf, dim_embedding, bias=False)
        self.resblock_x = ResidualMLP(dim_embedding, num_residual_x, activation_fn, zero_init=False)
        self.resblock_s = ResidualMLP(dim_embedding, num_residual_s, activation_fn, zero_init=False)
        self.resblock_p = ResidualMLP(dim_embedding, num_residual_p, activation_fn, zero_init=False)
        self.resblock_d = ResidualMLP(dim_embedding, num_residual_d, activation_fn, zero_init=False)
        self.projection_p = Linear(dim_embedding, 2 * dim_embedding, bias=False)
        self.projection_d = Linear(dim_embedding, 2 * dim_embedding, bias=False)
        self.resblock = ResidualMLP(
            dim_embedding, num_residual, activation_fn, zero_init=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        init.orthogonal_(self.radial_s.weight)
        init.orthogonal_(self.radial_p.weight)
        init.orthogonal_(self.radial_d.weight)
        init.orthogonal_(self.projection_p.weight)
        init.orthogonal_(self.projection_d.weight)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        pij: torch.Tensor,
        dij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.
        P: Number of atom pairs.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        rbf (FloatTensor [N, num_basis_functions]):
            Values of the radial basis functions for the pairwise distances.
        idx_i (LongTensor [P]):
            Index of atom i for all atomic pairs ij. Each pair must be
            specified as both ij and ji.
        idx_j (LongTensor [P]):
            Same as idx_i, but for atom j.
        """
        # interaction functions
        gs = self.radial_s(rbf)
        gp = self.radial_p(rbf).unsqueeze(-2) * pij.unsqueeze(-1)
        gd = self.radial_d(rbf).unsqueeze(-2) * dij.unsqueeze(-1)
        # atom featurizations
        xx = self.resblock_x(x)
        xs = self.resblock_s(x)
        xp = self.resblock_p(x)
        xd = self.resblock_d(x)
        # collect neighbors
        if x.device.type == "cpu":  # indexing is faster on CPUs
            xs = xs[idx_j]  # L=0
            xp = xp[idx_j]  # L=1
            xd = xd[idx_j]  # L=2
        else:  # gathering is faster on GPUs
            j = idx_j.view(-1, 1).expand(-1, x.shape[-1])  # index for gathering
            xs = torch.gather(xs, 0, j)  # L=0
            xp = torch.gather(xp, 0, j)  # L=1
            xd = torch.gather(xd, 0, j)  # L=2
        # sum over neighbors
        N = len(x)
        s = xx + segment_sum_coo(gs * xs, idx_i, dim_size=N)
        p = segment_sum_coo(gp * xp.unsqueeze(-2), idx_i, dim_size=N)
        d = segment_sum_coo(gd * xd.unsqueeze(-2), idx_i, dim_size=N)
        # project tensorial features to scalars
        pa, pb = torch.split(self.projection_p(p), p.shape[-1], dim=-1)
        da, db = torch.split(self.projection_d(d), d.shape[-1], dim=-1)
        return self.resblock(s + (pa * pb).sum(-2) + (da * db).sum(-2))


class NonlocalInteraction(Module):
    """
    Block for updating atomic features through nonlocal interactions with all
    atoms.

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
        num_residual_q: int,
        num_residual_k: int,
        num_residual_v: int,
        activation_fn: ACTIVATION_KEY_TYPE="swish",
    ) -> None:
        """ Initializes the NonlocalInteraction class. """
        super().__init__()
        self.resblock_q = ResidualMLP(
            dim_embedding, num_residual_q, activation_fn, zero_init=True
        )
        self.resblock_k = ResidualMLP(
            dim_embedding, num_residual_k, activation_fn, zero_init=True
        )
        self.resblock_v = ResidualMLP(
            dim_embedding, num_residual_v, activation_fn, zero_init=True
        )
        self.attention = Attention(dim_embedding, dim_embedding)

    def forward(
        self,
        x: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        q = self.resblock_q(x)  # queries
        k = self.resblock_k(x)  # keys
        v = self.resblock_v(x)  # values
        return self.attention(q, k, v, num_batch, batch_seg, mask)


class InteractionModule(Module):
    """
    InteractionModule of SpookyNet, which computes a single iteration.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features before
            interaction with neighbouring atoms.
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms.
        num_residual_pre_local_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the local interaction.
        num_residual_pre_local_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the local interaction.
        num_residual_post_local (int):
            Number of residual blocks applied to interaction features.
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        num_residual_pre: int,
        num_residual_local_x: int,
        num_residual_local_s: int,
        num_residual_local_p: int,
        num_residual_local_d: int,
        num_residual_local: int,
        num_residual_nonlocal_q: int,
        num_residual_nonlocal_k: int,
        num_residual_nonlocal_v: int,
        num_residual_post: int,
        num_residual_output: int,
        activation_fn: ACTIVATION_KEY_TYPE="swish",
    ) -> None:
        """ Initializes the InteractionModule class. """
        super().__init__()
        # initialize modules
        self.local_interaction = LocalInteraction(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            num_residual_x=num_residual_local_x,
            num_residual_s=num_residual_local_s,
            num_residual_p=num_residual_local_p,
            num_residual_d=num_residual_local_d,
            num_residual=num_residual_local,
            activation_fn=activation_fn,
        )
        self.nonlocal_interaction = NonlocalInteraction(
            dim_embedding=dim_embedding,
            num_residual_q=num_residual_nonlocal_q,
            num_residual_k=num_residual_nonlocal_k,
            num_residual_v=num_residual_nonlocal_v,
            activation_fn=activation_fn,
        )
        self.residual_pre = ResidualStack(dim_embedding, num_residual_pre, activation_fn)
        self.residual_post = ResidualStack(dim_embedding, num_residual_post, activation_fn)
        self.resblock = ResidualMLP(
            dim_embedding, num_residual_output, activation_fn=activation_fn
        )

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        pij: torch.Tensor,
        dij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate all modules in the block.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            x (FloatTensor [N, num_features]):
                Latent atomic feature vectors.
            rbf (FloatTensor [P, num_basis_functions]):
                Values of the radial basis functions for the pairwise distances.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.
            num_batch (int):
                Batch size (number of different molecules).
            batch_seg (LongTensor [N]):
                Index for each atom that specifies to which molecule in the
                batch it belongs.
        Returns:
            x (FloatTensor [N, num_features]):
                Updated latent atomic feature vectors.
            y (FloatTensor [N, num_features]):
                Contribution to output atomic features (environment
                descriptors).
        """
        x = self.residual_pre(x)
        l = self.local_interaction(x, rbf, pij, dij, idx_i, idx_j)
        n = self.nonlocal_interaction(x, num_batch, batch_seg, mask)
        x = self.residual_post(x + l + n)
        return x, self.resblock(x)