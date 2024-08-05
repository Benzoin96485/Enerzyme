import torch
from torch.nn import Module, Linear, init
from ..functional import segment_sum
from ..layers.electron_embedding import ResidualMLP


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
        activation: str = "swish",
    ) -> None:
        """ Initializes the LocalInteraction class. """
        super().__init__()
        self.radial_s = Linear(num_rbf, dim_embedding, bias=False)
        self.radial_p = Linear(num_rbf, dim_embedding, bias=False)
        self.radial_d = Linear(num_rbf, dim_embedding, bias=False)
        self.resblock_x = ResidualMLP(dim_embedding, num_residual_x, activation)
        self.resblock_s = ResidualMLP(dim_embedding, num_residual_s, activation)
        self.resblock_p = ResidualMLP(dim_embedding, num_residual_p, activation)
        self.resblock_d = ResidualMLP(dim_embedding, num_residual_d, activation)
        self.projection_p = Linear(dim_embedding, 2 * dim_embedding, bias=False)
        self.projection_d = Linear(dim_embedding, 2 * dim_embedding, bias=False)
        self.resblock = ResidualMLP(
            dim_embedding, num_residual, activation, zero_init=True
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
        s = xx + segment_sum(gs * xs, idx_i)
        p = segment_sum(gp * xp.unsqueeze(-2), idx_i)
        d = segment_sum(gd * xd.unsqueeze(-2), idx_i)
        # project tensorial features to scalars
        pa, pb = torch.split(self.projection_p(p), p.shape[-1], dim=-1)
        da, db = torch.split(self.projection_d(d), d.shape[-1], dim=-1)
        return self.resblock(s + (pa * pb).sum(-2) + (da * db).sum(-2))


class NonlocalInteraction(Module):
    ...


class InteractionModule(Module):
    ...