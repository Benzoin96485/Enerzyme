from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from . import BaseFFLayer
from ..mp.ops import scatter_sum


class NeuralSpinChargeEquilibrationLayer(BaseFFLayer):
    r"""
    Neural spin–charge equilibration following an AIMNet2-NSE-style update
    for each spin channel :math:`s \in \{\alpha, \beta\}`.

    .. math::

        q_i^s = \tilde{q}_i^s +
        f_i^s \frac{Q^s - \sum_j \tilde{q}_j^s}{\sum_j f_j^s}.

    The readout predicts the per-atom quantities ``Qa_alpha_tilde``,
    ``Qa_beta_tilde``, ``fa_alpha`` and ``fa_beta`` from ``atom_feature``.
    Following the Enerzyme/UMA convention, :math:`Q` is the total charge
    per graph and :math:`S = 2 S_{\mathrm{phys}}` is its spin multiplicity
    minus one. The spin-resolved targets satisfy

    .. math::

        Q^\alpha = (Q + S)/2, \qquad Q^\beta = (Q - S)/2,

    so that :math:`Q^\alpha + Q^\beta = Q` and :math:`Q^\alpha - Q^\beta = S`.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_output(
        self,
        Qa_alpha_tilde: Tensor,
        Qa_beta_tilde: Tensor,
        fa_alpha: Tensor,
        fa_beta: Tensor,
        Q: Tensor,
        S: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[
        Literal[
            "Qa_alpha",
            "Qa_beta",
            "Qa",
            "Sa",
            "Q_alpha",
            "Q_beta",
            "Q",
            "S",
        ],
        Tensor,
    ]:
        if batch_seg is None:
            batch_seg = torch.zeros_like(Qa_alpha_tilde, dtype=torch.long)

        # Per-graph sums of tilde charges and weights
        sum_q_alpha_tilde = scatter_sum(Qa_alpha_tilde, batch_seg)
        sum_q_beta_tilde = scatter_sum(Qa_beta_tilde, batch_seg)
        raw_Q = sum_q_alpha_tilde + sum_q_beta_tilde
        raw_S = sum_q_alpha_tilde - sum_q_beta_tilde
        sum_f_alpha = scatter_sum(fa_alpha, batch_seg)
        sum_f_beta = scatter_sum(fa_beta, batch_seg)

        # Spin-resolved target total charges (per graph)
        Q_alpha_target = 0.5 * (Q + S)
        Q_beta_target = 0.5 * (Q - S)

        # Avoid division by zero when all f_i^s are zero for a graph
        eps = torch.finfo(Qa_alpha_tilde.dtype).eps
        den_alpha = sum_f_alpha + eps
        den_beta = sum_f_beta + eps

        delta_alpha = (Q_alpha_target - sum_q_alpha_tilde) / den_alpha
        delta_beta = (Q_beta_target - sum_q_beta_tilde) / den_beta

        # Broadcast per-graph corrections back to atoms
        delta_alpha_expanded = delta_alpha[batch_seg]
        delta_beta_expanded = delta_beta[batch_seg]

        Qa_alpha = Qa_alpha_tilde + fa_alpha * delta_alpha_expanded
        Qa_beta = Qa_beta_tilde + fa_beta * delta_beta_expanded

        Qa = Qa_alpha + Qa_beta
        Sa = Qa_alpha - Qa_beta

        Q_alpha = scatter_sum(Qa_alpha, batch_seg)
        Q_beta = scatter_sum(Qa_beta, batch_seg)

        return {
            "Qa_alpha": Qa_alpha,
            "Qa_beta": Qa_beta,
            "Qa": Qa,
            "Sa": Sa,
            "Q_alpha": Q_alpha,
            "Q_beta": Q_beta,
            "Q": raw_Q,
            "S": raw_S,
        }

