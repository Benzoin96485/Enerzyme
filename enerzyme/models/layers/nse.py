from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from . import BaseFFLayer
from ..functional import segment_sum_coo


class NeuralSpinChargeEquilibrationLayer(BaseFFLayer):
    r"""
    Neural spin–charge equilibration layer following an AIMNet2-NSE-style update:

        q_i^s = \tilde{q}_i^s + f_i^s * (Q^s - \sum_j \tilde{q}_j^s) / \sum_j f_j^s

    for each spin channel \(s \in \{\alpha, \beta\}\).

    Design notes:
    - Inputs `Qa_alpha_tilde`, `Qa_beta_tilde`, `fa_alpha`, `fa_beta` are per-atom
      NSE intermediate quantities predicted by a readout head from `atom_feature`.
    - Graph-level `Q` and `S` follow the Enerzyme/UMA convention:
        * `Q` is the total charge per graph.
        * `S` is the spin multiplicity minus 1, i.e. \(S = 2 S_{\text{phys}}\).
    - Within this layer we construct spin-resolved targets
        Q^α = (Q + S) / 2,   Q^β = (Q - S) / 2,
      so that Q^α + Q^β = Q and Q^α - Q^β = S.
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
        sum_q_alpha_tilde = segment_sum_coo(Qa_alpha_tilde, batch_seg)
        sum_q_beta_tilde = segment_sum_coo(Qa_beta_tilde, batch_seg)
        raw_Q = sum_q_alpha_tilde + sum_q_beta_tilde
        raw_S = sum_q_alpha_tilde - sum_q_beta_tilde
        sum_f_alpha = segment_sum_coo(fa_alpha, batch_seg)
        sum_f_beta = segment_sum_coo(fa_beta, batch_seg)

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

        Q_alpha = segment_sum_coo(Qa_alpha, batch_seg)
        Q_beta = segment_sum_coo(Qa_beta, batch_seg)

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

