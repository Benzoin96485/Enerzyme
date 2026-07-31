"""SO3LR-style charge / spin embedding (Unke et al., Nat. Commun. 2021).

Architecture-agnostic pre-core layer used by SO3LR stacks. Distinct from
SpookyNet :class:`ElectronicEmbedding`.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential, SiLU
from torch_scatter import segment_sum_coo

from . import BaseFFLayer


class ChargeSpinEmbeddingLayer(BaseFFLayer):
    """Graph-level charge or spin → per-atom embedding (SO3LR / SpookyNet paper).

    Expects Enerzyme fields:
    - ``Q``: total molecular charge (when ``attribute='charge'``)
    - ``S``: unpaired-electron count / multiplicity−1 (when ``attribute='spin'``)

    Outputs ``charge_embedding`` or ``spin_embedding`` for
    :class:`GatherAtomEmbedding`.
    """

    def __init__(
        self,
        dim_embedding: int,
        max_Za: int = 118,
        attribute: Literal["charge", "spin"] = "charge",
        activation_fn: Optional[Callable[[], Module]] = None,
    ) -> None:
        if attribute not in {"charge", "spin"}:
            raise ValueError("attribute must be 'charge' or 'spin'")
        input_fields = {"Za", "batch_seg", "Q" if attribute == "charge" else "S"}
        output_field = f"{attribute}_embedding"
        super().__init__(input_fields=input_fields, output_fields={output_field})
        self.attribute = attribute
        self.dim_embedding = dim_embedding
        self.max_Za = max_Za
        self.num_elements = max_Za + 1
        act = SiLU if activation_fn is None else activation_fn

        self.Wq = Linear(self.num_elements, dim_embedding, bias=False)
        self.Wk = Parameter(torch.empty(2, dim_embedding))
        self.Wv = Parameter(torch.empty(2, dim_embedding))
        self.sqrt_dim = dim_embedding ** 0.5
        self.mlp = Sequential(
            act(),
            Linear(dim_embedding, dim_embedding, bias=False),
            act(),
            Linear(dim_embedding, dim_embedding, bias=False),
        )
        self.reset_parameters()
        self.reset_field_name(**{output_field: output_field})

    def reset_parameters(self) -> None:
        std = 1.0 / (self.Wq.out_features ** 0.5)
        torch.nn.init.normal_(self.Wq.weight, mean=0.0, std=std)
        std_k = 1.0 / (self.Wk.size(1) ** 0.5)
        torch.nn.init.normal_(self.Wk, mean=0.0, std=std_k)
        torch.nn.init.normal_(self.Wv, mean=0.0, std=std_k)
        for m in self.mlp:
            if isinstance(m, Linear):
                std_m = 1.0 / (m.in_features ** 0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std_m)

    def _embed(self, Za: Tensor, psi: Tensor, batch_seg: Tensor) -> Tensor:
        if batch_seg is None:
            batch_seg = torch.zeros(Za.shape[0], dtype=torch.long, device=Za.device)
        if psi is None:
            psi = torch.zeros(
                int(batch_seg.max().item()) + 1 if batch_seg.numel() else 1,
                dtype=torch.get_default_dtype(),
                device=Za.device,
            )
        psi = psi.to(dtype=torch.get_default_dtype())
        one_hot = F.one_hot(Za.long().clamp(min=0, max=self.max_Za), self.num_elements).to(
            dtype=psi.dtype
        )
        q = self.Wq(one_hot)
        # Floor-div by +inf: 0 if psi >= 0, -1 if psi < 0 (indexes last Wk/Wv row).
        idx = torch.div(
            psi,
            torch.tensor(float("inf"), device=psi.device, dtype=psi.dtype),
            rounding_mode="floor",
        ).to(torch.long)
        idx_atom = idx[batch_seg]
        k = self.Wk[idx_atom]
        v = self.Wv[idx_atom]
        y = F.softplus((q * k).sum(dim=-1) / self.sqrt_dim)
        num_graphs = int(batch_seg.max().item()) + 1 if batch_seg.numel() else 1
        denom = segment_sum_coo(y, batch_seg, dim_size=num_graphs) + 1e-6
        att = psi[batch_seg] * y / denom[batch_seg]
        v_att = att[:, None] * v
        return v_att + self.mlp(v_att)

    def get_output(
        self,
        Za: Tensor,
        batch_seg: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
    ):
        psi = Q if self.attribute == "charge" else S
        emb = self._embed(Za, psi, batch_seg)
        return {f"{self.attribute}_embedding": emb}
