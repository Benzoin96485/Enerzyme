# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License)
# into Enerzyme with package-local imports.
"""Equivariant layer norm for flat e3nn irreps features."""

import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class EquivariantLayerNormV2(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"], (
            "normalization needs to be 'norm' or 'component'"
        )
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, **kwargs):
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d)

            if ir.l == 0 and ir.p == 1:
                field_mean = torch.mean(field, dim=1, keepdim=True)
                field = field - field_mean

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            field_norm = (field_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]
                iw += mul
                field_norm = field_norm * weight

            field = field * field_norm.reshape(-1, mul, 1)

            if self.affine and d == 1 and ir.p == 1:
                bias = self.affine_bias[ib : ib + mul]
                ib += mul
                field += bias.reshape(mul, 1)

            fields.append(field.reshape(-1, mul * d))

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            raise AssertionError(fmt.format(dim, ix))

        return torch.cat(fields, dim=-1)
