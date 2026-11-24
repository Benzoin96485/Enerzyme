"""
1. Normalize features of shape (N, sphere_basis, C),
with sphere_basis = (lmax + 1) ** 2.

2. The difference from `layer_norm.py` is that all type-L vectors have
the same number of channels and input features are of shape (N, sphere_basis, C).
"""

from typing import Optional

import torch
import torch.nn as nn

from ...nn.norm import EquivariantNorm, inverse_rms, rms_normalize
from ...nn.norm import degree_index as get_l_to_all_m_expand_index  # noqa: F401


def get_normalization_layer(
    norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
):
    """Factory for SH-array norms (EquiformerV2 + EquiformerV3 names)."""
    from functools import partial

    v2_types = ("layer_norm", "layer_norm_sh", "rms_norm_sh")
    v3_types = (
        "equivariant_layer_norm",
        "sep_layer_norm",
        "merge_layer_norm",
        "merge_layer_norm_attn_rms_norm",
        "merge_rms_norm",
    )
    assert norm_type in v2_types + v3_types, f"Unknown norm_type={norm_type!r}"
    if norm_type == "layer_norm":
        norm_class = EquivariantLayerNormArray
    elif norm_type == "layer_norm_sh":
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == "rms_norm_sh":
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    elif norm_type == "equivariant_layer_norm":
        norm_class = EquivariantLayerNormArray
    elif norm_type == "sep_layer_norm":
        norm_class = EquivariantSeparableLayerNorm
    elif norm_type in ("merge_layer_norm", "merge_layer_norm_attn_rms_norm"):
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == "merge_rms_norm":
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError(norm_type)
    return norm_class(lmax, num_channels, eps, affine, normalization)


class EquivariantLayerNormArray(EquivariantNorm):
    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__(
            lmax,
            num_channels,
            scope="per_degree",
            eps=eps,
            affine=affine,
            normalization=normalization,
        )


class EquivariantLayerNormArraySphericalHarmonics(EquivariantNorm):
    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__(
            lmax,
            num_channels,
            scope="separable",
            eps=eps,
            affine=affine,
            normalization=normalization,
            degree_balance=std_balance_degrees,
            channels_first=False,
        )


class EquivariantRMSNormArraySphericalHarmonicsV2(EquivariantNorm):
    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        if normalization == "norm" and std_balance_degrees:
            raise ValueError(
                "rms_norm_sh with norm normalization requires std_balance_degrees=False"
            )
        super().__init__(
            lmax,
            num_channels,
            kind="rms",
            center_l0=centering,
            eps=eps,
            affine=affine,
            normalization=normalization,
            degree_balance=std_balance_degrees,
            channels_first=False,
        )


class EquivariantSeparableLayerNorm(EquivariantNorm):
    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__(
            lmax,
            num_channels,
            scope="separable",
            eps=eps,
            affine=affine,
            normalization=normalization,
            degree_balance=std_balance_degrees,
        )


class EquivariantMergeLayerNorm(EquivariantNorm):
    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
        centering=True,
    ):
        super().__init__(
            lmax,
            num_channels,
            center_l0=centering,
            eps=eps,
            affine=affine,
            normalization=normalization,
            degree_balance=std_balance_degrees,
        )


class RMSNorm(nn.Module):
    """Channel RMSNorm used for EquiformerV3 attention re-normalization."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def _norm(self, x):
        return rms_normalize(x, -1, self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps})"


class EquivariantDegreeRMSNorm(nn.Module):
    """Degree-balanced equivariant RMSNorm on packed ``(N, D, F, C)`` features.

    Used by DPA4 / SeZM (Li et al., arXiv:2606.02419). Distinct from
    Equiformer ``EquivariantRMSNormArraySphericalHarmonicsV2`` (``[N, D, C]``
    layout without an explicit focus axis).
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        n_focus: int = 1,
        eps: float = 1e-5,
        mmax: Optional[int] = None,
    ) -> None:
        super().__init__()
        from ..indexing import build_m_major_l_index, map_degree_idx

        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.mmax = None if mmax is None else int(mmax)
        self.scale = nn.Parameter(
            torch.ones(self.lmax + 1, self.n_focus, self.channels)
        )
        self.bias = nn.Parameter(torch.zeros(self.n_focus, self.channels))
        if self.mmax is None:
            expand = map_degree_idx(self.lmax)
        else:
            expand = build_m_major_l_index(self.lmax, self.mmax)
        self.register_buffer("expand_index", torch.as_tensor(expand, dtype=torch.long))
        weights = []
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for l in expand.tolist():
            weights.append(scale / (2 * int(l) + 1))
        self.register_buffer(
            "balance_weight", torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x0 = x[:, :1]
        xt = x[:, 1:]
        x0 = x0 - x0.mean(dim=-1, keepdim=True)
        bw = self.balance_weight.to(device=x.device, dtype=x.dtype)
        mean_var = (x0 * x0).sum(dim=(1, 3)) * bw[0]
        if xt.numel() > 0:
            mean_var = mean_var + ((xt * xt) * bw[1:].view(1, -1, 1, 1)).sum(dim=(1, 3))
        inv_rms = inverse_rms(mean_var, self.eps, "floor").view(
            x.shape[0], 1, self.n_focus, 1
        )
        x = torch.cat([x0, xt], dim=1) * inv_rms
        scale = self.scale.index_select(0, self.expand_index).to(dtype=x.dtype)
        x = x * scale.unsqueeze(0)
        x = torch.cat(
            [
                x[:, :1]
                + self.bias.to(dtype=x.dtype).view(1, 1, self.n_focus, self.channels),
                x[:, 1:],
            ],
            dim=1,
        )
        return x.to(dtype=in_dtype)
