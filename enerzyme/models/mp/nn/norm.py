"""Normalization of scalar channels and degree-ordered spherical feature arrays."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def inverse_rms(moment: Tensor, eps: float, eps_mode: str = "inside") -> Tensor:
    """Inverse RMS with the architecture's additive epsilon or variance floor."""
    if eps_mode == "inside":
        return torch.rsqrt(moment + eps)
    if eps_mode == "outside":
        # The composed norm has a finite derivative at a zero input. Keep
        # sqrt's infinite derivative at zero out of its backward pass.
        zero = moment == 0
        root = moment.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)
        return (root + eps).reciprocal()
    if eps_mode == "floor":
        return torch.rsqrt(moment.clamp_min(eps))
    raise ValueError(f"unknown eps_mode {eps_mode!r}")


def rms_normalize(x: Tensor, dims, eps: float, eps_mode: str = "inside") -> Tensor:
    return x * inverse_rms(x.square().mean(dim=dims, keepdim=True), eps, eps_mode)


def balanced_degree_moment(
    values: Tensor, weights: Tensor, channels_first: bool
) -> Tensor:
    """Weighted sum of spherical component variances, averaged over channels."""
    square = values.square()
    if channels_first:
        return torch.einsum(
            "ai,nic->nac", weights.reshape(1, -1), square.mean(dim=2, keepdim=True)
        )
    moment = torch.einsum("nic,ia->nac", square, weights.reshape(-1, 1))
    return moment.mean(dim=2, keepdim=True)


def flat_rms_scale(
    values: Tensor, weights: Tensor, eps: float, divisor: int = 1
) -> Tensor:
    """One RMS scale over flat components with representation-dependent weights."""
    moment = (values.square() * weights).sum(dim=-1, keepdim=True)
    return inverse_rms(moment / divisor if divisor != 1 else moment, eps)


def degree_rms_scale(
    values: Tensor,
    component_dim: int,
    channel_dim: int,
    normalization: str,
    eps: float,
    eps_mode: str = "inside",
) -> Tensor:
    """Scale for one irrep block in either component-major or channel-major order."""
    square = values.square()
    moment = (
        square.sum(dim=component_dim, keepdim=True)
        if normalization == "norm"
        else square.mean(dim=component_dim, keepdim=True)
    )
    return inverse_rms(moment.mean(dim=channel_dim, keepdim=True), eps, eps_mode)


def feature_norm(name: str | None, width: int) -> nn.Module | None:
    if name in (None, "none", "null"):
        return None
    if name == "layer":
        return nn.LayerNorm(width)
    if name == "batch":
        return nn.BatchNorm1d(width)
    raise ValueError(f"normalization must be 'layer', 'batch', or None, got {name!r}")


def degree_index(lmax: int) -> Tensor:
    """Degree of each component in the packed l-major order."""
    return torch.tensor(
        [l for l in range(lmax + 1) for _ in range(2 * l + 1)], dtype=torch.long
    )


class EquivariantNorm(nn.Module):
    """Normalize a spherical array of shape (nodes, (lmax + 1)², channels).

    Per-degree normalization uses a separate moment for each degree. Separable
    normalization reserves channel normalization for the scalar and shares
    the higher-degree moment. Merged normalization uses one moment across
    all degrees. Global normalization averages all components uniformly.
    Centering and bias only apply to l=0 and all gains are tied across m.

    The two reduction orders preserve the reference models' buffer layout and
    arithmetic. Degree balancing gives each degree equal weight regardless of
    its number of components.
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        *,
        kind: str = "layer",
        scope: str = "merged",
        center_l0: bool | None = None,
        degree_balance: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
        eps_mode: str = "inside",
        normalization: str = "component",
        channels_first: bool = True,
    ):
        super().__init__()
        if kind not in ("layer", "rms") or scope not in (
            "per_degree",
            "separable",
            "merged",
            "global",
        ):
            raise ValueError(f"unknown equivariant norm kind={kind!r}, scope={scope!r}")
        if normalization not in ("norm", "component") or eps_mode not in (
            "inside",
            "outside",
            "floor",
        ):
            raise ValueError(
                f"unknown normalization={normalization!r} or eps_mode={eps_mode!r}"
            )
        if lmax < 0 or num_channels < 1 or eps < 0:
            raise ValueError(
                "lmax and epsilon must be nonnegative and num_channels positive"
            )
        self.lmax = lmax
        self.num_channels = num_channels
        self.scope = scope
        self.centering = kind == "layer" if center_l0 is None else center_l0
        self.std_balance_degrees = degree_balance and scope not in (
            "global",
            "per_degree",
        )
        self.affine = affine
        self.eps = eps
        self.eps_mode = eps_mode
        self.normalization = normalization
        self.channels_first = channels_first
        self.widths = tuple(2 * degree + 1 for degree in range(lmax + 1))
        separate = scope == "separable"
        first_degree = int(separate)
        self.norm_l0 = (
            nn.LayerNorm(
                num_channels,
                eps=eps,
                elementwise_affine=affine,
                bias=self.centering,
            )
            if separate
            else None
        )
        self.affine_weight = (
            nn.Parameter(torch.ones(lmax + 1 - first_degree, num_channels))
            if affine
            else None
        )
        self.affine_bias = (
            nn.Parameter(torch.zeros(num_channels))
            if affine and self.centering and not separate
            else None
        )

        if scope != "per_degree":
            index = degree_index(lmax)[first_degree:] - first_degree
            # The original component-first separable norm stored only balance weights.
            persistent = affine if channels_first else not separate
            self.register_buffer("expand_index", index, persistent=persistent)
        if self.std_balance_degrees:
            widths = self.widths[first_degree:]
            balance = torch.tensor(
                [1.0 / width for width in widths for _ in range(width)]
            )
            balance = balance / max(len(widths), 1)
            shape = (1, -1) if channels_first else (-1, 1)
            self.register_buffer("balance_degree_weight", balance.reshape(shape))
        else:
            self.balance_degree_weight = None

    def _shared_scale(self, x: Tensor) -> Tensor:
        if self.std_balance_degrees and self.normalization == "component":
            moment = balanced_degree_moment(
                x, self.balance_degree_weight, self.channels_first
            )
        else:
            square = x.square()
            if self.channels_first:
                square = square.mean(dim=2, keepdim=True)
            moment = (
                square.sum(dim=1, keepdim=True)
                if self.normalization == "norm"
                else square.mean(dim=1, keepdim=True)
            )
            if not self.channels_first:
                moment = moment.mean(dim=2, keepdim=True)
        scale = inverse_rms(moment, self.eps, self.eps_mode)
        if self.affine_weight is not None:
            scale = scale * self.affine_weight.index_select(
                0, self.expand_index
            ).unsqueeze(0)
        return scale

    def _per_degree(self, x: Tensor) -> Tensor:
        output = []
        for degree, feature in enumerate(x.split(self.widths, dim=1)):
            if degree == 0 and self.centering:
                feature = feature - feature.mean(dim=-1, keepdim=True)
            scale = degree_rms_scale(
                feature, 1, -1, self.normalization, self.eps, self.eps_mode
            )
            if self.affine_weight is not None:
                scale = scale * self.affine_weight[degree]
            feature = feature * scale
            if degree == 0 and self.affine_bias is not None:
                feature = feature + self.affine_bias
            output.append(feature)
        return torch.cat(output, dim=1)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        if self.scope == "per_degree":
            return self._per_degree(x)
        scalar = x[:, :1]
        if self.norm_l0 is not None:
            if self.centering and self.eps_mode == "inside":
                scalar = self.norm_l0(scalar)
            else:
                if self.centering:
                    scalar = scalar - scalar.mean(dim=-1, keepdim=True)
                scalar = rms_normalize(scalar, -1, self.eps, self.eps_mode)
                if self.affine:
                    scalar = scalar * self.norm_l0.weight
                    if self.centering:
                        scalar = scalar + self.norm_l0.bias
            if self.lmax == 0:
                return scalar
            higher = x[:, 1:]
            return torch.cat((scalar, higher * self._shared_scale(higher)), dim=1)
        if self.centering:
            x = torch.cat((scalar - scalar.mean(dim=-1, keepdim=True), x[:, 1:]), dim=1)
        result = x * self._shared_scale(x)
        if self.affine_bias is not None:
            result[:, :1] = result[:, :1] + self.affine_bias
        return result
