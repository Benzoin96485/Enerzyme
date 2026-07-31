"""Can SimpleReadout / Dense + rescale reproduce LinearRS?

Empirical findings (pure even-scalar irreps ``Dx0e``):

1. ``LinearRS`` ``rescale=True`` only multiplies weights by ``1/sqrt(fan_in)`` at
   *init*. Runtime output rescale is commented out in upstream / our port.
2. After init, a ``DenseLayer`` with the correctly laid-out ``tp.weight`` matches
   ``LinearRS`` — **no** extra rescale needed.
3. Applying an extra ``* 1/sqrt(fan_in)`` (weights or forward) **breaks** that match.
4. Full official energy MLP still differs from ``SimpleReadout(two_layer)`` even
   with mapped Dense weights, because official ``Activation`` wraps SiLU in
   ``normalize2mom`` (~1.68×), while ``two_layer`` uses Enerzyme ``swish``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enerzyme.models.blocks.mlp import DenseLayer
from enerzyme.models.equiformer.fast_activation import Activation
from enerzyme.models.equiformer.tensor_product import LinearRS
from enerzyme.models.layers.readout import SimpleReadout


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, atol: float, msg: str) -> None:
    diff = (a - b).abs().max().item()
    assert diff <= atol, f"{msg}: max |a-b|={diff} > atol={atol}"


def _linear_rs_to_dense(lin: LinearRS, d_in: int, d_out: int) -> DenseLayer:
    """Map LinearRS (pure 0e) → DenseLayer with equivalent affine map.

    e3nn FCTP weight view is ``(mul_in, mul_attr=1, mul_out)``; Dense wants ``(out, in)``.
    """
    # weight_views: (d_in, 1, d_out) → transpose to (d_out, d_in)
    w_view = next(lin.tp.weight_views()).detach()
    assert tuple(w_view.shape) == (d_in, 1, d_out), w_view.shape
    w = w_view.squeeze(1).T.contiguous()
    b = lin.bias[0].detach().clone()
    return DenseLayer(d_in, d_out, use_bias=True, initial_weight=w, initial_bias=b)


def test_dense_matches_linear_rs_without_extra_rescale():
    """Pure 0e LinearRS ≡ Dense with shared weights; extra rescale hurts."""
    dtype = torch.float64
    torch.manual_seed(0)
    d_in, d_out, n = 64, 32, 16
    x = torch.randn(n, d_in, dtype=dtype)

    lin = (
        LinearRS(
            o3.Irreps(f"{d_in}x0e"),
            o3.Irreps(f"{d_out}x0e"),
            bias=True,
            rescale=True,
        )
        .to(dtype)
        .eval()
    )
    dense = _linear_rs_to_dense(lin, d_in, d_out).to(dtype)
    _assert_close(lin(x), dense(x), atol=1e-12, msg="Dense(tp.weight) vs LinearRS")

    fan_in = d_in  # Dx0e × 1x0e → Fx0e, uvw fan_in = D
    sqrt_k = fan_in**-0.5
    dense_extra = DenseLayer(
        d_in,
        d_out,
        use_bias=True,
        initial_weight=dense.weight.detach() * sqrt_k,
        initial_bias=dense.bias.detach().clone(),
    ).to(dtype)
    err_extra = (lin(x) - dense_extra(x)).abs().max().item()
    assert err_extra > 1e-3, (
        "extra 1/sqrt(fan_in) on Dense weights should break LinearRS match, "
        f"got maxdiff={err_extra}"
    )

    err_fwd = (lin(x) - dense(x) * sqrt_k).abs().max().item()
    assert err_fwd > 1e-3, (
        f"forward Dense*sqrt_k should not match LinearRS, got maxdiff={err_fwd}"
    )


def test_linear_rs_rescale_false_needs_no_init_mul():
    """With rescale=False, correctly laid-out tp.weight still matches Dense."""
    dtype = torch.float64
    torch.manual_seed(1)
    d_in, d_out, n = 16, 8, 5
    x = torch.randn(n, d_in, dtype=dtype)
    lin = (
        LinearRS(
            o3.Irreps(f"{d_in}x0e"),
            o3.Irreps(f"{d_out}x0e"),
            bias=True,
            rescale=False,
        )
        .to(dtype)
        .eval()
    )
    dense = _linear_rs_to_dense(lin, d_in, d_out).to(dtype)
    _assert_close(lin(x), dense(x), atol=1e-12, msg="rescale=False Dense match")


def test_simple_readout_two_layer_vs_linear_rs_head_with_mapped_weights():
    """Mapped Dense matches LinearRS layers; SimpleReadout(swish) ≠ official act."""
    dtype = torch.float64
    torch.manual_seed(2)
    d, n = 64, 10
    x = torch.randn(n, d, dtype=dtype)

    lin1 = LinearRS(o3.Irreps(f"{d}x0e"), o3.Irreps(f"{d}x0e"), rescale=True).to(dtype)
    act_official = Activation(o3.Irreps(f"{d}x0e"), acts=[torch.nn.SiLU()])
    lin2 = LinearRS(o3.Irreps(f"{d}x0e"), o3.Irreps("1x0e"), rescale=True).to(dtype)
    head_rs = torch.nn.Sequential(lin1, act_official, lin2).eval()

    d1 = _linear_rs_to_dense(lin1, d, d).to(dtype)
    d2 = _linear_rs_to_dense(lin2, d, 1).to(dtype)

    y_rs = head_rs(x).view(-1)
    y_dense_official_act = d2(act_official(d1(x))).view(-1)
    _assert_close(
        y_rs,
        y_dense_official_act,
        atol=1e-11,
        msg="Dense+official Activation vs LinearRS head",
    )

    class _Core:
        dim_feature_out = d

    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="two_layer",
        activation_fn="swish",
    ).to(dtype)
    with torch.no_grad():
        ro.head[0].weight.copy_(d1.weight)
        ro.head[0].bias.copy_(d1.bias)
        ro.head[1].weight.copy_(d2.weight)
        ro.head[1].bias.copy_(d2.bias)

    y_simple = ro.get_output(x)["Ea"]
    err_swish = (y_rs - y_simple).abs().max().item()
    assert err_swish > 1e-3, (
        "SimpleReadout(swish) should still differ from LinearRS+normalize2mom(SiLU) "
        f"even with mapped weights; got maxdiff={err_swish}"
    )


def test_equiformer_linear_rs_head_matches_official_energy_mlp():
    """head_type=equiformer_linear_rs reproduces LinearRS + normalize2mom(SiLU)."""
    dtype = torch.float64
    torch.manual_seed(3)
    d, n = 64, 12
    x = torch.randn(n, d, dtype=dtype)

    lin1 = LinearRS(o3.Irreps(f"{d}x0e"), o3.Irreps(f"{d}x0e"), rescale=True).to(dtype)
    act = Activation(o3.Irreps(f"{d}x0e"), acts=[torch.nn.SiLU()])
    lin2 = LinearRS(o3.Irreps(f"{d}x0e"), o3.Irreps("1x0e"), rescale=True).to(dtype)
    head_rs = torch.nn.Sequential(lin1, act, lin2).eval()

    class _Core:
        dim_feature_out = d

    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="equiformer_linear_rs",
    ).to(dtype)
    assert ro.head_type == "equiformer_linear_rs"
    ro.head.load_state_dict(head_rs.state_dict())

    y_rs = head_rs(x).view(-1)
    y_ro = ro.get_output(x)["Ea"]
    _assert_close(y_rs, y_ro, atol=1e-12, msg="equiformer_linear_rs vs official MLP")


def test_equiformer_linear_rs_accepts_shallow_ensemble():
    class _Core:
        dim_feature_out = 8

    n, ensemble = 5, 3
    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="equiformer_linear_rs",
        shallow_ensemble_size=ensemble,
    )
    out = ro.get_output(torch.randn(n, 8))
    assert out["Ea"].shape == (n, ensemble)
    assert torch.isfinite(out["Ea"]).all()

    # size=1 still returns per-atom scalars
    ro1 = SimpleReadout(
        output_fields={"Ea", "Qa"},
        built_layers=[_Core()],
        head_type="equiformer_linear_rs",
        shallow_ensemble_size=1,
    )
    out1 = ro1.get_output(torch.randn(n, 8))
    assert out1["Ea"].shape == (n,)
    assert out1["Qa"].shape == (n,)
