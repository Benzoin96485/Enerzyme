"""Rotational invariance of AllScAIP core atomic outputs Ea and Qa w.r.t. positions R.

Graph preprocessing uses float32 internally; the core must match that dtype on inputs.

The default AllScAIP steerable path (full spherical-harmonic edge channels and frequency
masks) is not guaranteed to yield exactly rotation-invariant per-atom scalars for arbitrary
weights. This regression test uses the **scalar angular limit** (only l=0 harmonics, no
frequency mask) where edge and node angular features depend on rotation-invariant quantities
only, so Ea and Qa must match under global SO(3).

Use ``measure_rotation_breaking_for_l_gt_0()`` to quantify how ``Ea``/``Qa`` drift under
SO(3) when ``lmax > 0`` (typical steerable setup). The l>0 tests use ``capsys.disabled()``
so deviation lines are printed even when assertions pass (without needing ``pytest -s``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enerzyme.models.allscaip.core import AllScAIPCore


def _random_so3(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Proper rotation matrix Q with det(Q) = +1."""
    a = torch.randn(3, 3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _make_invariant_limit_core(**kwargs) -> AllScAIPCore:
    """Small core with rotation-invariant angular features (l=0 only, no freq mask)."""
    defaults = dict(
        dim_embedding=64,
        num_layers=4,
        atten_num_heads=4,
        ffn_hidden_layer_multiplier=2,
        use_node_path=True,
        use_freq_mask=False,
        cutoff_sr=5.0,
        knn_k=12,
        edge_distance_expansion_size=8,
        edge_direction_expansion_size=1,
        node_direction_expansion_size=1,
        attn_num_freq=16,
    )
    defaults.update(kwargs)
    return AllScAIPCore(**defaults)


def _make_higher_l_core(
    *,
    edge_direction_expansion_size: int = 3,
    node_direction_expansion_size: int = 3,
    use_freq_mask: bool = True,
    **kwargs: object,
) -> AllScAIPCore:
    """Core with l>0 spherical-harmonic edge/node channels (lmax = size - 1)."""
    defaults = dict(
        dim_embedding=64,
        num_layers=4,
        atten_num_heads=4,
        ffn_hidden_layer_multiplier=2,
        use_node_path=True,
        use_freq_mask=use_freq_mask,
        cutoff_sr=5.0,
        knn_k=12,
        edge_distance_expansion_size=8,
        edge_direction_expansion_size=edge_direction_expansion_size,
        node_direction_expansion_size=node_direction_expansion_size,
        attn_num_freq=16,
    )
    defaults.update(kwargs)
    return AllScAIPCore(**defaults)


def _print_rotation_invariance_deviation(
    stats: dict[str, float],
    *,
    seed: int,
    N: int,
    use_freq_mask: bool,
    edge_direction_expansion_size: int,
    node_direction_expansion_size: int,
    label: str,
) -> None:
    """Emit deviation metrics to stdout (flush) for interactive or pytest runs."""
    lmax_e = edge_direction_expansion_size - 1
    lmax_n = node_direction_expansion_size - 1
    print(
        f"\n--- {label} ---\n"
        f"  seed={seed}  N={N}  use_freq_mask={use_freq_mask}\n"
        f"  edge_direction_expansion_size={edge_direction_expansion_size} (lmax_edge={lmax_e})  "
        f"node_direction_expansion_size={node_direction_expansion_size} (lmax_node={lmax_n})\n"
        f"  deviation from rotational invariance |output(R) - output(R@Q.T)|:\n"
        f"    max_abs_delta_ea  = {stats['max_abs_delta_ea']:.6g}\n"
        f"    max_abs_delta_qa  = {stats['max_abs_delta_qa']:.6g}\n"
        f"    rms_delta_ea      = {stats['rms_delta_ea']:.6g}\n"
        f"    rms_delta_qa      = {stats['rms_delta_qa']:.6g}\n"
        f"    max_rel_delta_ea  = {stats['max_rel_delta_ea']:.6g}\n"
        f"    max_rel_delta_qa  = {stats['max_rel_delta_qa']:.6g}",
        flush=True,
    )


def measure_rotation_breaking_for_l_gt_0(
    *,
    seed: int = 123,
    N: int = 20,
    hidden: int = 64,
    edge_direction_expansion_size: int = 3,
    node_direction_expansion_size: int = 3,
    use_freq_mask: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    print_stats: bool = True,
    stats_label: str = "AllScAIP rotation (l>0)",
) -> dict[str, float]:
    """Forward AllScAIPCore with lmax>0 and report how much ``Ea`` / ``Qa`` change under SO(3).

    With l>0, edge (and BOO node) features carry measurable orientation. Fixed ``Linear`` /
    attention frequency mixing does **not** contract to rotation-invariant per-atom scalars in
    general, so ``|Ea(R) - Ea(R@Q^T)|`` and the same for ``Qa`` are typically **non-zero** for
    random weights—even though interatomic distances and kNN order are unchanged.

    Returns
    -------
    dict
        ``max_abs_delta_ea``, ``max_abs_delta_qa``, ``rms_delta_ea``, ``rms_delta_qa``,
        ``max_rel_delta_ea`` (relative to ``max(|Ea|)``), ``max_rel_delta_qa``.

    If ``print_stats`` is True, the same numbers are printed to stdout (use
    ``pytest ... --capture=no`` or ``capsys.disabled()`` in the test to see them on pass).
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)

    core = _make_higher_l_core(
        edge_direction_expansion_size=edge_direction_expansion_size,
        node_direction_expansion_size=node_direction_expansion_size,
        use_freq_mask=use_freq_mask,
        dim_embedding=hidden,
        atten_num_heads=4,
    ).to(device=device, dtype=dtype)
    core.eval()

    Ra = torch.randn(N, 3, device=device, dtype=dtype) * 2.0
    Za = torch.randint(1, 10, (N,), device=device)
    batch_seg = torch.zeros(N, dtype=torch.long, device=device)
    atom_embedding = torch.randn(N, hidden, device=device, dtype=dtype)
    charge_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1
    spin_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1

    Q = _random_so3(device, dtype)
    Ra_rot = Ra @ Q.T

    with torch.no_grad():
        out0 = core.get_output(
            Ra, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )
        out1 = core.get_output(
            Ra_rot, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )

    d_ea = (out0["Ea"] - out1["Ea"]).abs()
    d_qa = (out0["Qa"] - out1["Qa"]).abs()
    denom_ea = out0["Ea"].abs().max().clamp_min(torch.finfo(dtype).eps)
    denom_qa = out0["Qa"].abs().max().clamp_min(torch.finfo(dtype).eps)

    stats: dict[str, float] = {
        "max_abs_delta_ea": float(d_ea.max().item()),
        "max_abs_delta_qa": float(d_qa.max().item()),
        "rms_delta_ea": float((d_ea.square().mean().sqrt()).item()),
        "rms_delta_qa": float((d_qa.square().mean().sqrt()).item()),
        "max_rel_delta_ea": float((d_ea / denom_ea).max().item()),
        "max_rel_delta_qa": float((d_qa / denom_qa).max().item()),
    }
    if print_stats:
        _print_rotation_invariance_deviation(
            stats,
            seed=seed,
            N=N,
            use_freq_mask=use_freq_mask,
            edge_direction_expansion_size=edge_direction_expansion_size,
            node_direction_expansion_size=node_direction_expansion_size,
            label=stats_label,
        )
    return stats


def test_allscaip_core_ea_qa_rotation_invariant_scalar_angular_limit():
    """Ea and Qa invariant under R -> R @ Q.T when angular features are scalar-only."""
    device = torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(42)
    N = 20
    hidden = 64

    core = _make_invariant_limit_core().to(device=device, dtype=dtype)
    core.eval()

    Ra = torch.randn(N, 3, device=device, dtype=dtype) * 2.0
    Za = torch.randint(1, 10, (N,), device=device)
    batch_seg = torch.zeros(N, dtype=torch.long, device=device)
    atom_embedding = torch.randn(N, hidden, device=device, dtype=dtype)
    charge_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1
    spin_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1

    Q = _random_so3(device, dtype)
    Ra_rot = Ra @ Q.T

    with torch.no_grad():
        out0 = core.get_output(
            Ra, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )
        out1 = core.get_output(
            Ra_rot, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )

    rtol, atol = 5e-4, 5e-5
    assert_allclose(
        out0["Ea"].cpu().numpy(),
        out1["Ea"].cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="Ea not rotation-invariant (scalar angular limit)",
    )
    assert_allclose(
        out0["Qa"].cpu().numpy(),
        out1["Qa"].cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="Qa not rotation-invariant (scalar angular limit)",
    )


def test_allscaip_core_neighbor_order_stable_under_rotation():
    """Distances and kNN topology are unchanged by rotation."""
    device = torch.device("cpu")
    dtype = torch.float64
    torch.manual_seed(0)
    N = 12
    Ra = torch.randn(N, 3, device=device, dtype=dtype)
    Q = _random_so3(device, dtype)
    Ra_rot = Ra @ Q.T

    d0 = torch.cdist(Ra, Ra)
    d1 = torch.cdist(Ra_rot, Ra_rot)
    assert_allclose(d0.cpu().numpy(), d1.cpu().numpy(), rtol=0, atol=1e-12)

    k = min(8, N - 1)
    d0_fill = d0.clone()
    d0_fill.fill_diagonal_(float("inf"))
    d1_fill = d1.clone()
    d1_fill.fill_diagonal_(float("inf"))
    idx0 = torch.topk(d0_fill, k=k, largest=False, dim=-1).indices.sort(dim=-1).values
    idx1 = torch.topk(d1_fill, k=k, largest=False, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(idx0, idx1)


def test_allscaip_l_gt_0_per_atom_outputs_change_under_rotation(capsys):
    """With l>0 harmonics (and default freq mask), Ea/Qa generally depend on global frame."""
    with capsys.disabled():
        stats = measure_rotation_breaking_for_l_gt_0(
            seed=123,
            use_freq_mask=True,
            stats_label="AllScAIP l>0 + use_freq_mask=True",
        )
    # Random init produces clearly visible frame dependence (not just float noise).
    assert stats["max_abs_delta_ea"] > 1e-3, stats
    assert stats["max_abs_delta_qa"] > 1e-3, stats


def test_allscaip_l_gt_0_without_freq_mask_still_breaks_from_steercore_edges(capsys):
    """Frequency mask off: edge SH still mixed by InputBlock Linear → typically not invariant."""
    with capsys.disabled():
        stats = measure_rotation_breaking_for_l_gt_0(
            seed=456,
            use_freq_mask=False,
            stats_label="AllScAIP l>0 + use_freq_mask=False",
        )
    assert stats["max_abs_delta_ea"] > 1e-3, stats
    assert stats["max_abs_delta_qa"] > 1e-3, stats


def test_allscaip_equivariant_mode_l_gt_0_rotation_invariant():
    """-equivariant_mode uses TP + irrep norms and disables freq mask; Ea/Qa invariant under SO(3)."""
    device = torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(7)
    N = 20
    hidden = 64

    core = AllScAIPCore(
        dim_embedding=hidden,
        num_layers=2,
        atten_num_heads=4,
        ffn_hidden_layer_multiplier=2,
        use_node_path=True,
        use_freq_mask=True,
        equivariant_mode=True,
        equivariant_tp_hidden=32,
        equivariant_tp_mul=8,
        cutoff_sr=5.0,
        knn_k=12,
        edge_distance_expansion_size=8,
        edge_direction_expansion_size=3,
        node_direction_expansion_size=3,
        attn_num_freq=16,
    ).to(device=device, dtype=dtype)
    core.eval()
    assert core._use_freq_mask_resolved is False

    Ra = torch.randn(N, 3, device=device, dtype=dtype) * 2.0
    Za = torch.randint(1, 10, (N,), device=device)
    batch_seg = torch.zeros(N, dtype=torch.long, device=device)
    atom_embedding = torch.randn(N, hidden, device=device, dtype=dtype)
    charge_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1
    spin_embedding = torch.randn(N, hidden, device=device, dtype=dtype) * 0.1

    Q = _random_so3(device, dtype)
    Ra_rot = Ra @ Q.T

    with torch.no_grad():
        out0 = core.get_output(
            Ra, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )
        out1 = core.get_output(
            Ra_rot, Za, batch_seg, atom_embedding, charge_embedding, spin_embedding
        )

    rtol, atol = 5e-3, 5e-4
    assert_allclose(
        out0["Ea"].cpu().numpy(),
        out1["Ea"].cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="Ea not rotation-invariant with equivariant_mode",
    )
    assert_allclose(
        out0["Qa"].cpu().numpy(),
        out1["Qa"].cpu().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg="Qa not rotation-invariant with equivariant_mode",
    )