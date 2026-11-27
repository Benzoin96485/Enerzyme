"""BOTNet radial basis and interaction wiring."""

from __future__ import annotations

import math

import torch
from e3nn.o3 import Irreps


def test_mace_bessel_matches_botnet_polynomial():
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 4.0
    degree = 6
    num_basis = 5
    layer = BesselRBFLayer(
        num_rbf=num_basis,
        cutoff_sr=cutoff,
        flavor="mace",
        trainable=False,
        polynomial_p=degree,
    )
    assert layer.apply_cutoff_fn is False
    distances = torch.tensor([0.0, 0.3, 1.4, 3.9, 4.0, 5.5])
    got = layer.get_rbf(distances)
    prefactor = math.sqrt(2.0 / cutoff)
    a = -(degree + 1) * (degree + 2) / 2
    b = degree * (degree + 2)
    c = -degree * (degree + 1) / 2
    expected = []
    for dist in distances.tolist():
        if dist >= cutoff:
            expected.append(torch.zeros(num_basis))
            continue
        scaled = dist / cutoff
        env = 1.0 + a * scaled**degree + b * scaled ** (degree + 1) + c * scaled ** (degree + 2)
        row = []
        for n in range(1, num_basis + 1):
            weight = n * math.pi / cutoff
            quotient = weight if dist == 0.0 else math.sin(weight * dist) / dist
            row.append(prefactor * quotient * env)
        expected.append(torch.tensor(row))
    assert torch.allclose(got, torch.stack(expected), atol=1e-6, rtol=1e-6)


def test_message_is_written_on_idx_j_and_matches_the_interaction_block():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(0)
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=1,
        radial_MLP=[8],
        avg_num_neighbors=1.0,
    )
    core.eval()
    embedding = torch.randn(2, 4)
    idx_i = torch.tensor([0])
    idx_j = torch.tensor([1])
    vij = torch.tensor([[0.8, -0.2, 0.1]])
    rbf = torch.randn(1, 3)
    atomic_numbers = torch.tensor([6, 1])
    out = core.get_output(
        Za=atomic_numbers,
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=rbf,
        atom_embedding=embedding,
    )["atom_feature"]
    assert out.shape[0] == 2
    assert out.shape[-1] == 1
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(out[1]) > 0

    node_attrs = torch.nn.functional.one_hot(atomic_numbers, num_classes=9).to(embedding.dtype)
    block = core.blocks[0].module(
        node_attrs=node_attrs,
        node_feats=embedding,
        edge_attrs=core.embed.sph(vij),
        edge_feats=rbf,
        sender=idx_i,
        receiver=idx_j,
    )
    assert torch.allclose(block, out[:, :, 0])


def test_natural_parity_layers_share_one_layout():
    from enerzyme.models.botnet import natural_hidden_irreps
    from enerzyme.models.presets.irreps import botnet

    hidden = natural_hidden_irreps(4, 1)
    assert hidden == str(Irreps("4x0e+4x1o"))
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        hidden_irreps=hidden,
        radial_MLP=[8],
    )
    assert core.uniform_irreps
    assert core.feature_irreps == hidden
    assert core.last_readout == "nonlinear"
    assert core.num_output_blocks == 2


def test_later_layers_may_grow_unnatural_parity_and_early_channels_stay_zero():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(1)
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        hidden_irreps="4x0e+4x0o+4x1o+4x1e",
        radial_MLP=[8],
        avg_num_neighbors=2.0,
    )
    assert core.uniform_irreps is False
    assert core.head.placement_1.numel() > core.head.placement_0.numel()
    n = 4
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    pos = torch.randn(n, 3)
    out = core.get_output(
        Za=torch.tensor([6, 1, 8, 1]),
        vij_sr=pos[idx_j] - pos[idx_i],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 3),
        atom_embedding=torch.randn(n, 4),
    )["atom_feature"]
    present = set(core.head.placement_0.tolist())
    missing = [column for column in range(out.shape[1]) if column not in present]
    assert missing
    assert torch.count_nonzero(out[:, missing, 0]) == 0
    assert torch.count_nonzero(out[:, missing, 1]) > 0


def test_hidden_width_and_variant_are_checked():
    from enerzyme.models.presets.irreps import botnet

    try:
        botnet(
            max_Za=8,
            dim_embedding=4,
            num_rbf=3,
            max_ell=1,
            hidden_irreps="8x0e+8x1o",
        )
    except ValueError as exc:
        assert "0e" in str(exc)
    else:
        raise AssertionError("mismatched 0e multiplicity should be rejected")

    try:
        botnet(max_Za=8, dim_embedding=4, num_rbf=3, max_ell=1, variant="mace")
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("unknown variant should be rejected")

    try:
        botnet(
            max_Za=8,
            dim_embedding=4,
            num_rbf=3,
            max_ell=1,
            interaction_cls_first="RealAgnosticResidualInteractionBlock",
        )
    except KeyError as exc:
        assert "RealAgnostic" in str(exc)
    else:
        raise AssertionError("MACE product blocks are not BOTNet interactions")


def test_features_are_rotation_and_inversion_equivariant():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(2)
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        radial_MLP=[8],
        avg_num_neighbors=2.0,
    ).double()
    core.eval()
    n = 5
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    generator = torch.Generator().manual_seed(3)
    pos = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    embedding = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    rbf = torch.randn(idx_i.numel(), 3, generator=generator, dtype=torch.float64)
    atomic_numbers = torch.tensor([6, 1, 8, 1, 7])

    def run(positions):
        return core.get_output(
            Za=atomic_numbers,
            vij_sr=positions[idx_j] - positions[idx_i],
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            rbf=rbf,
            atom_embedding=embedding,
        )["atom_feature"]

    base = run(pos)
    irreps = Irreps(core.feature_irreps)
    rotation = _rotation(4).double()
    rotated = run(pos @ rotation)
    transformed = base.permute(0, 2, 1) @ irreps.D_from_matrix(rotation)
    assert torch.allclose(transformed, rotated.permute(0, 2, 1), atol=1e-5, rtol=1e-5)

    inversion = -torch.eye(3, dtype=torch.float64)
    inverted = run(pos @ inversion)
    parity = base.permute(0, 2, 1) @ irreps.D_from_matrix(inversion)
    assert torch.allclose(parity, inverted.permute(0, 2, 1), atol=1e-5, rtol=1e-5)
    shifted = run(pos + torch.tensor([2.0, -1.0, 0.4], dtype=torch.float64))
    assert torch.allclose(shifted, base, atol=1e-5, rtol=1e-5)


def test_isolated_atom_is_zero_until_the_first_layer_is_residual():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(5)
    kwargs = dict(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        radial_MLP=[8],
    )
    agnostic = botnet(**kwargs)
    agnostic.eval()
    empty = dict(
        Za=torch.tensor([8]),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        rbf=torch.zeros(0, 3),
        atom_embedding=torch.randn(1, 4),
    )
    alone = agnostic.get_output(**empty)["atom_feature"]
    assert torch.count_nonzero(alone) == 0

    residual = botnet(**kwargs, variant="residual_element")
    residual.eval()
    kept = residual.get_output(**empty)["atom_feature"]
    assert torch.isfinite(kept).all()
    assert torch.count_nonzero(kept) > 0


def _rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix
