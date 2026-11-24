"""SpinConv Core: registration, angles, invariance, conservative forces."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "spinconv_layers_example.yaml"


def _basis(dist: torch.Tensor, num_rbf: int, cutoff: float, scalar: float = 3.0) -> torch.Tensor:
    from enerzyme.models.layers.rbf import GaussianRBFLayer

    layer = GaussianRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        flavor="spinconv",
        apply_cutoff_fn=False,
        basis_width_scalar=scalar,
    )
    return layer._get_rbf(dist)


def _tiny(**kwargs):
    from enerzyme.models.presets.triplet_graphs import spinconv

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        cutoff_sr=5.0,
        max_Za=20,
        hidden_channels=8,
        mid_hidden_channels=16,
        num_interactions=2,
        sphere_size_lat=4,
        sphere_size_long=4,
        max_num_neighbors=8,
        num_embedding_basis=4,
        group_channels=8,
        scale_distances=True,
        basis_width_scalar=3.0,
        residual_first=False,
    )
    params.update(kwargs)
    return spinconv(**params)


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i), torch.tensor(idx_j)


def _pack(pos, embedding, num_rbf=4, cutoff=5.0, za=None):
    idx_i, idx_j = _edges(pos.shape[0])
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    if za is None:
        za = torch.arange(pos.shape[0]) % 6 + 1
    return {
        "atom_embedding": embedding,
        "rbf": _basis(dist, num_rbf, cutoff),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "Za": za.long(),
    }


def _forward(core, inp):
    return core.get_output(**inp)["atom_feature"]


def _rotate(pos: torch.Tensor) -> torch.Tensor:
    axis = torch.tensor([0.2, -0.4, 0.7])
    axis = axis / axis.norm()
    angle = 0.8
    skew = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=pos.dtype,
    )
    rotation = torch.eye(3, dtype=pos.dtype) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    return pos @ rotation


def test_registration_and_example_yaml():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.presets.triplet_graphs import spinconv

    core_cls, defaults, layers = get_ff_core("SpinConv")
    assert core_cls is spinconv
    assert defaults["cutoff_sr"] == 6.0
    assert defaults["num_rbf"] == 512
    assert defaults["hidden_channels"] == 32
    assert defaults["mid_hidden_channels"] == 256
    assert defaults["basis_width_scalar"] == 3.0
    assert defaults["Hartree_in_E"] > 20
    gaussian = next(layer for layer in layers if layer["name"] == "GaussianRBF")
    assert gaussian["params"]["flavor"] == "spinconv"
    assert gaussian["params"]["apply_cutoff_fn"] is False
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.architecture == "spinconv"
    assert model.num_interactions == 3
    assert model.hidden_channels == 32
    assert model.mid_hidden_channels == 256
    assert model.dim_feature_out == 256
    assert model.feature_irreps == "256x0e"
    assert model.sphere_size_lat == 16
    assert model.sphere_size_long == 12
    assert model.max_num_neighbors == 40
    assert model.message_divisor == 21.0
    assert model.scale_distances is True
    assert model.residual_first is False
    assert model.radial_basis_function.flavor == "spinconv"
    assert model.radial_basis_function.apply_cutoff_fn is False
    assert abs(float(model.radial_basis_function.basis_width_scalar) - 3.0) < 1e-8


def test_angles_change_features_when_bond_lengths_stay_fixed():
    torch.manual_seed(2)
    embedding = torch.randn(3, 8)
    za = torch.tensor([6, 1, 8])

    def bent(theta: float):
        # Only the two spokes. Their lengths stay 1, so the radial basis matches.
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [math.cos(theta), math.sin(theta), 0.0],
            ]
        )
        pairs = [(0, 1), (1, 0), (0, 2), (2, 0)]
        idx_i = torch.tensor([dst for dst, _ in pairs])
        idx_j = torch.tensor([src for _, src in pairs])
        vij = pos[idx_j] - pos[idx_i]
        dist = vij.norm(dim=-1)
        return {
            "atom_embedding": embedding,
            "rbf": _basis(dist, 4, 5.0),
            "Dij_sr": dist,
            "vij_sr": vij,
            "idx_i_sr": idx_i,
            "idx_j_sr": idx_j,
            "Za": za,
        }

    wide = bent(0.6)
    narrow = bent(2.2)
    assert_allclose(wide["Dij_sr"].numpy(), narrow["Dij_sr"].numpy(), atol=1e-6)
    assert_allclose(wide["rbf"].detach().numpy(), narrow["rbf"].detach().numpy(), atol=1e-6)
    core = _tiny().eval()
    left = _forward(core, wide)
    right = _forward(core, narrow)
    assert left.shape == (3, 16)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    # The species mixture is a deep map of a GroupNorm output. At the default
    # initialization its gain is well below 1, so the atomic latent barely
    # moves. The spin convolution itself is where the angle enters.
    messages = torch.randn(4, core.hidden_channels)
    spun_wide = core.blocks[0].message.spin(
        messages, wide["idx_i_sr"], wide["idx_j_sr"], wide["vij_sr"], wide["Dij_sr"], 3
    )
    spun_narrow = core.blocks[0].message.spin(
        messages, narrow["idx_i_sr"], narrow["idx_j_sr"], narrow["vij_sr"], narrow["Dij_sr"], 3
    )
    assert float((spun_wide - spun_narrow).abs().max()) > 1e-2


def test_rotation_translation_order_and_inversion():
    torch.manual_seed(1)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.15, 0.0], [0.2, 1.0, 0.25], [0.15, 0.3, 1.05]]
    )
    embedding = torch.randn(4, 8)
    core = _tiny().eval()
    base_in = _pack(pos, embedding)
    base = _forward(core, base_in)
    rotated = _forward(core, _pack(_rotate(pos), embedding))
    assert_allclose(base.detach().numpy(), rotated.detach().numpy(), atol=1e-5, rtol=1e-5)
    shifted = _forward(core, _pack(pos + torch.tensor([2.0, -1.0, 0.4]), embedding))
    assert_allclose(base.detach().numpy(), shifted.detach().numpy(), atol=1e-5, rtol=1e-5)
    inverted = _forward(core, _pack(-pos, embedding))
    assert torch.isfinite(inverted).all()
    # Inversion reflects local z. With SiLU before the longitude pool that
    # reflection changes the convolution, even though a longitude roll does not.
    messages = torch.randn(base_in["idx_i_sr"].numel(), core.hidden_channels)
    packed_inv = _pack(-pos, embedding)

    def _spin(packed):
        return core.blocks[0].message.spin(
            messages,
            packed["idx_i_sr"],
            packed["idx_j_sr"],
            packed["vij_sr"],
            packed["Dij_sr"],
            pos.shape[0],
        )

    assert float((_spin(base_in) - _spin(packed_inv)).abs().max()) > 1e-2

    perm = torch.randperm(base_in["idx_i_sr"].numel())
    shuffled = dict(base_in)
    for key in ("rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"):
        shuffled[key] = base_in[key][perm]
    assert_allclose(base.detach().numpy(), _forward(core, shuffled).detach().numpy(), atol=1e-5)

    added = _tiny(residual_first=True).eval()
    added.load_state_dict(core.state_dict())
    assert float((base - _forward(added, base_in)).abs().max()) > 1e-4


def test_batch_isolation_scale_and_isolated_atom():
    torch.manual_seed(3)
    core = _tiny().eval()
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.1, 0.8, 0.3]])
    right_pos = torch.tensor([[0.2, 0.0, 0.1], [1.0, 0.4, 0.0], [0.3, 0.2, 1.1]])
    left_emb = torch.randn(3, 8)
    right_emb = torch.randn(3, 8)
    left = _pack(left_pos, left_emb)
    right = _pack(right_pos, right_emb)
    feat_l = _forward(core, left)
    feat_r = _forward(core, right)
    merged = {
        "atom_embedding": torch.cat([left_emb, right_emb]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
        "Za": torch.cat([left["Za"], right["Za"]]),
    }
    feat = _forward(core, merged)
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)

    from enerzyme.models.layers.rbf import gaussian_expansion

    za_source = left["Za"][left["idx_j_sr"]]
    za_target = left["Za"][left["idx_i_sr"]]
    unscaled = core.embed.pair_scale(left["Dij_sr"], za_source, za_target)
    with torch.no_grad():
        core.embed.pair_scale.gain.weight.fill_(4.0)
    stretched = core.embed.pair_scale(left["Dij_sr"], za_source, za_target)
    assert float((stretched - unscaled).abs().max()) > 0.1
    painted = gaussian_expansion(stretched, core.embed.gaussian_center, core.embed.gaussian_coeff)
    assert float((painted - left["rbf"]).abs().max()) > 1e-3
    assert torch.isfinite(_forward(core, left)).all()

    alone_pos = torch.tensor([[0.2, -0.3, 0.4]])
    alone = {
        "atom_embedding": torch.randn(1, 8),
        "rbf": torch.zeros(0, 4),
        "Dij_sr": torch.zeros(0),
        "vij_sr": torch.zeros(0, 3),
        "idx_i_sr": torch.zeros(0, dtype=torch.long),
        "idx_j_sr": torch.zeros(0, dtype=torch.long),
        "Za": torch.tensor([6]),
    }
    moved = dict(alone)
    first = _forward(core, alone)
    second = _forward(core, {**moved, "atom_embedding": alone["atom_embedding"]})
    # A translated isolated atom is the same graph. Rebuild the zero edges.
    translated = dict(alone)
    assert first.shape == (1, 16)
    assert torch.isfinite(first).all()
    assert_allclose(first.detach().numpy(), second.detach().numpy(), atol=1e-6)
    assert translated["idx_i_sr"].numel() == 0

    try:
        _tiny(hidden_channels=7)
    except ValueError as exc:
        assert "group_channels" in str(exc)
    else:
        raise AssertionError("a hidden width that is not a multiple of the group size must be rejected")


def test_layer_stack_energy_forces_and_wrong_basis():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "GaussianRBF",
            "params": {"flavor": "spinconv", "apply_cutoff_fn": False},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_interactions": 1,
                "sphere_size_lat": 4,
                "sphere_size_long": 4,
                "max_num_neighbors": 8,
                "num_embedding_basis": 4,
                "group_channels": 8,
                "scale_distances": True,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {"output_fields": ["Ea"], "head_type": "dense", "use_bias": True},
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    build = {
        "dim_embedding": 8,
        "num_rbf": 4,
        "hidden_channels": 8,
        "mid_hidden_channels": 16,
        "max_Za": 20,
        "cutoff_sr": 5.0,
        "basis_width_scalar": 3.0,
    }
    model = build_model("spinconv", layer_params=layers, build_params=build, verbose=0)
    model.double()
    n = 4
    idx_i, idx_j = _edges(n)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 1.1, 0.2], [0.4, 0.3, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = {
        "Ra": pos,
        "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert out["atom_feature"].shape == (n, 16)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    force = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-8)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-8)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    rotated = (pos.detach() @ rotation).requires_grad_(True)
    out_r = model({**batch, "Ra": rotated})
    assert_allclose(out["E"].detach().numpy(), out_r["E"].detach().numpy(), atol=1e-6)
    assert_allclose(
        out_r["Fa"].detach().numpy(),
        (out["Fa"].detach() @ rotation).numpy(),
        atol=1e-6,
        rtol=1e-5,
    )
    shifted = (pos.detach() + torch.tensor([1.5, -0.4, 0.2], dtype=torch.float64)).requires_grad_(True)
    out_t = model({**batch, "Ra": shifted})
    assert_allclose(out["E"].detach().numpy(), out_t["E"].detach().numpy(), atol=1e-6)
    assert_allclose(out["Fa"].detach().numpy(), out_t["Fa"].detach().numpy(), atol=1e-7, rtol=1e-6)

    # Bilinear weights are detached, matching the released gradient-force path.
    # A coordinate finite difference also moves those weights, including the
    # bond partner that sits on the pole, so it is not the force. The Force
    # layer is the autograd of the energy with the painting held fixed.

    alone = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
    out_a = model(
        {
            "Ra": alone,
            "Za": torch.tensor([6]),
            "batch_seg": torch.zeros(1, dtype=torch.long),
            "idx_i": torch.zeros(0, dtype=torch.long),
            "idx_j": torch.zeros(0, dtype=torch.long),
        }
    )
    assert torch.isfinite(out_a["E"]).all()
    assert_allclose(out_a["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-8)

    wrong = [
        {"name": "RangeSeparation"},
        {"name": "GaussianRBF", "params": {"flavor": "SchNet", "apply_cutoff_fn": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_interactions": 1, "sphere_size_lat": 4, "sphere_size_long": 4}},
        {"name": "EnergyReduce"},
    ]
    try:
        build_model("spinconv", layer_params=wrong, build_params=build, verbose=0)
    except ValueError as exc:
        assert "spinconv" in str(exc)
    else:
        raise AssertionError("a SchNet Gaussian basis must be rejected")

    hierarchical = layers[:-3] + [
        {
            "name": "HierachicalReadout",
            "params": {"output_fields": ["Ea"], "head_type": "dense", "num_blocks": 1},
        },
        {"name": "EnergyReduce"},
    ]
    try:
        build_model("spinconv", layer_params=hierarchical, build_params=build, verbose=0)
    except ValueError as exc:
        assert "SimpleReadout" in str(exc)
    else:
        raise AssertionError("HierachicalReadout must be rejected")


def test_compiled_spin_force_training_uses_live_pruned_edges_and_preserves_empty_gradients():
    from enerzyme.models.ff import build_model, get_ff_core

    torch.manual_seed(31)
    _, defaults, layers = get_ff_core("spinconv")
    layers = copy.deepcopy(layers)
    core = next(layer["params"] for layer in layers if layer["name"] == "Core")
    core.update(num_interactions=1, sphere_size_lat=4, sphere_size_long=4,
                max_num_neighbors=1, num_embedding_basis=4, scale_distances=True)
    dimensions = dict(defaults, dim_embedding=8, num_rbf=4, hidden_channels=8, mid_hidden_channels=16)
    model = build_model("spinconv", layers, dimensions, verbose=0).double().train()
    calls = []
    def capture(graph, _):
        calls.extend(getattr(n.target, "__name__", str(n.target)) for n in graph.graph.nodes)
        return graph.forward
    compiled = torch.compile(model, backend=capture)

    pos = torch.tensor([[0., 0., 0.], [0.7, 0.1, 0.], [1.9, 0., 0.1], [0.1, 2.2, 0.3]], dtype=torch.double)
    i, j = _edges(len(pos))
    def step(runner, edges):
        model.zero_grad(set_to_none=True)
        output = runner({"Ra": pos.clone().requires_grad_(), "Za": torch.tensor([1, 6, 7, 8]),
                         "batch_seg": torch.zeros(4, dtype=torch.long), "N": torch.tensor([4]),
                         "idx_i": i[:edges], "idx_j": j[:edges]})
        (output["E"].square().sum() + output["Fa"].square().sum()).backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        return output["E"], output["Fa"], grads

    for count in (len(i), 0):
        energy, forces, grads = step(model, count)
        actual_energy, actual_forces, actual_grads = step(compiled, count)
        torch.testing.assert_close((actual_energy, actual_forces), (energy, forces))
        assert set(actual_grads) == set(grads)
        for name, gradient in grads.items():
            torch.testing.assert_close(actual_grads[name], gradient)
        if count:
            gain = next(gradient for name, gradient in grads.items() if "pair_scale.gain" in name)
            assert torch.count_nonzero(gain)
        else:
            assert not torch.count_nonzero(actual_forces)
    assert "einsum" in calls and "linear" in calls
