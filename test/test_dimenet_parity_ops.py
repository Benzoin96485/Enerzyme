"""Operator-level numerical parity vs torch_geometric original DimeNet.

Not DimeNet++. ``DimeNet.forward`` is not called: PyG needs torch-cluster and
torch-sparse, which are optional. Tests use PyG Envelope / BesselBasis /
SphericalBasis / Embedding / Residual / Interaction / Output modules.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.extend(["..", "."])

from dimenet_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    SWISH_PARAMS,
    assert_close,
    copy_embedding_block,
    copy_interaction,
    copy_linear_to_dense,
    copy_output_head,
    copy_pyg_dimenet_into_enerzyme,
    copy_residual,
    enerzyme_energy,
    make_parity_graph,
    make_tiny_enerzyme_stack,
    make_tiny_pyg_dimenet,
    pyg_dimenet_energy,
    pyg_swish,
    triplet_indices,
)


@pytest.fixture(scope="module")
def pyg_dimenet():
    return make_tiny_pyg_dimenet()


def test_envelope_matches_pyg():
    from torch_geometric.nn.models.dimenet import Envelope as PygEnvelope

    from enerzyme.models.so3.envelope import DimeNetEnvelope

    dtype = torch.float64
    x = torch.linspace(1e-3, 1.2, 64, dtype=dtype)
    pyg = PygEnvelope(PARITY_HPARAMS["envelope_exponent"])
    ours = DimeNetEnvelope(PARITY_HPARAMS["envelope_exponent"])
    assert_close(ours(x), pyg(x), atol=1e-12, rtol=0.0, err_msg="envelope")


def test_bessel_rbf_dimenet_flavor_matches_pyg():
    from torch_geometric.nn.models.dimenet import BesselBasisLayer as PygBessel

    from enerzyme.models.layers.rbf import BesselRBFLayer

    hp = PARITY_HPARAMS
    dtype = torch.float64
    pyg = PygBessel(hp["num_radial"], hp["cutoff"], hp["envelope_exponent"]).to(dtype)
    ours = BesselRBFLayer(
        num_rbf=hp["num_radial"],
        cutoff_sr=hp["cutoff"],
        flavor="dimenet",
        trainable=True,
        envelope_exponent=hp["envelope_exponent"],
    ).to(dtype)
    assert_close(ours.bessel_weights, pyg.freq, atol=0.0, rtol=0.0, err_msg="freq")
    dist = torch.tensor([0.4, 1.2, 2.5, 4.9, 5.2], dtype=dtype)
    assert_close(ours.get_rbf(dist), pyg(dist), atol=1e-12, rtol=0.0, err_msg="rbf")


def test_sbf_matches_pyg_spherical_basis():
    from torch_geometric.nn.models.dimenet import SphericalBasisLayer as PygSBF

    from enerzyme.models.dimenet.basis import SphericalFourierBesselBasis

    hp = PARITY_HPARAMS
    dtype = torch.float64
    pyg = PygSBF(
        hp["num_spherical"], hp["num_radial"], hp["cutoff"], hp["envelope_exponent"]
    )
    ours = SphericalFourierBesselBasis(
        hp["num_spherical"], hp["num_radial"], hp["cutoff"], hp["envelope_exponent"]
    )
    dist = torch.tensor([0.8, 1.5, 2.2, 3.1], dtype=dtype)
    angle = torch.tensor([0.0, 0.4, 1.1, 2.0, torch.pi], dtype=dtype)
    idx_kj = torch.tensor([0, 1, 1, 2, 3])
    # PyG Bessel zeros are float32; ours are float64 bisection.
    assert_close(
        ours(dist, angle, idx_kj),
        pyg(dist, angle, idx_kj),
        atol=1e-5,
        rtol=1e-5,
        err_msg="sbf",
    )


def test_residual_matches_pyg():
    from torch_geometric.nn.models.dimenet import ResidualLayer as PygResidual

    from enerzyme.models.dimenet.interaction import DimeNetResidualLayer

    dtype = torch.float64
    hidden = PARITY_HPARAMS["hidden_channels"]
    pyg = PygResidual(hidden, pyg_swish()).to(dtype)
    ours = DimeNetResidualLayer(hidden, "swish", dict(SWISH_PARAMS)).to(dtype)
    copy_residual(ours, pyg)
    x = torch.randn(7, hidden, dtype=dtype)
    assert_close(ours(x), pyg(x), atol=1e-12, rtol=1e-12, err_msg="residual")


def test_embedding_block_matches_pyg():
    from torch_geometric.nn.models.dimenet import EmbeddingBlock as PygEmb

    from enerzyme.models.dimenet.interaction import DimeNetEmbeddingBlock

    hp = PARITY_HPARAMS
    dtype = torch.float64
    pyg = PygEmb(hp["num_radial"], hp["hidden_channels"], pyg_swish()).to(dtype)
    ours = DimeNetEmbeddingBlock(
        hp["hidden_channels"], hp["num_radial"], "swish", dict(SWISH_PARAMS)
    ).to(dtype)
    copy_embedding_block(ours, pyg)
    graph = make_parity_graph(dtype=dtype)
    rbf = torch.randn(graph["idx_i"].numel(), hp["num_radial"], dtype=dtype)
    atom = pyg.emb(graph["z"]).to(dtype)
    assert_close(
        ours(atom, rbf, graph["idx_i"], graph["idx_j"]),
        pyg(graph["z"], rbf, graph["idx_i"], graph["idx_j"]),
        atol=1e-12,
        rtol=1e-12,
        err_msg="embedding",
    )


def test_interaction_block_matches_pyg():
    from torch_geometric.nn.models.dimenet import InteractionBlock as PygIB

    from enerzyme.models.dimenet.interaction import DimeNetInteractionBlock

    hp = PARITY_HPARAMS
    dtype = torch.float64
    pyg = PygIB(
        hp["hidden_channels"],
        hp["num_bilinear"],
        hp["num_spherical"],
        hp["num_radial"],
        hp["num_before_skip"],
        hp["num_after_skip"],
        pyg_swish(),
    ).to(dtype)
    ours = DimeNetInteractionBlock(
        dim_embedding=hp["hidden_channels"],
        num_bilinear=hp["num_bilinear"],
        num_sbf=hp["num_spherical"] * hp["num_radial"],
        num_rbf=hp["num_radial"],
        num_before_skip=hp["num_before_skip"],
        num_after_skip=hp["num_after_skip"],
        activation_fn="swish",
        activation_params=dict(SWISH_PARAMS),
    ).to(dtype)
    copy_interaction(ours, pyg)
    num_edges = 6
    num_trip = 4
    x = torch.randn(num_edges, hp["hidden_channels"], dtype=dtype)
    rbf = torch.randn(num_edges, hp["num_radial"], dtype=dtype)
    sbf = torch.randn(num_trip, hp["num_spherical"] * hp["num_radial"], dtype=dtype)
    idx_kj = torch.tensor([0, 1, 2, 4])
    idx_ji = torch.tensor([1, 1, 3, 5])
    assert_close(
        ours(x, rbf, sbf, idx_kj, idx_ji),
        pyg(x, rbf, sbf, idx_kj, idx_ji),
        atol=1e-12,
        rtol=1e-12,
        err_msg="interaction",
    )


def test_output_block_matches_pyg():
    from torch.nn import Sequential
    from torch_geometric.nn.models.dimenet import OutputBlock as PygOut

    from enerzyme.models.blocks.mlp import DenseLayer
    from enerzyme.models.dimenet.interaction import DimeNetAtomReadout

    hp = PARITY_HPARAMS
    dtype = torch.float64
    pyg = PygOut(
        hp["num_radial"],
        hp["hidden_channels"],
        hp["out_channels"],
        hp["num_output_layers"],
        pyg_swish(),
        "glorot_orthogonal",
    ).to(dtype)
    scatter = DimeNetAtomReadout(hp["num_radial"], hp["hidden_channels"]).to(dtype)
    copy_linear_to_dense(scatter.dense_rbf, pyg.lin_rbf)
    layers = [
        DenseLayer(
            hp["hidden_channels"],
            hp["hidden_channels"],
            activation_fn="swish",
            activation_params=dict(SWISH_PARAMS),
            initial_weight="semi_orthogonal_glorot",
        )
        for _ in range(hp["num_output_layers"])
    ]
    layers.append(
        DenseLayer(
            hp["hidden_channels"],
            hp["out_channels"],
            use_bias=False,
            initial_weight="semi_orthogonal_glorot",
        )
    )
    head = Sequential(*layers).to(dtype)
    copy_output_head(head, pyg)
    graph = make_parity_graph(dtype=dtype)
    x = torch.randn(graph["idx_i"].numel(), hp["hidden_channels"], dtype=dtype)
    rbf = torch.randn(graph["idx_i"].numel(), hp["num_radial"], dtype=dtype)
    n = graph["z"].numel()
    ours = head(scatter(x, rbf, graph["idx_i"], n))
    assert_close(
        ours,
        pyg(x, rbf, graph["idx_i"], num_nodes=n),
        atol=1e-12,
        rtol=1e-12,
        err_msg="output",
    )


def test_energy_matches_pyg_with_official_angles(pyg_dimenet):
    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype)
    core, rbf, atom_embedding, readout = make_tiny_enerzyme_stack(dtype=dtype)
    copy_pyg_dimenet_into_enerzyme(core, rbf, atom_embedding, readout, pyg_dimenet)
    pyg_e = pyg_dimenet_energy(
        pyg_dimenet, graph["z"], graph["pos"], graph["idx_i"], graph["idx_j"]
    )
    ours_e = enerzyme_energy(
        core,
        rbf,
        atom_embedding,
        readout,
        graph["z"],
        graph["pos"],
        graph["idx_i"],
        graph["idx_j"],
        angle_at="i",
    )
    assert_close(ours_e, pyg_e, atol=1e-5, rtol=1e-5, err_msg="energy")


def test_force_matches_pyg_with_official_angles(pyg_dimenet):
    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype)
    pos = graph["pos"].clone().requires_grad_(True)
    core, rbf, atom_embedding, readout = make_tiny_enerzyme_stack(dtype=dtype)
    copy_pyg_dimenet_into_enerzyme(core, rbf, atom_embedding, readout, pyg_dimenet)
    pyg_e = pyg_dimenet_energy(
        pyg_dimenet, graph["z"], pos, graph["idx_i"], graph["idx_j"]
    )
    ours_e = enerzyme_energy(
        core,
        rbf,
        atom_embedding,
        readout,
        graph["z"],
        pos,
        graph["idx_i"],
        graph["idx_j"],
        angle_at="i",
    )
    pyg_f = torch.autograd.grad(pyg_e.sum(), pos, create_graph=False)[0]
    ours_f = torch.autograd.grad(ours_e, pos, create_graph=False)[0]
    assert_close(ours_f, pyg_f, atol=1e-4, rtol=1e-4, err_msg="force")


def test_paper_angles_differ_from_pyg_on_bent_molecule(pyg_dimenet):
    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype)
    core, rbf, atom_embedding, readout = make_tiny_enerzyme_stack(dtype=dtype)
    copy_pyg_dimenet_into_enerzyme(core, rbf, atom_embedding, readout, pyg_dimenet)
    kwargs = dict(
        core=core,
        rbf_layer=rbf,
        atom_embedding=atom_embedding,
        readout=readout,
        z=graph["z"],
        pos=graph["pos"],
        idx_i=graph["idx_i"],
        idx_j=graph["idx_j"],
    )
    e_i = enerzyme_energy(**kwargs, angle_at="i")
    e_j = enerzyme_energy(**kwargs, angle_at="j")
    assert not torch.allclose(e_i, e_j, atol=1e-6, rtol=1e-6)


def test_core_get_output_matches_paper_angle_features():
    from enerzyme.models.dimenet.triplets import triplet_angles

    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype)
    core, rbf_layer, atom_embedding, _ = make_tiny_enerzyme_stack(dtype=dtype)
    idx_i, idx_j, pos, z = graph["idx_i"], graph["idx_j"], graph["pos"], graph["z"]
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    rbf = rbf_layer.get_rbf(dist)
    atom = atom_embedding.get_embedding(z)
    out = core.get_output(
        atom_embedding=atom,
        rbf=rbf,
        Dij_sr=dist,
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
    )["atom_feature"]
    idx_kj, idx_ji, _, _, _ = triplet_indices(idx_i, idx_j, z.numel())
    sbf = core.sbf(dist, triplet_angles(vij, idx_kj, idx_ji), idx_kj)
    messages = core.emb_block(atom, rbf, idx_i, idx_j)
    features = [core.atom_readouts[0](messages, rbf, idx_i, z.numel())]
    for block, atom_readout in zip(core.interactions, core.atom_readouts[1:]):
        messages = block(messages, rbf, sbf, idx_kj, idx_ji)
        features.append(atom_readout(messages, rbf, idx_i, z.numel()))
    expected = torch.stack(features, dim=-1)
    assert_close(out, expected, atol=1e-12, rtol=1e-12, err_msg="core get_output")
