"""Algebraic checks for one NewtonNet paper step and the cutoff window."""

from __future__ import annotations

import torch
from torch.nn.functional import silu


def _fill_silu_mlp(mlp, scale=1.0):
    """Make ``mlp(x) = scale * silu(x)`` for a width-1 SiLU MLP."""
    first, _, second = mlp
    with torch.no_grad():
        first.weight.fill_(1.0)
        if first.bias is not None:
            first.bias.zero_()
        second.weight.fill_(scale)
        if second.bias is not None:
            second.bias.zero_()


def test_paper_step_matches_newton_updates():
    from enerzyme.models.newtonnet.interaction import NewtonNetPaperInteraction

    block = NewtonNetPaperInteraction(n_features=1, n_basis=1, displacement_init_gain=1.0)
    _fill_silu_mlp(block.phi_a)
    _fill_silu_mlp(block.phi_f)
    _fill_silu_mlp(block.phi_r)
    _fill_silu_mlp(block.phi_r_ext)
    _fill_silu_mlp(block.phi_u)
    with torch.no_grad():
        block.phi_rbf.weight.fill_(1.0)
        block.phi_rbf.bias.zero_()
        block.phi_force.weight.fill_(2.0)

    # Two atoms on the x axis. Both directed edges are present.
    atom = torch.tensor([[0.5], [1.5]])
    force = torch.zeros(2, 3, 1)
    displacement = torch.zeros(2, 3, 1)
    latent = torch.zeros(2, 3)
    rbf = torch.tensor([[0.4], [0.4]])
    envelope = torch.tensor([[0.5], [0.5]])
    direction = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])

    edge = 0.4 * 0.5
    message = edge * silu(atom[idx_i]) * silu(atom[idx_j])
    atom_msg = atom + torch.zeros_like(atom)
    atom_msg[0] = atom[0] + message[0]
    atom_msg[1] = atom[1] + message[1]
    pair = 2.0 * message * direction
    latent_ref = pair
    force_ref = torch.zeros(2, 3, 1)
    force_ref[0, :, 0] = silu(message[0]) * pair[0]
    force_ref[1, :, 0] = silu(message[1]) * pair[1]
    disp_ref = silu(atom_msg).view(2, 1, 1) * force_ref
    contracted = torch.sum(force_ref * disp_ref, dim=1)
    atom_ref = atom_msg - silu(atom_msg) * contracted

    got_atom, got_force, got_disp, got_latent = block(
        atom, force, displacement, latent, rbf, envelope, direction, idx_i, idx_j, 2
    )
    assert torch.allclose(got_atom, atom_ref, atol=1e-6)
    assert torch.allclose(got_force, force_ref, atol=1e-6)
    assert torch.allclose(got_disp, disp_ref, atol=1e-6)
    assert torch.allclose(got_latent, latent_ref, atol=1e-6)
    assert torch.allclose(got_latent[0], -got_latent[1], atol=1e-6)


def test_repo_step_matches_self_contraction():
    from enerzyme.models.newtonnet.interaction import NewtonNetRepoInteraction

    block = NewtonNetRepoInteraction(n_features=1, n_basis=1)
    _fill_silu_mlp(block.phi_a)
    _fill_silu_mlp(block.equiv_direction, scale=3.0)
    _fill_silu_mlp(block.equiv_neighbor, scale=0.0)
    with torch.no_grad():
        block.phi_rbf.weight.fill_(1.0)
        block.equiv_update.weight.fill_(4.0)

    atom = torch.tensor([[0.25], [-0.5]])
    force = torch.tensor([[[0.0], [1.0], [0.0]], [[0.0], [-1.0], [0.0]]])
    rbf = torch.tensor([[0.2], [0.2]])
    envelope = torch.tensor([[1.0], [1.0]])
    direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])

    edge = 0.2
    message = edge * silu(atom[idx_i]) * silu(atom[idx_j])
    atom_msg = atom.clone()
    atom_msg[0] = atom[0] + message[0]
    atom_msg[1] = atom[1] + message[1]
    directional = 3.0 * silu(message).view(2, 1, 1) * direction.unsqueeze(-1)
    force_ref = force + directional
    # Neighbor gate scale is 0, so silu output is multiplied by 0.
    mixed = 4.0 * force_ref
    atom_ref = atom_msg + torch.sum(force_ref * mixed, dim=1)

    got_atom, got_force = block(
        atom, force, rbf, envelope, direction, idx_i, idx_j, 2
    )
    assert torch.allclose(got_force, force_ref, atol=1e-6)
    assert torch.allclose(got_atom, atom_ref, atol=1e-6)


def test_envelope_matches_paper_polynomial():
    from enerzyme.models.mp.basis.envelope import polynomial_envelope

    dist = torch.tensor([0.0, 2.5, 5.0, 6.0])
    env = polynomial_envelope(dist, 5.0, 7)
    assert abs(env[0].item() - 1.0) < 1e-6
    assert env[2].item() == 0.0
    assert env[3].item() == 0.0
    x = 0.5
    p = 7.0
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    expected = 1 + a * x**p + b * x ** (p + 1) + c * x ** (p + 2)
    assert abs(env[1].item() - expected) < 1e-6
