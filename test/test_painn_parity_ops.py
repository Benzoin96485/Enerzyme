"""Numerical parity of PaiNN blocks with a SchNetPack 2.0.4 transcription.

The reference keeps SchNetPack's dummy feature axis on scalars, ``[N, 1, F]``,
and ``index_add`` aggregation. Enerzyme uses ``[N, F]`` and ``scatter_sum``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _scatter(x: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[0] = dim_size
    out = x.new_zeros(shape)
    return out.index_add(0, idx, x)


class _RefInteraction(nn.Module):
    """schnetpack.representation.painn.PaiNNInteraction (2.0.4)."""

    def __init__(self, n_atom_basis: int) -> None:
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis),
        )

    def forward(self, q, mu, wij, dir_ij, idx_i, idx_j, n_atoms):
        x = self.interatomic_context_net(q)
        x = wij * x[idx_j]
        dq, dmu_r, dmu_mu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = _scatter(dq, idx_i, n_atoms)
        dmu = dmu_r * dir_ij[..., None] + dmu_mu * mu[idx_j]
        dmu = _scatter(dmu, idx_i, n_atoms)
        return q + dq, mu + dmu


class _RefMixing(nn.Module):
    """schnetpack.representation.painn.PaiNNMixing (2.0.4)."""

    def __init__(self, n_atom_basis: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon
        self.intraatomic_context_net = nn.Sequential(
            nn.Linear(2 * n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis),
        )
        self.mu_channel_mix = nn.Linear(n_atom_basis, 2 * n_atom_basis, bias=False)

    def forward(self, q, mu):
        mixed = self.mu_channel_mix(mu)
        mu_v, mu_w = torch.split(mixed, self.n_atom_basis, dim=-1)
        mu_vn = torch.sqrt(torch.sum(mu_v**2, dim=-2, keepdim=True) + self.epsilon)
        x = self.intraatomic_context_net(torch.cat([q, mu_vn], dim=-1))
        dq, dmu_scale, dq_prod = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = dmu_scale * mu_w
        dq_prod = dq_prod * torch.sum(mu_v * mu_w, dim=1, keepdim=True)
        return q + dq + dq_prod, mu + dmu


class _RefGated(nn.Module):
    """schnetpack.nn.equivariant.GatedEquivariantBlock (2.0.4)."""

    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, sactivation: bool) -> None:
        super().__init__()
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.mix_vectors = nn.Linear(n_vin, 2 * n_vout, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(n_sin + n_vout, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_sout + n_vout),
        )
        self.sactivation = nn.SiLU() if sactivation else None

    def forward(self, inputs):
        scalars, vectors = inputs
        mixed = self.mix_vectors(vectors)
        vectors_v, vectors_w = torch.split(mixed, self.n_vout, dim=-1)
        vectors_vn = torch.linalg.vector_norm(vectors_v, dim=-2)
        x = self.scalar_net(torch.cat([scalars, vectors_vn], dim=-1))
        s_out, gate = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = gate.unsqueeze(-2) * vectors_w
        if self.sactivation is not None:
            s_out = self.sactivation(s_out)
        return s_out, v_out


def _copy_dense(dst, src) -> None:
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is None:
        assert src.bias is None
    else:
        dst.bias.data.copy_(src.bias.data)


def test_cartesian_dense_is_silu_and_keeps_vector_axis():
    from enerzyme.models.layers.gated_equivariant import cartesian_dense

    layer = cartesian_dense(4, 6, activation=True).double()
    x = torch.randn(5, 3, 4, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (5, 3, 6)
    assert torch.allclose(y, F.silu(F.linear(x, layer.weight, layer.bias)))


def test_interaction_mixing_and_core_match_schnetpack_layout():
    from enerzyme.models.painn.core import PaiNNCore
    from enerzyme.models.painn.interaction import PaiNNInteraction, PaiNNMixing

    torch.manual_seed(0)
    n_atoms, n_basis, n_rbf, n_layers = 6, 5, 4, 2
    idx_i, idx_j = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    pos = torch.randn(n_atoms, 3, dtype=torch.float64)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1, keepdim=True)
    direction = vij / dist
    rbf = torch.randn(idx_i.numel(), n_rbf, dtype=torch.float64)
    cutoff = (0.5 * (torch.cos(dist.squeeze(-1) * torch.pi / 5.0) + 1.0)).unsqueeze(-1)
    q0 = torch.randn(n_atoms, n_basis, dtype=torch.float64)
    mu0 = torch.randn(n_atoms, 3, n_basis, dtype=torch.float64)

    ref_filter = nn.Linear(n_rbf, n_layers * 3 * n_basis).double()
    our_filter_owner = PaiNNCore(
        dim_embedding=n_basis,
        num_rbf=n_rbf,
        num_interactions=n_layers,
        cutoff_sr=5.0,
    ).double()
    _copy_dense(our_filter_owner.filter_net, ref_filter)

    ref_interactions = nn.ModuleList([_RefInteraction(n_basis).double() for _ in range(n_layers)])
    ref_mixings = nn.ModuleList([_RefMixing(n_basis).double() for _ in range(n_layers)])
    our_interactions = nn.ModuleList(
        [PaiNNInteraction(n_basis).double() for _ in range(n_layers)]
    )
    our_mixings = nn.ModuleList([PaiNNMixing(n_basis).double() for _ in range(n_layers)])
    for ref_block, our_block, ref_mix, our_mix in zip(
        ref_interactions, our_interactions, ref_mixings, our_mixings
    ):
        _copy_dense(our_block.interatomic_context_net[0], ref_block.interatomic_context_net[0])
        _copy_dense(our_block.interatomic_context_net[1], ref_block.interatomic_context_net[2])
        _copy_dense(our_mix.intraatomic_context_net[0], ref_mix.intraatomic_context_net[0])
        _copy_dense(our_mix.intraatomic_context_net[1], ref_mix.intraatomic_context_net[2])
        _copy_dense(our_mix.mu_channel_mix, ref_mix.mu_channel_mix)

    filters = ref_filter(rbf) * cutoff
    width = 3 * n_basis
    q_ref = q0[:, None, :]
    mu_ref = mu0.clone()
    q = q0.clone()
    mu = mu0.clone()
    for step, (ref_block, our_block, ref_mix, our_mix) in enumerate(
        zip(ref_interactions, our_interactions, ref_mixings, our_mixings)
    ):
        wij = filters[:, step * width : (step + 1) * width]
        q_ref, mu_ref = ref_block(
            q_ref, mu_ref, wij[:, None, :], direction, idx_i, idx_j, n_atoms
        )
        q_ref, mu_ref = ref_mix(q_ref, mu_ref)
        q, mu = our_block(q, mu, wij, direction, idx_i, idx_j, n_atoms)
        q, mu = our_mix(q, mu)
        assert torch.allclose(q, q_ref.squeeze(1), atol=1e-8, rtol=1e-8)
        assert torch.allclose(mu, mu_ref, atol=1e-8, rtol=1e-8)

    # Wire the same weights through PaiNNCore, including its filter and cutoff.
    for step in range(n_layers):
        _copy_dense(
            our_filter_owner.interactions[step].interatomic_context_net[0],
            ref_interactions[step].interatomic_context_net[0],
        )
        _copy_dense(
            our_filter_owner.interactions[step].interatomic_context_net[1],
            ref_interactions[step].interatomic_context_net[2],
        )
        _copy_dense(
            our_filter_owner.mixings[step].intraatomic_context_net[0],
            ref_mixings[step].intraatomic_context_net[0],
        )
        _copy_dense(
            our_filter_owner.mixings[step].intraatomic_context_net[1],
            ref_mixings[step].intraatomic_context_net[2],
        )
        _copy_dense(
            our_filter_owner.mixings[step].mu_channel_mix, ref_mixings[step].mu_channel_mix
        )
    # Restart the reference from the embedding with zero vectors, as the Core does.
    q_ref = q0[:, None, :]
    mu_ref = torch.zeros_like(mu0)
    owned_filters = (our_filter_owner.filter_net(rbf) * cutoff).split(width, dim=-1)
    for step, (ref_block, ref_mix) in enumerate(zip(ref_interactions, ref_mixings)):
        q_ref, mu_ref = ref_block(
            q_ref,
            mu_ref,
            owned_filters[step][:, None, :],
            direction,
            idx_i,
            idx_j,
            n_atoms,
        )
        q_ref, mu_ref = ref_mix(q_ref, mu_ref)
    out = our_filter_owner.get_output(
        atom_embedding=q0,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=dist.squeeze(-1),
        cutoff_values_sr=cutoff.squeeze(-1),
    )
    assert torch.allclose(out["atom_feature"], q_ref.squeeze(1), atol=1e-8, rtol=1e-8)
    assert torch.allclose(out["atom_vector_feature"], mu_ref, atol=1e-8, rtol=1e-8)


def test_gated_block_and_mlp_match_schnetpack():
    from enerzyme.models.layers.gated_equivariant import (
        GatedEquivariantBlock,
        build_gated_equivariant_mlp,
    )

    torch.manual_seed(1)
    n, f_in, f_out, hidden = 7, 8, 3, 5
    scalars = torch.randn(n, f_in, dtype=torch.float64)
    vectors = torch.randn(n, 3, f_in, dtype=torch.float64)
    ref = _RefGated(f_in, f_in, f_out, f_out, hidden, sactivation=True).double()
    ours = GatedEquivariantBlock(
        f_in, f_in, f_out, f_out, hidden, sactivation=True
    ).double()
    _copy_dense(ours.mix_vectors, ref.mix_vectors)
    _copy_dense(ours.scalar_net[0], ref.scalar_net[0])
    _copy_dense(ours.scalar_net[1], ref.scalar_net[2])
    s_ref, v_ref = ref((scalars, vectors))
    s_our, v_our = ours((scalars, vectors))
    assert torch.allclose(s_our, s_ref, atol=1e-8, rtol=1e-8)
    assert torch.allclose(v_our, v_ref, atol=1e-8, rtol=1e-8)

    ref_blocks = [
        _RefGated(8, 8, 4, 4, 8, True).double(),
        _RefGated(4, 4, 1, 1, 4, False).double(),
    ]
    mlp = build_gated_equivariant_mlp(n_in=8, n_out=1, n_layers=2).double()
    assert len(list(mlp)) == 2
    for ref_block, our_block in zip(ref_blocks, mlp):
        _copy_dense(our_block.mix_vectors, ref_block.mix_vectors)
        _copy_dense(our_block.scalar_net[0], ref_block.scalar_net[0])
        _copy_dense(our_block.scalar_net[1], ref_block.scalar_net[2])
    s_in = torch.randn(n, 8, dtype=torch.float64)
    v_in = torch.randn(n, 3, 8, dtype=torch.float64)
    s_ref, v_ref = s_in, v_in
    for block in ref_blocks:
        s_ref, v_ref = block((s_ref, v_ref))
    s_our, v_our = mlp((s_in, v_in))
    assert s_our.shape == (n, 1)
    assert v_our.shape == (n, 3, 1)
    assert torch.allclose(s_our, s_ref, atol=1e-8, rtol=1e-8)
    assert torch.allclose(v_our, v_ref, atol=1e-8, rtol=1e-8)


def test_dipole_head_matches_charge_plus_atomic_dipole_formula():
    from enerzyme.models.layers.tensorial import EquivariantDipoleLayer

    torch.manual_seed(2)
    n = 4
    layer = EquivariantDipoleLayer(
        dim_embedding=4, n_layers=1, return_charges=True, correct_charges=False
    ).double()

    class _Fixed(nn.Module):
        def __init__(self, charges, dipoles):
            super().__init__()
            self.charges = charges
            self.dipoles = dipoles

        def forward(self, inputs):
            return self.charges, self.dipoles

    charges = torch.tensor([0.2, -0.1, 0.4, -0.5], dtype=torch.float64)
    dipoles = torch.randn(n, 3, dtype=torch.float64)
    layer.outnet = _Fixed(charges[:, None], dipoles[:, :, None])
    pos = torch.randn(n, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1])
    out = layer.get_output(
        atom_feature=torch.randn(n, 4, dtype=torch.float64),
        atom_vector_feature=torch.randn(n, 3, 4, dtype=torch.float64),
        Ra=pos,
        batch_seg=batch,
    )
    moment = pos * charges[:, None] + dipoles
    expected = torch.stack([moment[:2].sum(0), moment[2:].sum(0)])
    assert torch.allclose(out["M2"], expected, atol=1e-8, rtol=1e-8)
    assert torch.allclose(out["Qa"], charges, atol=1e-8, rtol=1e-8)
