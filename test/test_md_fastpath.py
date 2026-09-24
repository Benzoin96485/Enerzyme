import numpy as np
import pytest
import torch
from ase import Atoms

from enerzyme.data.neighbor_list import (
    full_neighbor_list, full_neighbor_list_torch, radius_neighbor_list, VerletNeighborList
)
from enerzyme.data.transform import Transform
from enerzyme.models.functional import segment_sum_coo
from enerzyme.models.layers import ElectrostaticEnergyLayer
from enerzyme.models.layers.electrostatics import DenseCoulombPotential
from enerzyme.models.ff import build_model
from enerzyme.tasks.calculator import ASECalculator, get_dense_lr_cutoff


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
CUTOFF_SR = 5.291772108


def random_positions(N, box=14.0, min_dist=0.8, seed=0):
    rng = np.random.default_rng(seed)
    R = []
    while len(R) < N:
        x = rng.uniform(0, box, 3)
        if all(np.linalg.norm(x - y) > min_dist for y in R):
            R.append(x)
    return np.array(R)


def brute_force_pairs(R, cutoff, batch_seg=None):
    pairs = []
    for i in range(len(R)):
        for j in range(len(R)):
            if i != j and np.linalg.norm(R[i] - R[j]) < cutoff:
                if batch_seg is None or batch_seg[i] == batch_seg[j]:
                    pairs.append((i, j))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


@pytest.mark.parametrize("N", [1, 2, 3, 17])
def test_full_neighbor_list_torch(N):
    idx_i, idx_j = full_neighbor_list(N)
    t_i, t_j = full_neighbor_list_torch(N, DEVICE)
    np.testing.assert_array_equal(t_i.cpu().numpy(), idx_i)
    np.testing.assert_array_equal(t_j.cpu().numpy(), idx_j)


@pytest.mark.parametrize("max_block_elements", [1 << 24, 1000])
def test_radius_neighbor_list(max_block_elements):
    R = random_positions(150)
    batch_seg = np.repeat([0, 1], [70, 80])
    Ra = torch.tensor(R, dtype=DTYPE, device=DEVICE)
    for bs in (None, batch_seg):
        ref = brute_force_pairs(R, 4.0, bs)
        idx_i, idx_j = radius_neighbor_list(
            Ra, 4.0, None if bs is None else torch.tensor(bs, device=DEVICE), max_block_elements
        )
        # same pairs and the same i-major/j-minor order as the filtered full list
        np.testing.assert_array_equal(torch.stack([idx_i, idx_j], 1).cpu().numpy(), ref)


def test_verlet_neighbor_list_is_superset():
    R = random_positions(120)
    Ra = torch.tensor(R, dtype=DTYPE, device=DEVICE)
    nl = VerletNeighborList(4.0, skin=1.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = R + rng.normal(scale=0.05, size=R.shape)
        Ra = torch.tensor(R, dtype=DTYPE, device=DEVICE)
        idx_i, idx_j = nl(Ra)
        got = set(zip(idx_i.tolist(), idx_j.tolist()))
        assert set(map(tuple, brute_force_pairs(R, 4.0))) <= got
    assert nl.num_builds < 20


def reference_potential(Ra, Q, batch_seg):
    N = Ra.shape[0]
    idx_i, idx_j = full_neighbor_list_torch(N, Ra.device)
    if batch_seg is not None:
        keep = batch_seg[idx_i] == batch_seg[idx_j]
        idx_i, idx_j = idx_i[keep], idx_j[keep]
    r = (Ra[idx_i] - Ra[idx_j]).norm(dim=-1)
    Qj = Q[idx_j]
    w = (1 / r).view((-1,) + (1,) * (Q.dim() - 1))
    return segment_sum_coo(Qj * w, idx_i, dim_size=N)


@pytest.mark.parametrize("q_shape", [(), (4,)])
@pytest.mark.parametrize("batched", [False, True])
def test_dense_coulomb_potential(q_shape, batched):
    N = 90
    R = torch.tensor(random_positions(N), dtype=DTYPE, device=DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(0)
    Q0 = torch.randn((N,) + q_shape, generator=gen, dtype=DTYPE).to(DEVICE)
    G = torch.randn((N,) + q_shape, generator=gen, dtype=DTYPE).to(DEVICE)
    batch_seg = torch.tensor(np.repeat([0, 1], [40, 50]), device=DEVICE) if batched else None
    outputs = []
    for fn in (
        lambda R, Q: reference_potential(R, Q, batch_seg),
        lambda R, Q: DenseCoulombPotential.apply(R, Q, batch_seg, 7),
    ):
        Ra = R.clone().requires_grad_(True)
        Q = Q0.clone().requires_grad_(True)
        V = fn(Ra, Q)
        gR, gQ = torch.autograd.grad((V * G).sum(), (Ra, Q))
        outputs.append((V.detach(), gR, gQ))
    for ref, got in zip(*outputs):
        torch.testing.assert_close(got, ref, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("q_shape", [(), (10,)])
@pytest.mark.parametrize("cutoff_fn", ["smooth", "bump"])
def test_electrostatics_dense_equals_pairs(q_shape, cutoff_fn):
    N = 200
    R = torch.tensor(random_positions(N, box=16.0), dtype=DTYPE, device=DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(2)
    Q0 = (0.3 * torch.randn((N,) + q_shape, generator=gen, dtype=DTYPE)).to(DEVICE)
    layer = ElectrostaticEnergyLayer(CUTOFF_SR, dielectric_constant=10.0, cutoff_fn=cutoff_fn).eval()
    results = {}
    for mode in ("pairs", "dense"):
        layer.lr_mode = mode
        Ra = R.clone().requires_grad_(True)
        Qa = Q0.clone().requires_grad_(True)
        if mode == "pairs":
            idx_i, idx_j = full_neighbor_list_torch(N, DEVICE)
        else:
            idx_i, idx_j = radius_neighbor_list(Ra, CUTOFF_SR + 1e-3)
        Dij = torch.nn.functional.pairwise_distance(Ra[idx_i], Ra[idx_j], eps=1e-15)
        E = layer.get_E_ele_a(Dij, Qa, idx_i, idx_j, Ra, None)
        gR, gQ = torch.autograd.grad(E.sum(), (Ra, Qa))
        results[mode] = (E.detach(), gR, gQ)
    for ref, got in zip(results["pairs"], results["dense"]):
        torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


def test_dense_electrostatics_refuses_training():
    layer = ElectrostaticEnergyLayer(CUTOFF_SR).train()
    layer.lr_mode = "dense"
    R = torch.tensor(random_positions(10), dtype=DTYPE)
    idx_i, idx_j = radius_neighbor_list(R, CUTOFF_SR)
    with pytest.raises(RuntimeError):
        layer.get_E_ele_a((R[idx_i] - R[idx_j]).norm(dim=-1), torch.zeros(10, dtype=DTYPE), idx_i, idx_j, R, None)


def small_spookynet(extra_layers=(), shallow_ensemble_size=4):
    build_params = {
        "Bohr_in_R": 0.5291772108, "Hartree_in_E": 1, "activation_fn": "swish",
        "activation_params": {"learnable": True}, "cutoff_sr": CUTOFF_SR,
        "dim_embedding": 32, "max_Za": 86, "num_rbf": 8,
    }
    residual = {k: 1 for k in (
        "num_residual_local", "num_residual_local_d", "num_residual_local_p", "num_residual_local_s",
        "num_residual_local_x", "num_residual_nonlocal_k", "num_residual_nonlocal_q",
        "num_residual_nonlocal_v", "num_residual_output", "num_residual_post", "num_residual_pre",
    )}
    layers = [
        {"name": "RangeSeparation", "params": {"cutoff_fn": "bump"}},
        {"name": "ExponentialBernsteinRBF", "params": {"exp_weighting": False, "init_alpha": 0.944863062918464, "learnable_shape": True, "no_basis_at_infinity": False}},
        {"name": "NuclearEmbedding", "params": {"use_electron_config": True, "zero_init": False}},
        {"name": "ElectronicEmbedding", "params": {"attribute": "charge", "num_residual": 1}},
        {"name": "ElectronicEmbedding", "params": {"attribute": "spin", "num_residual": 1}},
        {"name": "Core", "params": {"dropout_rate": 0.0, "num_modules": 2, "shallow_ensemble_size": shallow_ensemble_size, "use_irreps": True, **residual}},
        {"name": "AtomicAffine", "params": {"scales": {"Ea": {"learnable": True, "values": 1}, "Qa": {"learnable": True, "values": 1}}, "shifts": {"Ea": {"learnable": True, "values": 0}, "Qa": {"learnable": True, "values": 0}}}},
        {"name": "ChargeConservation"},
        *extra_layers,
        {"name": "ElectrostaticEnergy", "params": {"dielectric_constant": 10.0, "flavor": "SpookyNet"}},
        {"name": "AtomicCharge2Dipole"},
        {"name": "EnergyReduce"},
        {"name": "ShallowEnsembleReduce", "params": {"reduce_mean": ["E", "Qa", "M2", "Q"], "train_only": True, "var": ["E"]}},
        {"name": "Force"},
        {"name": "ShallowEnsembleReduce", "params": {"eval_only": True, "reduce_mean": ["E", "Qa", "M2", "Q", "Fa"], "var": ["E", "Fa"]}},
    ]
    torch.manual_seed(0)
    model = build_model("SpookyNet", layers, build_params, verbose=0).to(DEVICE).type(DTYPE)
    model.eval()
    return model


def test_dense_lr_whitelist():
    assert get_dense_lr_cutoff(small_spookynet()) == pytest.approx(CUTOFF_SR)
    with_d3 = small_spookynet(extra_layers=[{"name": "GrimmeD3Energy", "params": {"learnable": True}}])
    assert get_dense_lr_cutoff(with_d3) is None


def test_calculator_fast_path_matches_full_pairs():
    R = random_positions(120, box=13.0, min_dist=0.9)
    Z = np.random.default_rng(3).choice([1, 6, 7, 8, 16], size=len(R))
    model = small_spookynet()
    results = {}
    for nl_type, skin in (("full_pairs", 0.0), ("full", 0.0), ("full", 1.0)):
        calc = ASECalculator(model=model, device=DEVICE, dtype=DTYPE, transform=Transform(None), neighbor_list_type=nl_type, neighbor_list_skin=skin)
        atoms = Atoms(numbers=Z, positions=R)
        atoms.info.update({"charge": -1, "spin": 1})
        calc.calculate(atoms, properties=["energy", "forces", "dipole", "charges"])
        results[(nl_type, skin)] = {k: np.array(v) for k, v in calc.results.items()}
        assert calc.lr_mode == ("pairs" if nl_type == "full_pairs" else "dense")
    ref = results[("full_pairs", 0.0)]
    for key in (("full", 0.0), ("full", 1.0)):
        for prop in ("energy", "forces", "dipole", "charges"):
            np.testing.assert_allclose(results[key][prop], ref[prop], rtol=1e-10, atol=1e-10)


def test_mean_force_equals_mean_of_member_forces():
    model = small_spookynet()
    force_layer = [m for m in model.modules() if type(m).__name__ == "ForceLayer"][0]
    R = random_positions(60, box=10.0, min_dist=0.9)
    N = len(R)
    idx_i, idx_j = full_neighbor_list_torch(N, DEVICE)
    outputs = {}
    for mean_only in (False, True):
        force_layer.ensemble_mean_only = mean_only
        net_input = {
            "Ra": torch.tensor(R, dtype=DTYPE, device=DEVICE, requires_grad=True),
            "Za": torch.tensor(np.full(N, 6), device=DEVICE),
            "Q": torch.zeros(1, dtype=DTYPE, device=DEVICE),
            "S": torch.zeros(1, dtype=DTYPE, device=DEVICE),
            "N": [N], "batch_seg": torch.zeros(N, dtype=torch.long, device=DEVICE),
            "idx_i": idx_i, "idx_j": idx_j,
        }
        outputs[mean_only] = model(net_input)
    torch.testing.assert_close(outputs[True]["Fa"], outputs[False]["Fa"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(outputs[True]["E"], outputs[False]["E"], rtol=1e-12, atol=1e-12)
    assert "Fa_var" in outputs[False] and "Fa_var" not in outputs[True]
    assert "E_var" in outputs[True]
